import torch
from torch import Tensor
import torch.nn.functional as f
from nn.transformers.utils import position_embedding, feed_forward
from nn.transformers.utils import AttentionHead, MultiHeadAttention, Residual

# https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self, 
        num_layers: int = 6,
        dim_feature: int = 9,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.feature2hidden = torch.nn.Linear(dim_feature, dim_model)
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        if src_mask is not None:
            src = torch.mul(src, src_mask)
        #import pdb; pdb.set_trace()
        src = self.feature2hidden(src)
        seq_len, dimension = src.size(1), src.size(2)
        src += position_embedding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)


class GeneralTransformerDecoder(torch.nn.Module):
    def __init__(
        self, 
        num_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        no_generated_mask: bool = False,
        use_prob_mask: bool = False,
        linear: torch.nn.Module = None,
        linear_m: torch.nn.Module = None
    ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.use_prob_mask = use_prob_mask
        self.max_length = max_length
        self.feature_size = dim_feature
        self.dim_model = dim_model
        self.no_generated_mask = no_generated_mask
        self.feature2hidden = torch.nn.Linear(dim_feature, dim_model)
        self.layers = torch.nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        if linear is not None:
            self.linear = linear
        else:
            self.linear = torch.nn.Linear(dim_model, dim_feature)
        if no_generated_mask == False:
            if linear_m is not None:
                self.linear_m = linear_m
            else:
                self.linear_m = torch.nn.Linear(dim_model, dim_feature)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None) -> Tensor:
        if tgt_mask is not None:
            tgt = torch.mul(tgt, tgt_mask)
        tgt = self.feature2hidden(tgt)

        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_embedding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        output = torch.nn.functional.sigmoid(self.linear(tgt))
        
        if tgt_mask is None or self.use_prob_mask: # if use_prob_mask, no need to generate mask
            return output, None
        if self.no_generated_mask == False:
            mask =  torch.nn.functional.sigmoid(self.linear_m(tgt))
        else:
            mask = None
        return output, mask


    def inference(self, start_feature: Tensor, memory: Tensor, start_mask: Tensor = None, prob_mask: Tensor = None):
        if self.use_prob_mask:
            assert prob_mask is not None and start_mask is not None
        
        start_feature = start_feature.cuda()
        memory = memory.cuda()
        z = memory
        batch_size = z.size(0)
        zs = torch.unbind(z, dim=1) 
        
        
        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        sequence_length = torch.zeros(batch_size, out=self.tensor()).long()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_length, self.feature_size).fill_(0.0).float()
        gen_masks = self.tensor(batch_size, self.max_length, self.feature_size).fill_(0.0).float()
        t=0
        pos_emb = position_embedding(self.max_length, self.dim_model)[0]
        while(t<self.max_length and len(running_seqs)>0):
            batch_size = z.size(0)
            z_ = z[:, :t+1, :] # [batch, t, fea]
            #import pdb; pdb.set_trace()

            if t == 0:
                # input for time step 0
                input_sequence = start_feature.float().cuda() # [batch, feature_size]
                # save next input
                #generations = self._save_sample(generations, input_sequence, sequence_running, 0, add_grad=True)
                
                input_sequence = input_sequence.unsqueeze(dim=1)
                fisrt_input = input_sequence
                if start_mask is None:
                    input_mask = None
                else:
                    input_mask = start_mask.float().cuda() # [batch, feature_size]
                    # save next input
                    #gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, 0, add_grad=True)

                    input_mask = input_mask.unsqueeze(dim=1) # [batch, 1, fea]
                    fisrt_mask = input_mask
                        
            input_ = input_sequence
            if input_mask is not None:
                input_ = torch.mul(input_, input_mask)
                
            #import pdb; pdb.set_trace()
            input_ = self.feature2hidden(input_)

            #input_batch_size, seq_len, dimension = input_.size(0), input_.size(1), input_.size(2)
            input_batch_size = input_.size(0)
            if input_batch_size != batch_size:
                import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()
            input_ += pos_emb[:t+1] # position_embedding(seq_len, dimension)
            for layer in self.layers:
                try:
                    input_ = layer(input_, z_)
                except:
                    import pdb; pdb.set_trace()


            gen_input = torch.nn.functional.sigmoid(self.linear(input_))
            last_input = gen_input[:, -1, :]
            # concat
            input_sequence = torch.cat([fisrt_input, gen_input], dim=1)
            # save next input
            #generations = self._save_sample(generations, last_input, sequence_running, t+1, add_grad=True)
            
            #import pdb; pdb.set_trace()
            #
            if input_mask is not None:
                if self.use_prob_mask:
                    input_mask = sample_mask_from_prob(prob_mask, input_mask.shape[0], input_mask.shape[1])
                    input_mask = input_mask.squeeze(dim=1)
                else:
                    # generate mask
                    # decoder forward pass
                    gen_mask = torch.nn.functional.sigmoid(self.linear_m(input_))
                    last_mask =  gen_mask[:, -1, :]
                    # save next input
                    input_mask = torch.cat([fisrt_mask, gen_mask], dim=1)
                    #gen_masks = self._save_sample(gen_masks, last_mask, sequence_running, t+1, add_grad=True)
            
            #
            # update gloabl running sequence
            sequence_length[sequence_running] += 1
            sequence_mask[sequence_running] = (last_input.sum(dim=1) > 0).data #
            sequence_running = sequence_idx.masked_select(sequence_mask)
            #
            sequence_ended = sequence_idx.masked_select(torch.logical_not(sequence_mask))

            # update local running sequences
            running_mask = (last_input.sum(dim=1) > 0).data # ??
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                z = z[running_seqs]
                if input_mask is not None:
                    input_mask = input_mask[running_seqs]
                # save
                generations = self._save_sample(generations, input_sequence, sequence_running, t, add_grad=True)
                if input_mask is not None:
                    gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, t, add_grad=True)

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1
        if len(sequence_running) > 0:
            generations = self._save_sample(generations, input_sequence, sequence_running, t, add_grad=True)
            if input_mask is not None:
                gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, t, add_grad=True)
        #import pdb; pdb.set_trace()
        output = generations
        
        if start_mask is None or self.use_prob_mask:
            return output, None
        
        mask = gen_masks

        return output, mask
        

    def _save_sample(self, save_to, sample, running_seqs, t, add_grad=False):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        tgt_sample = sample[:, 1:, :]
        if add_grad:
            running_latest[:,:t+1,:] = torch.tensor(tgt_sample.data, requires_grad=True)
        else:
            running_latest[:,:t+1,:] = tgt_sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


    def compute_mask_loss(self, output_mask, mask, type="xent"):
        if type == "mse":
            loss = torch.mean(torch.sum(torch.square(output_mask-mask), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(mask, torch.log(output_mask + 1e-12)) + torch.mul((1-mask), torch.log((1-output_mask + 1e-12)))
            loss = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))
        else:
            raise "Wrong loss type"
        return loss
    

    def compute_recon_loss(self, output, target, output_mask=None, mask=None, type="xent"):
        if type == "mse":
            loss_1 = torch.mean(torch.sum(torch.square(output-target), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(target, torch.log(output + 1e-12)) + torch.mul((1-target), torch.log((1-output + 1e-12)))
            loss_1 = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))
        else:
            raise "Wrong loss type"
        if output_mask is not None:
            output = torch.mul(output, output_mask)
        if mask is not None:
            target = torch.mul(target, mask)
        if type == "mse":
            loss_2 = torch.mean(torch.sum(torch.square(output-target), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(target, torch.log(output + 1e-12)) + torch.mul((1-target), torch.log((1-output + 1e-12)))
            loss_2 = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))  
        else:
            raise "Wrong loss type"
 
        return (loss_1 + loss_2)/2
    


class TransformerDecoder(GeneralTransformerDecoder):
    def __init__(
        self, 
        num_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        no_generated_mask: bool = False,
        use_prob_mask: bool = False,
    ):
        super().__init__(
            num_layers=num_layers,
            max_length=max_length,
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            no_generated_mask=no_generated_mask,
            use_prob_mask=use_prob_mask
        )
    

class Transformer(torch.nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        encoder_dropout: float = 0.1, 
        decoder_dropout: float = 0.1, 
        activation: torch.nn.Module = torch.nn.ReLU(),
        no_generated_mask: bool = False,
        use_prob_mask: bool = False,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            max_length=max_length,
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
            no_generated_mask=no_generated_mask,
            use_prob_mask=use_prob_mask
        )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        src = src.float(); 
        tgt = tgt.float();
        if src_mask is not None: src_mask = src_mask.float()
        if tgt_mask is not None: tgt_mask = tgt_mask.float()
        #import pdb; pdb.set_trace()
        memory = self.encoder(src, src_mask)
        output, mask = self.decoder(tgt, memory, tgt_mask)
        # build a target prob tensor
        if tgt_mask is None:
            p_input = tgt
        else:
            p_input = torch.mul(tgt, tgt_mask)
        return memory, p_input, output, mask


    def compute_mask_loss(self, output_mask, mask, type="xent"):
        if type == "mse":
            loss = torch.mean(torch.sum(torch.square(output_mask-mask), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(mask, torch.log(output_mask + 1e-12)) + torch.mul((1-mask), torch.log((1-output_mask + 1e-12)))
            loss = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))
        else:
            raise "Wrong loss type"
        return loss
    

    def compute_recon_loss(self, output, target, output_mask=None, mask=None, type="xent"):
        if type == "mse":
            loss_1 = torch.mean(torch.sum(torch.square(output-target), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(target, torch.log(output + 1e-12)) + torch.mul((1-target), torch.log((1-output + 1e-12)))
            loss_1 = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))
        else:
            raise "Wrong loss type"
        if output_mask is not None:
            output = torch.mul(output, output_mask)
        if mask is not None:
            target = torch.mul(target, mask)
        if type == "mse":
            loss_2 = torch.mean(torch.sum(torch.square(output-target), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(target, torch.log(output + 1e-12)) + torch.mul((1-target), torch.log((1-output + 1e-12)))
            loss_2 = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))  
        else:
            raise "Wrong loss type"
 
        return (loss_1 + loss_2)/2