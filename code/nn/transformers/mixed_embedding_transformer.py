import torch
from torch import Tensor
import torch.nn.functional as f
from utils.train_utils import descale_time, extract_day_hour_from_time
from nn.transformers.utils import position_embedding, feed_forward
from nn.transformers.utils import AttentionHead, MultiHeadAttention, Residual

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1
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


    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.attention(src, src, src)
        src = self.feed_forward(src)
        return src, src_mask


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self, 
        time_embedding: Tensor = None,
        num_layers: int = 6,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        if time_embedding is None:
            time_embedding = torch.nn.Embedding(dim_time, dim_model) # an embedding lookup dict with key range from 0 to dim_time-1
        
        gender_embedding = kwargs.get("gender_embedding", None)
        dim_gender = kwargs.get("dim_gender", None)
        race_embedding = kwargs.get("race_embedding", None)
        dim_race = kwargs.get("dim_race", None)

        day_embedding = kwargs.get("day_embedding", None)
        dim_day = kwargs.get("dim_day", None)
        hour_embedding = kwargs.get("hour_embedding", None)
        dim_hour = kwargs.get("dim_hour", None)

        if gender_embedding is None:
            if dim_gender is None: gender_embedding = None
            else:
                gender_embedding = torch.nn.Embedding(dim_gender, dim_model)

        if race_embedding is None:
            if dim_race is None: race_embedding = None
            else:
                race_embedding = torch.nn.Embedding(dim_race, dim_model)
        
        if day_embedding is None:
            if dim_day is None: day_embedding = None
            else:
                day_embedding = torch.nn.Embedding(dim_day, dim_model)

        if hour_embedding is None:
            if dim_hour is None: hour_embedding = None
            else:
                hour_embedding = torch.nn.Embedding(dim_hour, dim_model)
            
        self.time_embedding = time_embedding
        self.gender_embedding = gender_embedding
        self.race_embedding = race_embedding
        self.day_embedding = day_embedding
        self.hour_embedding = hour_embedding
        self.feature2hidden = torch.nn.Linear(dim_feature, dim_model)
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, time: Tensor, mask: Tensor = None, ava: Tensor = None, **kwargs) -> Tensor:
        #import pdb; pdb.set_trace()
        if ava is not None:
            src = torch.mul(src, ava)
        if mask is not None:
            src = torch.mul(src, mask)
        src = self.feature2hidden(src)
        seq_len, dimension = src.size(1), src.size(2)
        src += position_embedding(seq_len, dimension)
        src += self.time_embedding(time)
        #
        gender = kwargs.get("gender", None)
        race = kwargs.get("race", None)
        day = kwargs.get("day", None)
        hour = kwargs.get("hour", None)

        if gender is not None: src += self.gender_embedding(gender)
        if race is not None: src += self.race_embedding(race)
        if day is not None: src += self.day_embedding(day)
        if hour is not None: src += self.hour_embedding(hour)

        for layer in self.layers:
            src, mask = layer(src, mask)

        return src


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1
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


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        tgt = self.feed_forward(tgt)
        
        return tgt, tgt_mask


class GeneralTransformerDecoder(torch.nn.Module):
    def __init__(
        self, 
        time_embedding: Tensor = None,
        num_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        use_prob_mask: bool = False,
        linear: torch.nn.Module = None,
        linear_m: torch.nn.Module = None,
        **kwargs
    ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        if time_embedding is None:
            time_embedding = torch.nn.Embedding(dim_time, dim_model) # an embedding lookup dict with key range from 0 to dim_time-1
        gender_embedding = kwargs.get("gender_embedding", None)
        dim_gender = kwargs.get("dim_gender", None)
        race_embedding = kwargs.get("race_embedding", None)
        dim_race = kwargs.get("dim_race", None)

        day_embedding = kwargs.get("day_embedding", None)
        dim_day = kwargs.get("dim_day", None)
        hour_embedding = kwargs.get("hour_embedding", None)
        dim_hour = kwargs.get("dim_hour", None)

        if gender_embedding is None:
            if dim_gender is None: gender_embedding = None
            else:
                gender_embedding = torch.nn.Embedding(dim_gender, dim_model)

        if race_embedding is None:
            if dim_race is None: race_embedding = None
            else:
                race_embedding = torch.nn.Embedding(dim_race, dim_model)
        
        if day_embedding is None:
            if dim_day is None: day_embedding = None
            else:
                day_embedding = torch.nn.Embedding(dim_day, dim_model)

        if hour_embedding is None:
            if dim_hour is None: hour_embedding = None
            else:
                hour_embedding = torch.nn.Embedding(dim_hour, dim_model)

        self.time_embedding = time_embedding
        self.gender_embedding = gender_embedding
        self.race_embedding = race_embedding
        self.day_embedding = day_embedding
        self.hour_embedding = hour_embedding
        self.use_prob_mask = use_prob_mask
        self.max_length = max_length
        self.feature_size = dim_feature
        self.dim_model = dim_model
        self.dim_time = dim_time
        self.feature2hidden = torch.nn.Linear(dim_feature, dim_model)
        self.layers = torch.nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        if linear is not None:
            self.linear = linear
        else:
            self.linear = torch.nn.Linear(dim_model, dim_feature)
        if linear_m is not None:
            self.linear_m = linear_m
        else:
            self.linear_m = torch.nn.Linear(dim_model, dim_feature)
        

    def forward(self, tgt: Tensor, memory: Tensor, time: Tensor, mask: Tensor = None, ava: Tensor = None, **kwargs) -> Tensor:
        if ava is not None:
            tgt = torch.mul(tgt, ava)
        if mask is not None:
            tgt = torch.mul(tgt, mask)
        tgt = self.feature2hidden(tgt)

        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_embedding(seq_len, dimension)
        tgt += self.time_embedding(time)
        
        gender = kwargs.get("gender", None)
        race = kwargs.get("race", None)
        day = kwargs.get("day", None)
        hour = kwargs.get("hour", None)

        #import pdb; pdb.set_trace()
        if gender is not None: tgt += self.gender_embedding(gender)
        if race is not None: tgt += self.race_embedding(race)
        if day is not None: tgt += self.day_embedding(day)
        if hour is not None: tgt += self.hour_embedding(hour)

        for layer in self.layers:
            tgt, mask = layer(tgt, memory, mask)

        output = torch.nn.functional.sigmoid(self.linear(tgt))
        
        if mask is None or self.use_prob_mask: # if use_prob_mask, no need to generate mask
            return output, None
        
        mask =  torch.nn.functional.sigmoid(self.linear_m(tgt))
        return output, time, mask


    def inference(self, start_feature: Tensor, memory: Tensor, start_mask: Tensor = None, prob_mask: Tensor = None, **kwargs):
        start_feature = start_feature.cuda()
        memory = memory.cuda()
        z = memory
        
        if self.use_prob_mask:
            assert prob_mask is not None and start_mask is not None
        
        if prob_mask is not None:
            prob_mask = prob_mask.cuda()
        if start_mask is not None:
            start_mask = start_mask.cuda()
            
        time_shift = kwargs["time_shift"]
        time_scale = kwargs["time_scale"]
        
        if len(time_shift.shape) == 2: time_shift = time_shift[0, 0]
        if len(time_scale.shape) == 2: time_scale = time_scale[0, 0]

        extract_incr_time_from_tempo_step = kwargs["extract_incr_time_from_tempo_step"]
        
        start_time = kwargs["start_time"]
        gender = kwargs.get("gender", None)
        race = kwargs.get("race", None)
        if gender is not None: gender = gender.cuda()
        if race is not None: race = race.cuda()
        use_day_embedding = kwargs.get("use_day", False)
        use_hour_embedding = kwargs.get("use_hour", False)
        
        gen_gender = gender
        gen_race = race

        batch_size = z.size(0)
        zs = torch.unbind(z, dim=1) 
        

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long().cuda() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long().cuda() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool().cuda()
        sequence_length = torch.zeros(batch_size, out=self.tensor()).long().cuda()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long().cuda() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_length, self.feature_size).fill_(0.0).float().cuda()
        gen_masks = self.tensor(batch_size, self.max_length, self.feature_size).fill_(0.0).float().cuda()
        #gen_times = self.tensor(batch_size, self.max_length, 1).fill_(0.0).float().cuda()
        gen_times = self.tensor(batch_size, self.max_length, 1).fill_(0.0).int().cuda()
        
        t=0
        time = start_time.cuda()
        descaled_time = descale_time(time, time_shift, time_scale).cuda()
        pos_emb = position_embedding(self.max_length, self.dim_model)[0].cuda()
        
        while(t<self.max_length and len(running_seqs)>0):
            #import pdb; pdb.set_trace()
            batch_size = z.size(0)
            z_ = z[:, :t+1, :] # [batch, t, fea]
            
            #
            if t == 0:
                # input for time step 0
                input_sequence = start_feature.float().cuda() # [batch, feature_size]
                # save next input
                #generations = self._save_sample(generations, input_sequence, sequence_running, 0, add_grad=True)
                # save time
                #gen_times = self._save_sample(gen_times, time, sequence_running, 0)
                #gen_times = self._save_sample(gen_times, descaled_time, sequence_running, 0)

                input_sequence = input_sequence.unsqueeze(dim=1) # [batch, 1, fea]

                first_input = input_sequence
                first_time = descaled_time.int()
                if start_mask == None:
                    input_mask = None
                else:
                    input_mask = start_mask.float().cuda() # [batch, feature_size]
                    # save next input
                    #gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, 0, add_grad=True)

                    input_mask = input_mask.unsqueeze(dim=1) # [batch, 1, fea]
                    first_mask = input_mask
            
            input_ = input_sequence
            if input_mask is not None:
                input_ = torch.mul(input_, input_mask)
            #import pdb; pdb.set_trace()
            input_ = self.feature2hidden(input_) # [batch, t, fea]

            #input_batch_size, seq_len, dimension = input_.size(0), input_.size(1), input_.size(2)
            input_batch_size = input_.size(0)
            if input_batch_size != batch_size:
                import pdb; pdb.set_trace()
            # [batch, t, fea]
            input_ += pos_emb[:t+1] # position_embedding(seq_len, dimension)
            # time embedding, de-scale before embedding
            input_ += self.time_embedding(descaled_time.int())
            #
            if gen_gender is not None: input_ += self.gender_embedding(gen_gender)
            if gen_race is not None: input_ += self.race_embedding(gen_race)
            if use_day_embedding: 
                descaled_day, _ = extract_day_hour_from_time(descaled_time)
                input_ += self.day_embedding(descaled_day.int())
            if use_hour_embedding: 
                _, descaled_hour = extract_day_hour_from_time(descaled_time)
                input_ += self.hour_embedding(descaled_hour.int())
            
            for layer in self.layers:
                try:
                    input_, input_mask = layer(input_, z_, input_mask)
                except:
                    import pdb; pdb.set_trace()

            gen_input = torch.nn.functional.sigmoid(self.linear(input_))
            last_input = gen_input[:, -1, :]
            # concat
            input_sequence = torch.cat([first_input, gen_input], dim=1)
            # save next input
            #generations = self._save_sample(generations, last_input, sequence_running, t+1, add_grad=True)
            # save time
            #gen_times = self._save_sample(gen_times, time, sequence_running, t+1)
            #last_time = descaled_time[:, [-1]]
            #gen_times = self._save_sample(gen_times, last_time, sequence_running, t+1)
            
            
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
                    input_mask = torch.cat([first_mask, gen_mask], dim=1)
                    #gen_masks = self._save_sample(gen_masks, last_mask, sequence_running, t+1, add_grad=True)
            
            

            # get incr time, update time
            time += extract_incr_time_from_tempo_step(last_input)
            next_time = descale_time(time, time_shift, time_scale).cuda()
            descaled_time = torch.cat([descaled_time, next_time], dim=1) #[batch, t]
            
            #import pdb; pdb.set_trace()
            # update gloabl running sequence
            sequence_length[sequence_running] += 1
            #import pdb; pdb.set_trace()
            # select criteria: sum of feature values > 0 and descaled time < dim_time
            sequence_mask[sequence_running] = torch.logical_and(last_input.sum(dim=1) > 0, next_time.squeeze(dim=1) < self.dim_time) # ??
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = torch.logical_and(last_input.sum(dim=1) > 0, next_time.squeeze(dim=1) < self.dim_time) # ??
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                z = z[running_seqs]
                if input_mask is not None:
                    input_mask = input_mask[running_seqs]
                time = time[running_seqs]
                descaled_time = descaled_time[running_seqs]
                if gen_gender is not None: gen_gender = gen_gender[running_seqs]
                if gen_race is not None: gen_race = gen_race[running_seqs]
                first_input = first_input[running_seqs]
                first_mask = first_mask[running_seqs]
                #
                # save
                generations = self._save_sample(generations, input_sequence, sequence_running, t, add_grad=True)
                gen_times = self._save_sample(gen_times, descaled_time.unsqueeze(dim=2), sequence_running, t)
                if input_mask is not None:
                    gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, t, add_grad=True)
                
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
            #
            t += 1

        if len(sequence_running) > 0:
            generations = self._save_sample(generations, input_sequence, sequence_running, t, add_grad=True)
            gen_times = self._save_sample(gen_times, descaled_time.unsqueeze(dim=2), sequence_running, t)
            if input_mask is not None:
                gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, t, add_grad=True)
        
        output = generations
        times = gen_times.squeeze(dim=2)
        if start_mask == None or self.use_prob_mask:
            return output, times, None
            #return self.forward(output, memory, times, gender, race, mask=None, ava=None)
        
        mask = gen_masks

        #import pdb; pdb.set_trace()
        return output, times, mask
        #return self.forward(output, memory, times, gender, race, mask=mask, ava=None)
        

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
        time_embedding: Tensor = None,
        num_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        use_prob_mask: bool = False,
        **kwargs
    ):
        super().__init__(
            time_embedding=time_embedding,
            num_layers=num_layers,
            max_length=max_length,
            dim_feature=dim_feature,
            dim_model=dim_model,
            dim_time=dim_time,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_prob_mask=use_prob_mask,
            **kwargs
        )



class Transformer(torch.nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        encoder_dropout: float = 0.1, 
        decoder_dropout: float = 0.1, 
        activation: torch.nn.Module = torch.nn.ReLU(),
        use_prob_mask: bool = False,
        **kwargs
    ):
        super().__init__()
        self.time_embedding = torch.nn.Embedding(dim_time, dim_model) # an embedding lookup dict with key range from 0 to dim_time-1
        dim_gender = kwargs.get("dim_gender", None)
        dim_race = kwargs.get("dim_race", None)
        dim_day = kwargs.get("dim_day", None)
        dim_hour = kwargs.get("dim_hour", None)
        dic = {}
        if dim_gender is not None: dic["gender_embedding"] = torch.nn.Embedding(dim_gender, dim_model)
        if dim_race is not None: dic["race_embedding"] = torch.nn.Embedding(dim_race, dim_model)
        if dim_day is not None: dic["day_embedding"] = torch.nn.Embedding(dim_day, dim_model)
        if dim_hour is not None: dic["hour_embedding"] = torch.nn.Embedding(dim_hour, dim_model)

        self.encoder = TransformerEncoder(
            time_embedding=self.time_embedding,
            num_layers=num_encoder_layers,
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            **dic
        )
        self.decoder = TransformerDecoder(
            time_embedding=self.time_embedding,
            num_layers=num_decoder_layers,
            max_length=max_length,
            dim_feature=dim_feature,
            dim_model=dim_model,
            dim_time=dim_time,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
            use_prob_mask=use_prob_mask,
            **dic
        )

    def forward(self, src: Tensor, tgt: Tensor, src_time: Tensor, tgt_time: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None, 
                src_ava: Tensor = None, tgt_ava: Tensor = None, **kwargs) -> Tensor:
        
        src = src.float(); src_time=src_time.int() 
        tgt = tgt.float(); tgt_time=tgt_time.int()
        if src_mask is not None: src_mask = src_mask.float()
        if tgt_mask is not None: tgt_mask = tgt_mask.float()
        if src_ava is not None: src_ava = src_ava.float()
        if tgt_ava is not None: tgt_ava = tgt_ava.float()
        #import pdb; pdb.set_trace()
        gender = kwargs.get("gender", None)
        race = kwargs.get("race", None)
        src_day = kwargs.get("src_day", None)
        src_hour = kwargs.get("src_hour", None)
        tgt_day = kwargs.get("tgt_day", None)
        tgt_hour = kwargs.get("tgt_hour", None)
        enc_dic = {}; dec_dic = {}
        if gender is not None: 
            enc_dic["gender"] = gender.int()
            dec_dic["gender"] = gender.int()
        if race is not None: 
            enc_dic["race"] = race.int()
            dec_dic["race"] = race.int()
        if src_day is not None: enc_dic["day"] = src_day.int()
        if src_hour is not None: enc_dic["hour"] = src_hour.int()
        if tgt_day is not None: dec_dic["day"] = tgt_day.int()
        if tgt_hour is not None: dec_dic["hour"] = tgt_hour.int()

        memory = self.encoder(src, src_time, src_mask, src_ava, **enc_dic)
        output, out_time, out_mask = self.decoder(tgt, memory, tgt_time, tgt_mask, tgt_ava, **dec_dic)
        # build a target prob tensor
        if tgt_mask == None:
            p_input = tgt
        else:
            p_input = torch.mul(tgt, tgt_mask)
        return memory, p_input, output, out_time, out_mask


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
