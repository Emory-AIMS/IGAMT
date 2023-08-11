import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils.train_utils import to_var, sample_mask_from_prob

class Encoder(nn.Module):

    def __init__(self, rnn, rnn_type, feature_size, hidden_size, hidden_factor, latent_size, dropout_rate=0.5, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        self.rnn_type = rnn_type
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.hidden_factor = hidden_factor
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder_rnn = rnn(self.feature_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.hidden_dropout = nn.Dropout(p=dropout_rate)
        self.hidden2mu = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2var = nn.Linear(hidden_size * self.hidden_factor, latent_size)

    def forward(self, input_sequence, input_mask=None):
        if input_mask == None:
            input_ = input_sequence
        else:
            input_ = torch.mul(input_sequence, input_mask)

        input_ = input_.float().cuda() # solve the issue: "RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM"
        if self.rnn_type != 'lstm':
            _, hidden = self.encoder_rnn(input_)
        else:
            _, (hidden, _) = self.encoder_rnn(input_)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        else:
            hidden = hidden.squeeze()
        #import pdb; pdb.set_trace()
        hidden = self.hidden_dropout(hidden)
        mu = self.hidden2mu(hidden)
        mu = torch.tanh(mu)

        log_var = self.hidden2var(hidden)
        log_var = torch.tanh(log_var)

        return mu, log_var

class Decoder(nn.Module):

    def __init__(self, rnn, rnn_type, feature_size, hidden_size, hidden_factor, latent_size,
                max_length, dropout_rate=0.5, num_layers=1, bidirectional=False, no_mask=False, use_prob_mask=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.rnn_type = rnn_type
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.no_mask = no_mask
        self.use_prob_mask = use_prob_mask

        self.decoder_rnn = rnn(self.feature_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.outputs_dropout = nn.Dropout(p=dropout_rate)
        self.outputs2feature = nn.Linear(hidden_size, self.feature_size)
        # mask
        self.outputs2mask = nn.Linear(hidden_size, self.feature_size)



    def forward(self, mu, log_var):
        z = self.reparameterize(mu, log_var)
        input_ = torch.zeros((z.size(0), self.max_length, self.feature_size))

        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0)

        # decoder forward pass
        if self.rnn_type != 'lstm':
            outputs, _ = self.decoder_rnn(input_, hidden)
        else:
            outputs, _ = self.decoder_rnn(input_, (hidden, torch.zeros_like(hidden)))
        
        outputs = self.outputs_dropout(outputs)
        b,s,_ = outputs.size()

        # project outputs to feature_size
        logits = self.outputs2feature(torch.reshape(outputs, (-1, outputs.size(2))))

        p_output = nn.functional.sigmoid(logits)
        p_output = p_output.view(b, s, self.feature_size)

        if self.no_mask:
            return p_output, None
        # generate mask
        input_mask = input_mask.float().cuda()
        # decoder forward pass
        if self.rnn_type != 'lstm':
            outputs_mask, _ = self.decoder_rnn(input_mask, hidden)
        else:
            outputs_mask, _ = self.decoder_rnn(input_mask, (hidden, torch.zeros_like(hidden)))

        b,s,_ = outputs_mask.size()

        # project outputs to feature_size
        logits_mask = self.outputs2mask(torch.reshape(outputs_mask, (-1, outputs_mask.size(2))))

        m_output = nn.functional.sigmoid(logits_mask)
        m_output = m_output.view(b, s, self.feature_size)
        return p_output, m_output


    def inference(self, n=4, memory=None):
        z = memory
        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        z = z.cuda()
        input_ = torch.zeros((z.size(0), self.max_length, self.feature_size))

        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0)

        # decoder forward pass
        if self.rnn_type != 'lstm':
            outputs, _ = self.decoder_rnn(input_, hidden)
        else:
            outputs, _ = self.decoder_rnn(input_, (hidden, torch.zeros_like(hidden)))
        
        outputs = self.outputs_dropout(outputs)
        b,s,_ = outputs.size()

        # project outputs to feature_size
        logits = self.outputs2feature(torch.reshape(outputs, (-1, outputs.size(2))))

        p_output = nn.functional.sigmoid(logits)
        p_output = p_output.view(b, s, self.feature_size)

        if args.no_mask:
            return p_output, None
        
        # generate mask
        input_mask = input_mask.float().cuda()
        # decoder forward pass
        if self.rnn_type != 'lstm':
            outputs_mask, _ = self.decoder_rnn(input_mask, hidden)
        else:
            outputs_mask, _ = self.decoder_rnn(input_mask, (hidden, torch.zeros_like(hidden)))

        b,s,_ = outputs_mask.size()

        # project outputs to feature_size
        logits_mask = self.outputs2mask(torch.reshape(outputs_mask, (-1, outputs_mask.size(2))))

        m_output = nn.functional.sigmoid(logits_mask)
        m_output = m_output.view(b, s, self.feature_size)
        return p_output, m_output
        


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class Variational_Autoencoder(nn.Module):

    def __init__(self, rnn_type, hidden_size, latent_size, max_length, feature_size,
                encoder_dropout=0.5, decoder_dropout=0.5, num_layers=1, bidirectional=False, no_mask=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_length = max_length
        self.feature_size = feature_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.no_mask = no_mask

        if rnn_type == 'rnn':
            self.rnn = nn.RNN
        elif rnn_type == 'gru':
            self.rnn = nn.GRU
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM
        else:
            raise ValueError()

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.encoder = Encoder(
            rnn=self.rnn,
            rnn_type=self.rnn_type,
            feature_size=self.feature_size,
            hidden_size=self.hidden_size,
            hidden_factor=self.hidden_factor,
            latent_size=self.latent_size,
            dropout_rate=self.encoder_dropout,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
            )

        self.decoder = Decoder(
            rnn=self.rnn,
            rnn_type=self.rnn_type,
            feature_size=self.feature_size,
            hidden_size=self.hidden_size,
            hidden_factor=self.hidden_factor,
            latent_size=self.latent_size,
            max_length=self.max_length,
            dropout_rate=self.decoder_dropout,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            no_mask=self.no_mask
            )

    def forward(self, input_sequence, target_sequence, input_mask=None, target_mask=None):
        mu, log_var = self.encoder(input_sequence, input_mask)
        if len(mu.shape) == 1: 
            mu = mu.unsqueeze(0)
            log_var = log_var.unsqueeze(0)
        p_output, m_output = self.decoder(mu, log_var)
        # build a target prob tensor
        if target_mask == None:
            p_input = target_sequence
        else:
            p_input = torch.mul(target_sequence, target_mask)
        
        return mu, log_var, p_input, p_output, m_output
        


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


    def compute_kl_diver_loss(self, mu, log_var):
        loss = torch.mean(-0.5 * torch.sum(1+log_var-mu**2 - log_var.exp(), dim=1), dim=0)

        return loss
