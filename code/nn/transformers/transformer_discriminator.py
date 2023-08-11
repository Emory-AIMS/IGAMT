import torch
from torch import Tensor
import torch.nn.functional as f
from utils.train_utils import descale_time, extract_incr_time_from_tempo_step
from nn.transformers.utils import position_embedding, feed_forward
from nn.transformers.utils import AttentionHead, MultiHeadAttention, Residual
from nn.transformers.mixed_embedding_transformer import TransformerEncoder


class Trans_Discriminator(torch.nn.Module):

    def __init__(
        self, 
        num_layers: int = 6,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 8, 
        max_length: int = 50,
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        use_spectral_norm=False
    ):
        super().__init__()
        self.trans = TransformerEncoder(
            num_layers=num_layers,
            dim_feature=dim_feature,
            dim_model=dim_model,
            dim_time=dim_time,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_size=1
        )


        layer_ = torch.nn.Linear(max_length*dim_model, output_size)
        if use_spectral_norm:
            layer_ = spectral_norm(layer_)
        self.feature2binary = layer_

    def forward(self, tempo: Tensor, time: Tensor, mask: Tensor = None):
        tempo = tempo.cuda().float(); time=time.cuda().int() 
        gender = gender.cuda().int(); race=race.cuda().int()
        if mask is not None: mask = mask.cuda().float()

        if len(time.shape) == 3: time = time.squeeze(dim=2)

        feature_vecs = self.trans(tempo, time, mask)
        feature_vecs = torch.reshape(feature_vecs, (feature_vecs.shape[0], -1))
        scores = self.feature2binary(feature_vecs).squeeze(1)
        return scores

    def cal_gradient_penalty(self, real_data, fake_data, real_time, fake_time, real_mask=None, fake_mask=None, gp_lambda=10):
        real_data = real_data.cuda().float(); fake_data = fake_data.cuda().float()
        real_time=real_time.cuda().int(); fake_time=fake_time.cuda().int()
        gender = gender.cuda().int(); race=race.cuda().int()
        if real_mask is not None: real_mask = real_mask.cuda().float()
        if fake_mask is not None: fake_mask = fake_mask.cuda().float()
        if len(real_time.shape) == 3: real_time = real_time.squeeze(dim=2)
        if len(fake_time.shape) == 3: fake_time = fake_time.squeeze(dim=2)

        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand(real_data.size())

        epsilon_t = torch.rand(batch_size, 1)
        epsilon_t = epsilon_t.expand(real_time.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
            epsilon_t = epsilon_t.cuda()

        if real_mask is not None:
            real_data = torch.mul(real_data, real_mask)
        if fake_mask is not None:
            fake_data = torch.mul(fake_data, fake_mask)

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        #
        inter_time = epsilon_t * real_time + (1 - epsilon_t) * fake_time
        D_interpolates, _, _ = self.forward(interpolates, inter_time, gender, race)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty


    def cal_xentropy_loss(self, preds, target):
        entropy = torch.mul(target, torch.log(preds + 1e-12)) + torch.mul((1-target), torch.log((1-preds + 1e-12)))
        loss = torch.mean(torch.sum(-1.0*entropy, dim=(1)))

        return loss



class Trans_Auxiliary_Discriminator(torch.nn.Module):
    def __init__(
        self, 
        num_layers: int = 6,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 8, 
        max_length: int = 50,
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        use_spectral_norm=False
    ):
        super().__init__()
        self.trans = TransformerEncoder(
            num_layers=num_layers,
            dim_feature=dim_feature,
            dim_model=dim_model,
            dim_time=dim_time,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_size=1
        )
        # for output
        layer_out = torch.nn.Linear(max_length*dim_model, output_size)
        if use_spectral_norm:
            layer_out = spectral_norm(layer_out)
        self.feature2binary = layer_out

        # for gender
        layer_gender = torch.nn.Linear(max_length*dim_model, 2)
        if use_spectral_norm:
            layer_gender = spectral_norm(layer_gender)
        self.feature2gender = layer_gender

        # for race
        layer_race = torch.nn.Linear(max_length*dim_model, 3)
        if use_spectral_norm:
            layer_race = spectral_norm(layer_race)
        self.feature2race = layer_race


    def forward(self, tempo: Tensor, time: Tensor, gender: Tensor, race: Tensor, mask: Tensor = None) -> Tensor:
        tempo = tempo.cuda().float(); time=time.cuda().int() 
        gender = gender.cuda().int(); race=race.cuda().int()
        if mask is not None: mask = mask.cuda().float()
        if len(time.shape) == 3: time = time.squeeze(dim=2)
        feature_vecs = self.trans(tempo, time, gender, race, mask)
        feature_vecs = torch.reshape(feature_vecs, (feature_vecs.shape[0], -1))
        real_scores = self.feature2binary(feature_vecs).squeeze(1)
        gender_probs = torch.nn.functional.softmax(self.feature2gender(feature_vecs), dim=1)
        race_probs = torch.nn.functional.softmax(self.feature2race(feature_vecs), dim=1)
        return real_scores, gender_probs, race_probs

    def cal_gradient_penalty(self, real_data, fake_data, real_time, fake_time, gender, race, real_mask=None, fake_mask=None, gp_lambda=10):
        real_data = real_data.cuda().float(); fake_data = fake_data.cuda().float()
        real_time=real_time.cuda().int(); fake_time=fake_time.cuda().int()
        gender = gender.cuda().int(); race=race.cuda().int()
        if real_mask is not None: real_mask = real_mask.cuda().float()
        if fake_mask is not None: fake_mask = fake_mask.cuda().float()
        if len(real_time.shape) == 3: real_time = real_time.squeeze(dim=2)
        if len(fake_time.shape) == 3: fake_time = fake_time.squeeze(dim=2)
        
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand(real_data.size())

        epsilon_t = torch.rand(batch_size, 1)
        epsilon_t = epsilon_t.expand(real_time.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
            epsilon_t = epsilon_t.cuda()

        if real_mask is not None:
            real_data = torch.mul(real_data, real_mask)
        if fake_mask is not None:
            fake_data = torch.mul(fake_data, fake_mask)

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        #
        inter_time = epsilon_t * real_time + (1 - epsilon_t) * fake_time
        D_interpolates, _, _ = self.forward(interpolates, inter_time, gender, race)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

    
    def cal_xentropy_loss(self, preds, target):
        entropy = torch.mul(target, torch.log(preds + 1e-12)) + torch.mul((1-target), torch.log((1-preds + 1e-12)))
        loss = torch.mean(torch.sum(-1.0*entropy, dim=(1)))

        return loss

