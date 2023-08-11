import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.spectral_norm as spectral_norm
from utils.train_utils import to_var


class MLP_Discriminator(torch.nn.Module):
    def __init__(self, input_size, archs, output_size=1, activation=torch.nn.LeakyReLU(0.2), use_spectral_norm=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + archs
        layers = []

        for i in range(len(layer_sizes)-1):
            layer_ = torch.nn .Linear(layer_sizes[i], layer_sizes[i+1])
            if use_spectral_norm:
                layer_ = spectral_norm(layer_)
            layers.append(layer_)
            layers.append(activation)

        layer_ = torch.nn.Linear(layer_sizes[-1], self.output_size)
        if use_spectral_norm:
            layer_ = spectral_norm(layer_)
        layers.append(layer_)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, early_stem=False):
        batch_size = x.size(0)
        x = torch.cat((x, x.mean(dim=0).repeat([batch_size, 1])), dim=1)
        xs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            xs.append(x)
        if early_stem:
            return xs[-1]
        return x.squeeze(1)

    def cal_gradient_penalty(self, real_data, fake_data, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        D_interpolates = self.forward(interpolates)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

    def cal_mse_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        loss = torch.mean(
                    torch.sum(torch.square(preds-target), dim=(-1))
                )

        return loss



class CNN_Discriminator(torch.nn.Module):

    def __init__(self, feature_size, feature_dropout, filter_size, window_sizes, output_size=1, use_spectral_norm=False):
        super().__init__()
        self.feature_size = feature_size
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)

        self.filter_size = filter_size
        self.window_sizes = window_sizes
        self.output_size = output_size

        layers = []
        for window_size in window_sizes:
            layer_ = torch.nn.Conv2d(1, filter_size, (window_size, self.feature_size), padding=(1, 0))
            if use_spectral_norm:
                layer_ = spectral_norm(layer_)
            layers.append(layer_)

        self.convs = torch.nn.ModuleList(layers)

        layer_ = torch.nn.Linear(filter_size * len(window_sizes), self.output_size)
        if use_spectral_norm:
            layer_ = spectral_norm(layer_)
        self.feature2binary = layer_

    def forward(self, prob_sequence, input_mask=None, use_logits=False, early_stem=False):
        batch_size = prob_sequence.size(0)

        input_ = prob_sequence.view(batch_size, -1, self.feature_size)
        if input_mask != None:
            input_ = torch.mul(input_, input_mask)

        input_ = input_.unsqueeze(1).float() # num_channel
        feature_maps = [torch.nn.functional.leaky_relu(conv(input_), 0.2).squeeze(3) for conv in self.convs]
        if early_stem:
            return feature_maps[-1]
        feature_maps = [torch.nn.functional.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        feature_vecs = torch.cat(feature_maps, 1)
        feature_vecs = self.feature_dropout(feature_vecs)
        if use_logits:
            return feature_vecs
        scores = self.feature2binary(feature_vecs).squeeze(1)
        return scores
        

    def cal_gradient_penalty(self, real_data, fake_data, real_mask=None, fake_mask=None, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        if real_mask is not None:
            real_data = torch.mul(real_data, real_mask)
        if fake_mask is not None:
            fake_data = torch.mul(fake_data, fake_mask)
        
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        D_interpolates = self.forward(interpolates)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

    def cal_mse_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        loss = torch.mean(
                    torch.sum(torch.square(preds-target), dim=(-1))
                )

        return loss


class CNN_Auxiliary_Discriminator(torch.nn.Module):

    def __init__(self, feature_size, feature_dropout, filter_size, window_sizes, output_size=1, use_spectral_norm=False, use_auxiliary=False):
        super().__init__()
        self.feature_size = feature_size
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)

        self.filter_size = filter_size
        self.window_sizes = window_sizes
        self.output_size = output_size
        self.use_auxiliary = use_auxiliary

        layers = []
        for window_size in window_sizes:
            layer_ = torch.nn.Conv2d(1, filter_size, (window_size, self.feature_size), padding=(1, 0))
            if use_spectral_norm:
                layer_ = spectral_norm(layer_)
            layers.append(layer_)

        self.convs = torch.nn.ModuleList(layers)

        # for output
        layer_out = torch.nn.Linear(filter_size * len(window_sizes), self.output_size)
        if use_spectral_norm:
            layer_out = spectral_norm(layer_out)
        self.feature2binary = layer_out

        if self.use_auxiliary:
            # for gender
            layer_gender = torch.nn.Linear(filter_size * len(window_sizes), 2)
            if use_spectral_norm:
                layer_gender = spectral_norm(layer_gender)
            self.feature2gender = layer_gender

            # for race
            layer_race = torch.nn.Linear(filter_size * len(window_sizes), 3)
            if use_spectral_norm:
                layer_race = spectral_norm(layer_race)
            self.feature2race = layer_race


    def forward(self, prob_sequence, input_mask=None, use_logits=False, early_stem=False):
        batch_size = prob_sequence.size(0)

        input_ = prob_sequence.view(batch_size, -1, self.feature_size)
        if input_mask is not None:
            input_ = torch.mul(input_, input_mask)
        
        input_ = input_.unsqueeze(1).float() # num_channel
        feature_maps = [torch.nn.functional.leaky_relu(conv(input_), 0.2).squeeze(3) for conv in self.convs]
        if early_stem:
            return feature_maps[-1]
        feature_maps = [torch.nn.functional.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        feature_vecs = torch.cat(feature_maps, 1)
        feature_vecs = self.feature_dropout(feature_vecs)
        if use_logits:
            return feature_vecs
        real_scores = self.feature2binary(feature_vecs).squeeze(1)
        if self.use_auxiliary:
            gender_vecs = self.feature2gender(feature_vecs)
            race_vecs = self.feature2race(feature_vecs)
            gender_probs = torch.nn.functional.softmax(gender_vecs, dim=1)
            race_probs = torch.nn.functional.softmax(race_vecs, dim=1)
            return real_scores, gender_probs, race_probs
        else:
            return real_scores

    def cal_gradient_penalty(self, real_data, fake_data, real_mask=None, fake_mask=None, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        if real_mask is not None:
            real_data = torch.mul(real_data, real_mask)
        if fake_mask is not None:
            fake_data = torch.mul(fake_data, fake_mask)

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        if self.use_auxiliary:
            D_interpolates, _, _ = self.forward(interpolates)
        else:
            D_interpolates = self.forward(interpolates)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

    
    def cal_xentropy_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        entropy = torch.mul(target, torch.log(preds + 1e-12)) + torch.mul((1-target), torch.log((1-preds + 1e-12)))
        loss = torch.mean(torch.sum(-1.0*entropy, dim=(-1)))

        return loss


    def cal_mse_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        loss = torch.mean(
                    torch.sum(torch.square(preds-target), dim=(-1))
                )

        return loss



class Auxiliary_Classifier(torch.nn.Module):

    def __init__(self, feature_size, feature_dropout, filter_size, window_sizes, use_spectral_norm=False):
        super().__init__()
        self.feature_size = feature_size
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)

        self.filter_size = filter_size
        self.window_sizes = window_sizes

        layers = []
        for window_size in window_sizes:
            layer_ = torch.nn.Conv2d(1, filter_size, (window_size, self.feature_size), padding=(1, 0))
            if use_spectral_norm:
                layer_ = spectral_norm(layer_)
            layers.append(layer_)

        self.convs = torch.nn.ModuleList(layers)

        # for gender
        layer_gender = torch.nn.Linear(filter_size * len(window_sizes), 2)
        if use_spectral_norm:
            layer_gender = spectral_norm(layer_gender)
        self.feature2gender = layer_gender

        # for race
        layer_race = torch.nn.Linear(filter_size * len(window_sizes), 3)
        if use_spectral_norm:
            layer_race = spectral_norm(layer_race)
        self.feature2race = layer_race


    def forward(self, prob_sequence, input_mask=None, use_logits=False, early_stem=False):
        batch_size = prob_sequence.size(0)

        input_ = prob_sequence.view(batch_size, -1, self.feature_size)
        if input_mask is not None:
            input_ = torch.mul(input_, input_mask)

        input_ = input_.unsqueeze(1).float() # num_channel
        feature_maps = [torch.nn.functional.leaky_relu(conv(input_), 0.2).squeeze(3) for conv in self.convs]
        if early_stem:
            return feature_maps[-1]
        feature_maps = [torch.nn.functional.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        feature_vecs = torch.cat(feature_maps, 1)
        feature_vecs = self.feature_dropout(feature_vecs)
        if use_logits:
            return feature_vecs
        gender_probs = torch.nn.functional.softmax(self.feature2gender(feature_vecs), dim=1)
        race_probs = torch.nn.functional.softmax(self.feature2race(feature_vecs), dim=1)
        return gender_probs, race_probs

    
    def cal_xentropy_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        entropy = torch.mul(target, torch.log(preds + 1e-12)) + torch.mul((1-target), torch.log((1-preds + 1e-12)))
        loss = torch.mean(torch.sum(-1.0*entropy, dim=(-1)))

        return loss


    def cal_mse_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        loss = torch.mean(
                    torch.sum(torch.square(preds-target), dim=(-1))
                )

        return loss



class CNN_Net(torch.nn.Module):

    def __init__(self, feature_size, feature_dropout, filter_size, window_sizes, output_size=1):
        super().__init__()
        self.feature_size = feature_size
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)

        self.filter_size = filter_size
        self.window_sizes = window_sizes
        self.output_size = output_size

        layers = []
        for window_size in window_sizes:
            layer_ = torch.nn.Conv2d(1, filter_size, (window_size, self.feature_size), padding=(1, 0))
            layers.append(layer_)

        self.convs = torch.nn.ModuleList(layers)

        # for output
        layer_out = torch.nn.Linear(filter_size * len(window_sizes), self.output_size)
        self.feature2out = layer_out


    def forward(self, prob_sequence, input_mask=None):
        batch_size = prob_sequence.size(0)

        input_ = prob_sequence.view(batch_size, -1, self.feature_size).cuda()
        if input_mask is not None:
            input_ = torch.mul(input_, input_mask.cuda())

        input_ = input_.unsqueeze(1).float() # num_channel
        feature_maps = [torch.nn.functional.leaky_relu(conv(input_), 0.2).squeeze(3) for conv in self.convs]
        feature_maps = [torch.nn.functional.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        feature_vecs = torch.cat(feature_maps, 1)
        feature_vecs = self.feature_dropout(feature_vecs)
        outs = self.feature2out(feature_vecs).squeeze(1)
        return outs
        
    
    def cal_xentropy_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        entropy = torch.mul(target, torch.log(preds + 1e-12)) + torch.mul((1-target), torch.log((1-preds + 1e-12)))
        loss = torch.mean(torch.sum(-1.0*entropy, dim=(-1)))

        return loss


    def cal_mse_loss(self, preds, target):
        batch_size = preds.shape[0]
        preds = torch.reshape(preds, (batch_size, -1))
        target = torch.reshape(target, (batch_size, -1))
        loss = torch.mean(
                    torch.sum(torch.square(preds-target), dim=(-1))
                )

        return loss