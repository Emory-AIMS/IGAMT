import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils.train_utils import to_var

class MLP_Generator(nn.Module):
    def __init__(self, input_size, output_size, archs, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + archs
        layers = []

        fc_layers = []
        bn_layers = []
        ac_layers = []

        for i in range(len(layer_sizes)-1):
            fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            bn_layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            ac_layers.append(activation)

        fc_layers.append(nn.Linear(layer_sizes[-1], output_size, bias=False))
        bn_layers.append(nn.BatchNorm1d(output_size))
        ac_layers.append(nn.Tanh())

        self.fc_layers = nn.ModuleList(fc_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.ac_layers = nn.ModuleList(ac_layers)

    def forward(self, x=None, batch_size=None):
        if x is None:
            x = to_var(torch.randn([batch_size, self.input_size]))
        
        for i in range(len(self.fc_layers)):
            x1 = self.fc_layers[i](x)
            x2 = self.bn_layers[i](x1)
            x3 = self.ac_layers[i](x2)
            if i != len(self.fc_layers)-1:
                x = x + x3
            else:
                x = x3
        
        return x