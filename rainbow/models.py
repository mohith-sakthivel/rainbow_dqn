import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from itertools import chain


class NoisyLinear(nn.Module):
    r"""
    Incorporates factorized gaussian noise to the Linear Layer
    Refer https://arxiv.org/abs/1706.10295 for more details.

    math:
    w = mu_w + (sigma_w . w_noise)
    b = mu_b + (sigma_b . b_noise)
    y = xw^T + b

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        sigma_not: hyperparameter to initialize the noisy layer parameters

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and
          :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight_mu, weight_sigma: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`.
        bias_mu, bias_sigma:   the learnable bias of the module of shape
                :math:`(\text{out\_features})`.
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, sigma_not=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_not = sigma_not
        # Noisy weight
        self.eps_w = Parameter(torch.Tensor(out_features, in_features),
                               requires_grad=False)
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # Noisy bias
            self.bias = True
            self.eps_b = Parameter(torch.Tensor(out_features),
                                   requires_grad=False)
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        mu_initial = 1/math.sqrt(self.in_features)
        sigma_initial = self.sigma_not/math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_initial, mu_initial)
        nn.init.constant_(self.weight_sigma, sigma_initial)
        if self.bias is not None:
            nn.init.uniform_(self.bias_mu, -mu_initial, mu_initial)
            nn.init.constant_(self.bias_sigma, sigma_initial)
        self.generate_noise()

    def forward(self, input):
        # feed noise into weights and bias
        self.weight = self.weight_mu + self.weight_sigma * self.eps_w
        if self.bias is not None:
            self.bias = self.bias_mu + self.bias_sigma * self.eps_b
        return F.linear(input, self.weight, self.bias)

    def generate_noise(self):
        """
        Sample a new set of value for noise variables

        math:
            x ~ Normal(mean=0, std dev=1)
            f(x) = sign(x) . sqrt(|x|)
        """
        noise_in = torch.Tensor(1, self.in_features)
        noise_out = torch.Tensor(self.out_features, 1)
        # sample values from std normal distribution
        nn.init.normal_(noise_in, mean=0.0, std=1.0)
        nn.init.normal_(noise_out, mean=0.0, std=1.0)
        noise_in = torch.sign(noise_in) * torch.sqrt(torch.abs(noise_in))
        noise_out = torch.sign(noise_out) * torch.sqrt(torch.abs(noise_out))
        self.eps_w.data.copy_(torch.matmul(noise_out, noise_in))
        if self.bias is not None:
            self.eps_b.data.copy_(noise_out.squeeze())

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class DuelNet(nn.Module):
    def __init__(self, obs_dim, hid_lyrs, num_actions, activation=nn.ReLU):
        super().__init__()
        assert len(hid_lyrs) >= 2, 'Aleast 2 hidden layers are required'
        # create network shared by both value and adavantage layers
        layers = [obs_dim] + hid_lyrs
        nn_layers = []
        for index in range(len(layers)-2):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
            nn_layers.append(activation())
        self.shared_net = nn.Sequential(*nn_layers)
        # neural network layers to calculate state values
        self.value = nn.Sequential(nn.Linear(hid_lyrs[-2], hid_lyrs[-1]),
                                   activation(),
                                   nn.Linear(hid_lyrs[-1], num_actions))
        # neural network layers to calculate action advantages
        self.advantage = nn.Sequential(nn.Linear(hid_lyrs[-2], hid_lyrs[-1]),
                                       activation(),
                                       nn.Linear(hid_lyrs[-1], num_actions))

    def forward(self, x):
        shared_net = self.shared_net(x)
        value = self.value(shared_net)
        advantage = self.advantage(shared_net)
        action_values = value + (advantage - advantage.mean(-1, keepdim=True))
        return action_values


class NoisyDuelNet(nn.Module):
    def __init__(self, obs_dim, hid_lyrs, num_actions, activation=nn.ReLU):
        super().__init__()
        assert len(hid_lyrs) >= 2, 'Aleast 2 hidden layers are required'
        # create network shared by both value and adavantage layers
        layers = [obs_dim] + hid_lyrs
        nn_layers = []
        for index in range(len(layers)-2):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
            nn_layers.append(activation())
        self.shared_net = nn.Sequential(*nn_layers)
        # neural network layers to calculate state values
        self.value = nn.Sequential(NoisyLinear(hid_lyrs[-2], hid_lyrs[-1]),
                                   activation(),
                                   NoisyLinear(hid_lyrs[-1], num_actions))
        # neural network layers to calculate action advantages
        self.advantage = nn.Sequential(NoisyLinear(hid_lyrs[-2], hid_lyrs[-1]),
                                       activation(),
                                       NoisyLinear(hid_lyrs[-1], num_actions))
        self.noisy_layers = []
        for layer in chain(self.shared_net, self.value, self.advantage):
            if isinstance(layer, NoisyLinear):
                self.noisy_layers.append(layer)

    def forward(self, x):
        shared_net = self.shared_net(x)
        value = self.value(shared_net)
        advantage = self.advantage(shared_net)
        action_values = value + (advantage - advantage.mean(-1, keepdim=True))
        return action_values

    def feed_noise(self):
        """ Feed new noise values into the noisy layers of the network """
        for layer in self.noisy_layers:
            layer.generate_noise()


class NormalNet(nn.Module):
    def __init__(self, obs_dim, hid_lyrs, num_actions, activation=nn.ReLU):
        super().__init__()
        layers = [obs_dim] + hid_lyrs
        nn_layers = []
        for index in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
            nn_layers.append(activation())
        nn_layers.append(nn.Linear(layers[-1], num_actions))
        self.shared_net = nn.Sequential(*nn_layers)

    def forward(self, x):
        return self.shared_net(x)
