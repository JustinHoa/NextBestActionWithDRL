import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized Noisy Linear layer."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scaled_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scaled_noise(self.in_features)
        epsilon_out = self._scaled_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowQNetwork(nn.Module):
    """Dueling network with factorized noisy layers."""

    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.feature = nn.Linear(state_size, 256)
        self.value_hidden = NoisyLinear(256, 128)
        self.value_out = NoisyLinear(128, 1)
        self.adv_hidden = NoisyLinear(256, 128)
        self.adv_out = NoisyLinear(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.feature(state))
        value = F.relu(self.value_hidden(x))
        value = self.value_out(value)
        adv = F.relu(self.adv_hidden(x))
        adv = self.adv_out(adv)
        return value + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
