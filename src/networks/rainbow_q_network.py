import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RAINBOW_ATOMS
from src.networks.noisy_linear import NoisyLinear


class RainbowQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, atoms: int = RAINBOW_ATOMS):
        super().__init__()
        torch.manual_seed(seed)
        self.atoms = atoms
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.value_stream = nn.Sequential(
            NoisyLinear(256, 256),
            nn.ReLU(),
            NoisyLinear(256, atoms),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(256, 256),
            nn.ReLU(),
            NoisyLinear(256, action_size * atoms),
        )

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x).view(-1, 1, self.atoms)
        advantage = self.advantage_stream(x).view(-1, self.action_size, self.atoms)
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        probs = F.softmax(q_atoms, dim=2)
        return probs

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
