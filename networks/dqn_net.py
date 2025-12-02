import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(StandardQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)