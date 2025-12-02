import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        
        # Value Stream
        self.val_fc = nn.Linear(256, 128)
        self.val_out = nn.Linear(128, 1)
        
        # Advantage Stream
        self.adv_fc = nn.Linear(256, 128)
        self.adv_out = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        val = F.relu(self.val_fc(x))
        val = self.val_out(val)
        
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        # Q = V + (A - mean(A))
        return val + adv - adv.mean(dim=1, keepdim=True)