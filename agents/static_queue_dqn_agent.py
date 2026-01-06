import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import BaseAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class StaticQueueDQNAgent(BaseAgent):
    """
    DQN Agent for Static Queue Environment.
    - State size: 66 dimensions
    - Action size: 22 (21 activities + 1 mock action)
    - Uses standard DQN learning with action masking
    """
    
    def __init__(self, state_size=66, action_size=22, seed=0, **kwargs):
        # Override state and action sizes
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize base agent with custom sizes
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            seed=seed,
            **kwargs
        )
        
        # Override networks with correct sizes
        self.qnetwork_local = StandardQNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = StandardQNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # Re-initialize optimizer with new network
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=kwargs.get('lr', 1e-4))
    
    def learn(self, experiences, gamma):
        """Standard DQN learning algorithm."""
        states, actions, rewards, next_states, dones, next_masks, _, _ = experiences
        
        # Get max Q value from target network
        with torch.no_grad():
            q_next = self.qnetwork_target(next_states)
            # Apply mask to next_state (mask invalid actions)
            q_next = q_next + (next_masks - 1.0) * 1e9
            max_q = q_next.max(dim=1, keepdim=True)[0]
            q_targets = rewards + (gamma * max_q * (1 - dones))
        
        # Get expected Q values from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        return loss.item()
