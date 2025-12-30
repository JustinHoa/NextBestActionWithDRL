import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.buffers import PrioritizedReplayBuffer
from common.utils import DEVICE
from networks.dqn_net import StandardQNetwork

class PenaltyDQNAgent:
    """
    PenaltyDQN Agent - Sử dụng penalty-based reward thay vì action masking.
    Agent sẽ tự học tránh invalid actions thông qua việc bị phạt.
    
    Sử dụng Prioritized Experience Replay để học hiệu quả hơn từ các trải nghiệm bị phạt.
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        buffer_size=int(1e5),
        batch_size=128,
        lr=1e-4,
        gamma=0.99,
        tau=1e-3,
        update_every=4,
        alpha=0.6,  # PER alpha parameter
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Q-Networks
        self.qnetwork_local = StandardQNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = StandardQNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, seed, alpha=alpha)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, next_mask=None):
        """
        Lưu experience và học.
        next_mask không được sử dụng trong PenaltyDQN (để tương thích với interface).
        """
        # Lưu vào memory với TD error ban đầu = 1.0
        self.memory.add(state, action, reward, next_state, done, next_mask)

        # Learn sau mỗi update_every bước
        self.t_step = (self.t_step + 1) % self.update_every
        loss = None
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences, self.gamma)
        return loss

    def act(self, state, mask=None, eps=0.0):
        """
        Chọn action dựa trên epsilon-greedy.
        KHÔNG sử dụng mask - agent tự học tránh invalid actions.
        
        mask parameter được giữ để tương thích với interface nhưng không được sử dụng.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy (KHÔNG dùng mask)
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.action_size)))

    def learn(self, experiences, gamma):
        """
        Học từ batch experiences sử dụng Prioritized Experience Replay.
        """
        if len(experiences) == 7:  # PER format
            states, actions, rewards, next_states, dones, next_masks, (weights, indices) = experiences
        else:
            raise ValueError("PenaltyDQN requires PrioritizedReplayBuffer")

        # Compute Q targets
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute weighted MSE loss
        td_errors = Q_targets - Q_expected
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # Update priorities in replay buffer
        new_priorities = td_errors.detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update của target network."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        """Lưu model."""
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        """Load model."""
        self.qnetwork_local.load_state_dict(torch.load(filename, map_location=DEVICE))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
