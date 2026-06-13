"""DQN agent for Soliman et al. (2025) benchmark.

State: [gender(1), marital(1), completed_bitmask(21)] = 23-dim.
No queue info, no time features — matches the paper's design.
Architecture: 23 → 32 → 32 → 21 (adapted from paper's 12 → 32 → 32 → 12).
Target network updated every `target_update_freq` episodes (paper uses 10).
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOLIMAN_STATE_SIZE = 23  # gender(1) + marital(1) + bitmask(21)


class _QNet(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SolimanDQNAgent:
    def __init__(self, action_size: int = 21, lr: float = 1e-3,
                 gamma: float = 0.99, batch_size: int = 128,
                 buffer_size: int = 10000, target_update_freq: int = 10,
                 seed: int = 0):
        self.state_size = SOLIMAN_STATE_SIZE
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.seed = seed
        self._episode = 0

        torch.manual_seed(seed)
        random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = _QNet(self.state_size, action_size).to(self.device)
        self.target_net = _QNet(self.state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory: deque = deque(maxlen=buffer_size)

    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        valid = np.where(mask == 1.0)[0]
        if len(valid) == 0:
            return 0
        if eps > 0.0 and random.random() < eps:
            return int(random.choice(valid))

        s = torch.FloatTensor(state[: self.state_size]).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(s).squeeze(0).cpu().numpy()
        self.q_net.train()
        return int(np.argmax(np.where(mask == 1.0, q, -np.inf)))

    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, next_mask: np.ndarray):
        self.memory.append((
            state[: self.state_size].copy(),
            int(action), float(reward),
            next_state[: self.state_size].copy(),
            float(done),
            next_mask.copy(),
        ))
        if len(self.memory) < self.batch_size:
            return None
        return self._learn()

    def _learn(self) -> float:
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        masks_t = torch.FloatTensor(np.array(next_masks)).to(self.device)

        current_q = self.q_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_t)
            next_q = next_q.masked_fill(masks_t == 0.0, -1e9)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q.max(1)[0]

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def notify_episode_done(self) -> None:
        """Update target network every `target_update_freq` episodes."""
        self._episode += 1
        if self._episode % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, filename: str) -> None:
        torch.save(self.q_net.state_dict(), filename)

    def load(self, filename: str) -> None:
        self.q_net.load_state_dict(
            torch.load(filename, map_location=self.device, weights_only=True)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
