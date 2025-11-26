import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

from src.config import (BATCH_SIZE, BUFFER_SIZE, DEVICE, PER_ALPHA,
                        PER_BETA_END, PER_BETA_FRAMES, PER_BETA_START, PER_EPS)


class PrioritizedReplayBuffer:
    """Simple PER buffer using proportional prioritization."""

    def __init__(self,
                 buffer_size: int = BUFFER_SIZE,
                 batch_size: int = BATCH_SIZE,
                 alpha: float = PER_ALPHA,
                 beta_start: float = PER_BETA_START,
                 beta_frames: int = PER_BETA_FRAMES,
                 per_eps: float = PER_EPS,
                 seed: int = 0):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = PER_BETA_END
        self.beta_increment = (PER_BETA_END - beta_start) / max(1, beta_frames)
        self.per_eps = per_eps
        self.seed = random.seed(seed)

        self.memory: List[Tuple] = [None] * buffer_size
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.next_idx = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.memory[self.next_idx] = (state, action, reward, next_state, done)
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def _get_probabilities(self):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        prios = self.priorities[:self.size]
        scaled = prios ** self.alpha
        total = scaled.sum()
        if total == 0:
            # fallback to uniform
            scaled = np.ones_like(scaled) / len(scaled)
        else:
            scaled /= total
        return scaled

    def sample(self):
        probs = self._get_probabilities()
        indices = np.random.choice(self.size, self.batch_size, p=probs)

        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().unsqueeze(1).to(DEVICE)

        states = torch.from_numpy(np.vstack([self.memory[i][0] for i in indices])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([self.memory[i][1] for i in indices])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([self.memory[i][2] for i in indices])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([self.memory[i][3] for i in indices])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([self.memory[i][4] for i in indices]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones), weights, indices

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]):
        for idx, priority in zip(indices, priorities):
            p = float(priority) + self.per_eps
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        return self.size
