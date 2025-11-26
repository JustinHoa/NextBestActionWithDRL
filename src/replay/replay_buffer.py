import random
from collections import deque

import numpy as np
import torch

from src.config import DEVICE

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
