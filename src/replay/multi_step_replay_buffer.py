import random
from collections import deque

import numpy as np
import torch

from src.config import BATCH_SIZE, BUFFER_SIZE, DEVICE, GAMMA, N_STEP


class MultiStepReplayBuffer:
    def __init__(self,
                 buffer_size: int = BUFFER_SIZE,
                 batch_size: int = BATCH_SIZE,
                 n_step: int = N_STEP,
                 gamma: float = GAMMA,
                 seed: int = 0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step:
            self._append_from_buffer()

        if done:
            while len(self.n_step_buffer) > 0:
                self._append_from_buffer()

    def _append_from_buffer(self):
        reward, next_state, done_flag = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        self.memory.append((state, action, reward, next_state, done_flag))
        self.n_step_buffer.popleft()

    def _get_n_step_info(self):
        reward, next_state, done_flag = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, n_state, done) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if done:
                next_state = n_state
                done_flag = True
                break
        return reward, next_state, done_flag

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
