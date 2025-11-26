import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.config import (ACTION_SIZE, BATCH_SIZE, BUFFER_SIZE, DEVICE, GAMMA,
                        LR, STATE_SIZE, TAU, UPDATE_EVERY)
from src.networks.q_network import QNetwork
from src.replay.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = DEVICE

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done) -> Optional[float]:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        loss_value = None
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            loss_value = self.learn(experiences, GAMMA)
        return loss_value

    def act(self, state, mask, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            action_values = action_values.cpu().data.numpy().squeeze()
            masked_action_values = action_values + (mask - 1.0) * 1e9
            return int(np.argmax(masked_action_values))
        valid_actions = np.where(mask == 1)[0]
        return int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return float(loss.item())

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
