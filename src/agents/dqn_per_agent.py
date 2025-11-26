import numpy as np
import torch
import torch.optim as optim

from src.config import (ACTION_SIZE, BATCH_SIZE, BUFFER_SIZE, DEVICE, GAMMA,
                        LR, PER_ALPHA, PER_BETA_FRAMES, PER_BETA_START,
                        PER_EPS, STATE_SIZE, TAU, UPDATE_EVERY)
from src.networks.q_network import QNetwork
from src.replay.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNPerAgent:
    """DQN agent that learns with Prioritized Experience Replay."""

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.device = DEVICE

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, alpha=PER_ALPHA,
                                              beta_start=PER_BETA_START, beta_frames=PER_BETA_FRAMES,
                                              per_eps=PER_EPS, seed=seed)
        self.t_step = 0
        self.per_eps = PER_EPS

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        loss_value = None
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences, weights, indices = self.memory.sample()
            loss_value = self.learn(experiences, weights, indices, GAMMA)
        return loss_value

    def act(self, state, mask, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            action_values = action_values.cpu().data.numpy().squeeze()
            masked_action_values = action_values + (mask - 1.0) * 1e9
            return int(np.argmax(masked_action_values))
        valid_actions = np.where(mask == 1)[0]
        return int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0

    def learn(self, experiences, weights, indices, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        td_errors = Q_expected - Q_targets
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        new_priorities = td_errors.detach().abs().cpu().numpy().squeeze() + self.per_eps
        self.memory.update_priorities(indices, new_priorities)
        return float(loss.item())

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
