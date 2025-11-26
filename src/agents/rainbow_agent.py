import torch
import torch.optim as optim

from src.config import (ACTION_SIZE, BATCH_SIZE, BUFFER_SIZE, DEVICE, GAMMA, LR,
                        N_STEP, RAINBOW_ATOMS, RAINBOW_VMAX, RAINBOW_VMIN,
                        STATE_SIZE, TAU)
from src.networks.rainbow_q_network import RainbowQNetwork
from src.replay.prioritized_replay_buffer import PrioritizedReplayBuffer


class RainbowAgent:
    """Rainbow DQN agent combining double, dueling, noisy, PER, and n-step returns."""

    def __init__(self,
                 state_size: int = STATE_SIZE,
                 action_size: int = ACTION_SIZE,
                 seed: int = 0,
                 n_step: int = N_STEP):
        self.state_size = state_size
        self.action_size = action_size
        self.device = DEVICE
        self.seed = seed

        self.atoms = RAINBOW_ATOMS
        self.v_min = RAINBOW_VMIN
        self.v_max = RAINBOW_VMAX
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.n_step = n_step
        self.gamma_n = GAMMA ** self.n_step

        self.qnetwork_local = RainbowQNetwork(state_size, action_size, seed, self.atoms).to(self.device)
        self.qnetwork_target = RainbowQNetwork(state_size, action_size, seed, self.atoms).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.memory = PrioritizedReplayBuffer(buffer_size=BUFFER_SIZE,
                                              batch_size=BATCH_SIZE,
                                              seed=seed,
                                              n_step=self.n_step,
                                              gamma=GAMMA)

    def act(self, state, mask, eps=0.):
        del eps  # exploration via noisy layers
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            dist = self.qnetwork_local(state)
            q_values = torch.sum(dist * self.support, dim=2)
            masked_q = q_values + (mask_tensor.unsqueeze(0) - 1.0) * 1e9
            action = int(torch.argmax(masked_q, dim=1).item())
        self.qnetwork_local.train()
        self.qnetwork_local.reset_noise()
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences, weights, indices = self.memory.sample()
            return self.learn(experiences, weights, indices)
        return None

    def learn(self, experiences, weights, indices):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_dist_target = self.qnetwork_target(next_states)
            next_dist_local = self.qnetwork_local(next_states)
            next_q_values = torch.sum(next_dist_local * self.support, dim=2)
            next_actions = next_q_values.argmax(dim=1)
            batch_indices = torch.arange(next_dist_target.size(0), device=self.device)
            next_dist = next_dist_target[batch_indices, next_actions]

            not_done = (1 - dones)
            Tz = rewards + not_done * self.gamma_n * self.support
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            l = l.clamp(0, self.atoms - 1)
            u = u.clamp(0, self.atoms - 1)

            m = torch.zeros_like(next_dist)
            m.scatter_add_(1, l, next_dist * (u.float() - b))
            m.scatter_add_(1, u, next_dist * (b - l.float()))

        dist = self.qnetwork_local(states)
        batch_indices = torch.arange(dist.size(0), device=self.device)
        dist = dist[batch_indices, actions.squeeze(1)]
        log_dist = torch.log(dist + 1e-6)

        per_sample_loss = -(m * log_dist).sum(dim=1)
        weighted_loss = (weights.squeeze(1) * per_sample_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        priorities = per_sample_loss.detach().abs().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)
        return float(weighted_loss.item())

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
