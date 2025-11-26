import torch
import torch.nn.functional as F

from src.agents.dqn_agent import DQNAgent
from src.config import GAMMA, TAU


class DoubleDQNAgent(DQNAgent):
    """Same structure as DQNAgent but with Double DQN target computation."""

    def learn(self, experiences, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_q_local = self.qnetwork_local(next_states)
            next_actions = next_q_local.argmax(dim=1, keepdim=True)
            next_q_target = self.qnetwork_target(next_states)
            Q_targets_next = next_q_target.gather(1, next_actions)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return float(loss.item())
