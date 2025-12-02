import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Cài đặt DQN chuẩn."""

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, next_masks, _, _ = experiences

        # --- DQN LOGIC ---
        # Lấy max Q value từ mạng TARGET
        with torch.no_grad():
            q_next = self.qnetwork_target(next_states)
            # Áp dụng mask cho next_state
            q_next = q_next + (next_masks - 1.0) * 1e9
            max_q = q_next.max(dim=1, keepdim=True)[0]
            q_targets = rewards + (gamma * max_q * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss.item()
