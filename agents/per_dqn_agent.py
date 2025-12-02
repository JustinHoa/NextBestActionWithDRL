import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import BaseAgent
from common.utils import ACTION_SIZE, STATE_SIZE


class PerDqnAgent(BaseAgent):
    """
    Cài đặt DQN với Prioritized Experience Replay (PER).
    - Kế thừa BaseAgent và bật `use_per=True`.
    - Logic học kết hợp DQN và các bước xử lý của PER.
    """

    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0, **kwargs):
        # Khởi tạo BaseAgent và bật Prioritized Replay
        super().__init__(state_size, action_size, seed, use_per=True, **kwargs)

    def learn(self, experiences, gamma):
        # Unpack experiences từ PER, bao gồm weights và indices
        states, actions, rewards, next_states, dones, next_masks, weights, indices = experiences

        # --- DQN LOGIC ---
        # Lấy max Q value từ mạng TARGET
        with torch.no_grad():
            q_next = self.qnetwork_target(next_states)
            # Áp dụng mask cho next_state
            q_next = q_next + (next_masks - 1.0) * 1e9
            max_q = q_next.max(dim=1, keepdim=True)[0]
            q_targets = rewards + (gamma * max_q * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)

        # --- PER LOGIC ---
        # 1. Tính TD errors để cập nhật priorities
        td_errors = q_targets - q_expected

        # 2. Tính loss có trọng số (importance sampling)
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # 3. Cập nhật priorities trong PER buffer
        self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss.item()