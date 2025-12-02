import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from agents.base_agent import BaseAgent

class DDQNAgent(BaseAgent):
    """Cài đặt Double DQN."""
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, next_masks, _, _ = experiences

        # --- DOUBLE DQN LOGIC ---
        # 1. Dùng mạng LOCAL để chọn hành động tốt nhất (a_max) cho state tiếp theo
        Q_local_next = self.qnetwork_local(next_states).detach()
        Q_local_next = Q_local_next + (next_masks - 1.0) * 1e9
        best_actions = Q_local_next.argmax(1).unsqueeze(1)

        # 2. Dùng mạng TARGET để tính giá trị Q cho hành động a_max đó
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        
        # Tính Q_target
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss.item()