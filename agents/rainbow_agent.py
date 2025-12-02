import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import BaseAgent
from common.utils import ACTION_SIZE, DEVICE, STATE_SIZE
from networks.noisy_net import RainbowQNetwork


class RainbowAgent(BaseAgent):
    """
    Rainbow Agent: Kết hợp các thành phần sau:
    - Double DQN: Để tính target Q-value.
    - Prioritized Experience Replay (PER): Để lấy mẫu kinh nghiệm hiệu quả.
    - Dueling Network: Để ước tính Q-value tốt hơn.
    - Multi-step Learning: Để nhìn xa hơn trong tương lai.
    - Noisy Nets: Để exploration thay cho epsilon-greedy.
    """

    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0):
        # Khởi tạo BaseAgent với các config cho Rainbow
        super().__init__(
            state_size,
            action_size,
            seed,
            use_per=True,      # Bật Prioritized Replay
            n_step=3,          # Bật Multi-step learning (3 bước)
        )

        # Override mạng mặc định bằng RainbowQNetwork (Dueling + Noisy)
        self.qnetwork_local = RainbowQNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = RainbowQNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=1e-4)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def learn(self, experiences, gamma):
        # Unpack experiences từ PER
        states, actions, rewards, next_states, dones, next_masks, weights, indices = experiences

        # Logic của Double DQN
        with torch.no_grad():
            q_local_next = self.qnetwork_local(next_states)
            q_local_next = q_local_next + (next_masks - 1.0) * 1e9
            best_actions = q_local_next.argmax(1).unsqueeze(1)
            
            q_target_next = self.qnetwork_target(next_states).gather(1, best_actions)
            q_targets = rewards + (gamma * q_target_next * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Tính loss với importance sampling weights từ PER
        td_errors = q_targets - q_expected
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # Cập nhật priorities trong PER buffer
        self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss.item()