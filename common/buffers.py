import random
from collections import deque
from typing import Deque, List, Sequence, Tuple

import numpy as np
import torch

from common.utils import DEVICE

# --- 1. Standard Replay Buffer ---
class ReplayBuffer:
    """Standard Replay Buffer dùng chung cho DQN, DDQN, Dueling."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray]] = deque(
            maxlen=buffer_size
        )
        self.batch_size = batch_size
        random.seed(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        next_mask: np.ndarray,
    ) -> None:
        """Thêm một experience mới vào memory."""
        self.memory.append((state, action, reward, next_state, done, next_mask))

    def sample(self):
        """Lấy một batch ngẫu nhiên từ memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(DEVICE)

        # Handle next_masks: convert None to zeros array
        next_masks_list = []
        for e in experiences:
            if e[5] is None:
                next_masks_list.append(np.zeros(21, dtype=np.float32))
            else:
                next_masks_list.append(e[5])
        next_masks = torch.from_numpy(np.vstack(next_masks_list)).float().to(DEVICE)
        
        # Trả về weights và indices để có cùng interface với PER
        weights = torch.ones_like(rewards)
        indices = None
        return (states, actions, rewards, next_states, dones, next_masks, weights, indices)

    def update_priorities(self, indices: Sequence[int], priorities: np.ndarray) -> None:
        """Interface giữ chỗ cho PER; buffer thường không cần cập nhật."""
        pass

    def __len__(self) -> int:
        return len(self.memory)

# --- 2. Prioritized Experience Replay Buffer ---
class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay (PER) buffer."""

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seed: int = 0,
        alpha: float = 0.6,
        beta_start: float = 0.5,
        beta_frames: int = 30_000,
        eps: float = 1e-6,
    ):
        super().__init__(buffer_size, batch_size, seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = eps
        self.priorities: Deque[float] = deque(maxlen=buffer_size) # Lưu trữ độ ưu tiên
        self.max_priority = 1.0

    def add(self, *args) -> None:
        """Thêm experience với priority cao nhất."""
        super().add(*args)
        self.priorities.append(self.max_priority)

    def sample(self):
        """Lấy mẫu dựa trên priority và tính importance sampling weights."""
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum() # Chuẩn hóa thành phân phối xác suất

        # Lấy mẫu indices dựa trên xác suất
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]

        # Tính Importance Sampling (IS) weights
        beta = self._beta_by_frame()
        weights = (len(self.memory) * probs[indices]) ** -beta
        weights /= weights.max() # Chuẩn hóa
        weights = torch.from_numpy(weights).float().unsqueeze(1).to(DEVICE)

        # Unpack data như buffer thường
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(DEVICE)
        
        # Handle next_masks: convert None to zeros array
        next_masks_list = []
        for e in experiences:
            if e[5] is None:
                next_masks_list.append(np.zeros(21))  # ACTION_SIZE = 21
            else:
                next_masks_list.append(e[5])
        next_masks = torch.from_numpy(np.vstack(next_masks_list)).float().to(DEVICE)

        # Trả về thêm weights và indices
        return (states, actions, rewards, next_states, dones, next_masks, weights, indices)

    def _beta_by_frame(self) -> float:
        """Tăng beta từ từ theo thời gian."""
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        return beta

    def update_priorities(self, indices: Sequence[int], priorities: np.ndarray) -> None:
        """Cập nhật priority cho các experience đã được học."""
        for idx, priority_val in zip(indices, priorities):
            # Đảm bảo priority > 0
            new_priority = np.abs(priority_val).item() + self.eps
            self.priorities[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)

# --- 3. Multi-step Learning Buffer ---
class MultiStepBuffer:
    """Wrapper cho phép tính n-step return trước khi đẩy vào buffer chính."""

    def __init__(self, base_buffer: ReplayBuffer, n_step: int = 3, gamma: float = 0.99):
        self.base_buffer = base_buffer
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_queue: Deque = deque(maxlen=n_step)

    def add(self, *args) -> None:
        self.n_step_queue.append(args)
        if len(self.n_step_queue) < self.n_step:
            if args[4]:  # done
                self._flush_queue()
            return

        # Khi queue đủ n_step, tính toán và đẩy vào buffer chính
        reward, next_state, done, next_mask = self._get_n_step_info()
        state, action, _, _, _, _ = self.n_step_queue[0]
        self.base_buffer.add(state, action, reward, next_state, done, next_mask)

        # Trượt cửa sổ n-step
        self.n_step_queue.popleft()

        # Nếu episode kết thúc, cần flush phần còn lại (n-1 transition cuối)
        if args[4]:  # done
            self._flush_queue()

    def _flush_queue(self) -> None:
        """Xả hết các experience còn lại trong queue khi hết episode."""
        while self.n_step_queue:
            reward, next_state, done, next_mask = self._get_n_step_info()
            state, action, _, _, _, _ = self.n_step_queue[0]
            self.base_buffer.add(state, action, reward, next_state, done, next_mask)
            self.n_step_queue.popleft()

    def _get_n_step_info(self):
        """Tính toán cumulative reward và lấy (next_state, done) của n bước sau."""
        cumulative_reward = 0.0

        # Mặc định lấy thông tin từ phần tử cuối (n bước sau)
        _, _, _, next_state, done, next_mask = self.n_step_queue[-1]

        # Nếu gặp 'done' giữa chừng, phải dùng next_state/done/next_mask tại đúng bước đó
        for i, (_, _, r, ns, d, nm) in enumerate(self.n_step_queue):
            cumulative_reward += (self.gamma ** i) * r
            if d:
                next_state, done, next_mask = ns, d, nm
                break

        return cumulative_reward, next_state, done, next_mask

    def sample(self):
        """Lấy mẫu từ buffer cơ sở."""
        return self.base_buffer.sample()

    def update_priorities(self, indices: Sequence[int], priorities: np.ndarray) -> None:
        """Chuyển lệnh update priorities xuống buffer cơ sở (nếu là PER)."""
        self.base_buffer.update_priorities(indices, priorities)

    def __len__(self) -> int:
        return len(self.base_buffer)
