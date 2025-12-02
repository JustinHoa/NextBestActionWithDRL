import numpy as np
import random
import torch
import torch.optim as optim

from common.buffers import ReplayBuffer, PrioritizedReplayBuffer, MultiStepBuffer
from common.utils import DEVICE
from networks.dqn_net import StandardQNetwork # Mặc định
from torch.nn.utils import clip_grad_norm_

class BaseAgent:
    """
    Base Agent xử lý các việc chung:
    - Init networks, optimizer, memory
    - Hàm step (lưu vào mem + trigger learn)
    - Hàm act (chọn hành động + MASKING)
    - Save/Load
    """

    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        # Các tham số có thể được override bởi agent con
        buffer_size=int(1e5),
        batch_size=128,
        lr=1e-4,
        gamma=0.99,
        tau=1e-3,
        update_every=4,
        use_per=False,
        n_step=1,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.n_step = n_step
        self.seed = random.seed(seed)

        # Mặc định dùng Standard Net, Agent con (như Dueling) sẽ override dòng này
        self.qnetwork_local = StandardQNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = StandardQNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Chọn Replay Buffer
        if use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, seed)
        else:
            self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        
        # Bọc buffer bằng MultiStep nếu cần
        if n_step > 1:
            self.memory = MultiStepBuffer(self.memory, n_step=n_step, gamma=gamma)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done, next_mask):
        # 1. Lưu experience vào Replay Buffer
        self.memory.add(state, action, reward, next_state, done, next_mask)

        # 2. Learn sau mỗi update_every bước
        self.t_step = (self.t_step + 1) % self.update_every
        loss = None
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences, self.gamma ** self.n_step)
        return loss

    def act(self, state, mask, eps=0.0):
        """
        Chọn hành động dựa trên Epsilon-Greedy và Action Masking.
        Hàm này dùng chung cho DQN, DDQN, Dueling, Rainbow.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        
        # Nếu là NoisyNet, reset noise
        if hasattr(self.qnetwork_local, "reset_noise"):
            self.qnetwork_local.reset_noise()
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy
        if random.random() > eps:
            action_values = action_values.cpu().data.numpy().squeeze()
            # Áp dụng MASK
            masked_action_values = action_values + (mask - 1.0) * 1e9
            return int(np.argmax(masked_action_values))
        # Chọn ngẫu nhiên từ các hành động hợp lệ
        else:
            valid_actions = np.where(mask == 1)[0]
            return int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0

    def learn(self, experiences, gamma):
        """Sẽ được override bởi các class con (DQN, DDQN)."""
        raise NotImplementedError

    def soft_update(self, local_model, target_model, tau):
        """Update trọng số mạng Target từ từ theo mạng Local."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename, map_location=DEVICE))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def reset_noisy_layers(self):
        """Reset noise cho các layer nếu dùng NoisyNet."""
        if hasattr(self.qnetwork_local, "reset_noise"):
            self.qnetwork_local.reset_noise()
        if hasattr(self.qnetwork_target, "reset_noise"):
            self.qnetwork_target.reset_noise()
