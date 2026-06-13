from collections import defaultdict
import pickle
from typing import Optional

import numpy as np


class TabularQLAgent:
    """Off-policy tabular Q-learning (Hundogan et al., 2025).

    State key: 21-bit bitmask of completed activities (prefix).
    Hyperparameters follow the paper: alpha=0.2, gamma=0.2, epsilon=0.1.
    """

    def __init__(self, state_size: int, action_size: int,
                 alpha: float = 0.2, gamma: float = 0.2, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self.q_table: dict = defaultdict(lambda: np.zeros(action_size, dtype=np.float32))

    def _state_to_key(self, state: np.ndarray) -> int:
        prefix = state[2: 2 + 21]
        bits = (prefix > 0.5).astype(np.uint8)
        key = 0
        for i, b in enumerate(bits):
            if b:
                key |= 1 << i
        return int(key)

    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        valid = np.where(mask == 1.0)[0]
        if len(valid) == 0:
            return 0
        if eps > 0.0 and np.random.random() < eps:
            return int(np.random.choice(valid))
        key = self._state_to_key(state)
        masked_q = np.where(mask == 1.0, self.q_table[key], -np.inf)
        return int(np.argmax(masked_q))

    def update(self, s_key: int, a: int, r: float,
               s_next_key: int, mask_next: np.ndarray, done: bool) -> None:
        if done:
            td_target = r
        else:
            valid_next = np.where(mask_next == 1.0)[0]
            next_max = (float(np.max(self.q_table[s_next_key][valid_next]))
                        if len(valid_next) > 0 else 0.0)
            td_target = r + self.gamma * next_max
        self.q_table[s_key][a] += self.alpha * (td_target - self.q_table[s_key][a])

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))
        self.q_table.update({k: np.array(v, dtype=np.float32) for k, v in data.items()})
