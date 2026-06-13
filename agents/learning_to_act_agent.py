import pickle
from typing import Dict, Optional

import numpy as np

class LearningToActAgent:
    def __init__(self, state_size: int, action_size: int, seed: int = 0, policy: Optional[Dict[int, int]] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.policy: Dict[int, int] = policy or {}

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.policy, f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.policy = pickle.load(f)

    def _state_to_key(self, state: np.ndarray) -> int:
        """Minimal state: bitmask of completed activities (prefix).

        Coordinator.predict() encodes state as:
        [gender, marital] + prefix(21) + norm_wait(21) + norm_time(1)
        """
        prefix = state[2 : 2 + 21]
        bits = (prefix > 0.5).astype(np.uint8)
        key = 0
        for i, b in enumerate(bits):
            if b:
                key |= 1 << i
        return int(key)

    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        """Return action id compatible with BaseAgent.act signature.

        eps is ignored; policy is deterministic.
        """
        key = self._state_to_key(state)
        a = self.policy.get(key)

        if a is not None and 0 <= a < self.action_size and mask[a] == 1.0:
            return int(a)

        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        # Fallback: choose uniformly among valid actions
        return int(np.random.choice(valid_actions))
