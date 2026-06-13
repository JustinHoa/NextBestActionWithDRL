"""Simplified environment for SolimanDQN online training (Soliman et al., 2025).

State: [gender(1), marital(1), completed_bitmask(21)] = 23-dim binary vector.
No queue info, no elapsed-time feature.

Reward design (DQN version from paper):
  - Invalid action (masked out): -10
  - Valid, highest-probability transition at this state: +0.1
  - Valid, other transitions: -1.0
  - Terminal (all required activities done): +100
"""
import numpy as np


class SolimanEnv:
    REWARD_INVALID = -10.0
    REWARD_VALID_BEST = 0.1
    REWARD_VALID_OTHER = -1.0
    REWARD_TERMINAL = 100.0

    def __init__(self, action_size: int, trans_probs: dict):
        """
        action_size : number of activities (21)
        trans_probs : {s_key (int): {action (int): probability (float)}}
                      pre-computed from the random-baseline event log
        """
        self.action_size = action_size
        self.trans_probs = trans_probs

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _create_goal_mask(self) -> np.ndarray:
        goal = np.ones(self.action_size, dtype=np.float32)
        if self.features[0] == 1:   # Male
            goal[8] = 0.0; goal[9] = 0.0
        else:                        # Female
            if self.features[1] == 0:  # Single
                goal[8] = 0.0
        return goal

    def _s_key(self) -> int:
        key = 0
        for i, b in enumerate(self.blanks):
            if b:
                key |= 1 << i
        return int(key)

    def _get_state(self) -> np.ndarray:
        return np.concatenate((self.features, self.blanks)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        self.features = np.random.randint(0, 2, size=2).astype(np.float32)
        self.blanks = np.zeros(self.action_size, dtype=np.float32)
        self.done = False
        self.goal_mask = self._create_goal_mask()
        return self._get_state()

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(self.action_size, dtype=np.float32)

        # Already-done activities
        for i in range(self.action_size):
            if self.blanks[i] == 1:
                mask[i] = 0.0

        # Gender / marital constraints
        if self.features[0] == 1:
            mask[8] = 0.0; mask[9] = 0.0
        else:
            if self.features[1] == 0:
                mask[8] = 0.0

        # Sequential prefix rules (identical to FillBlanksEnv)
        prefix_rules = {
            0: [], 1: [0], 2: [0, 1], 3: [0, 1, 2],
            4: [0, 1, 2, 3], 5: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4],
            7: [0, 1, 2, 3, 4], 8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4],
        }
        for act, deps in prefix_rules.items():
            if any(self.blanks[d] != 1 for d in deps):
                mask[act] = 0.0

        def check_full(s, e):
            return all(self.blanks[k] == 1 for k in range(s, e))

        if self.features[0] == 1:   # Male
            if not check_full(0, 8): mask[10:20] = 0.0
            if not (check_full(0, 8) and check_full(10, 20)): mask[20] = 0.0
        else:
            if self.features[1] == 0:   # Female, Single
                if not (check_full(0, 8) and self.blanks[9] == 1): mask[10:20] = 0.0
                if not (check_full(0, 8) and self.blanks[9] == 1
                        and check_full(10, 20)): mask[20] = 0.0
            else:                        # Female, Married
                if not check_full(0, 10): mask[10:20] = 0.0
                if not check_full(0, 20): mask[20] = 0.0

        return mask

    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True

        mask = self.get_action_mask()

        if mask[action] == 0.0:
            return self._get_state(), self.REWARD_INVALID, False

        s_key = self._s_key()
        self.blanks[action] = 1.0

        # Terminal check
        if np.array_equal(self.blanks, self.goal_mask):
            self.done = True
            return self._get_state(), self.REWARD_TERMINAL, True

        # Determine whether this was the highest-probability action
        probs = self.trans_probs.get(s_key, {})
        valid_probs = {a: p for a, p in probs.items() if mask[a] == 1.0}
        is_best = (
            bool(valid_probs) and action == max(valid_probs, key=valid_probs.get)
        )
        reward = self.REWARD_VALID_BEST if is_best else self.REWARD_VALID_OTHER
        return self._get_state(), reward, False
