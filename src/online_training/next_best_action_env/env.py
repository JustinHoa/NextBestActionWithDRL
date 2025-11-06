import numpy as np
import pandas as pd
import json
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from typing import Optional, Dict, List, Tuple
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, '../../../data/raw_data/')

class NextBestActionEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        activity_info_path: str = os.path.join(input_dir, "general_check_up_activities_info.json"),
        possible_state_path: str = os.path.join(input_dir, "possible_state.csv"),
        end_state_path: str = os.path.join(input_dir, "end_state.csv"),
        constraints_path: str = os.path.join(input_dir, "constraint.json"),
        max_queue: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # --- Load activity info ---
        with open(activity_info_path, "r", encoding="utf-8") as f:
            self.activity_info: Dict[str, Dict] = json.load(f)

        self.actions = list(self.activity_info.keys())
        self.num_actions = len(self.actions)

        # --- Load constraint info ---
        with open(constraints_path, 'r', encoding='utf-8') as f:
            self.constraints: Dict[str, Dict] = json.load(f)

        # Get mean test time for blood & urine
        self.mean_blood_test_time = float(
            self.activity_info.get("Blood Test", {}).get("mean_test_time", 0)
        )
        self.mean_urine_test_time = float(
            self.activity_info.get("Urine Test", {}).get("mean_test_time", 0)
        )

        # --- Load possible states ---
        df = pd.read_csv(possible_state_path, header=None)
        self.possible_states = df.values
        self.num_possible_states = len(self.possible_states)

        # --- Load end states ---
        end_df = pd.read_csv(end_state_path, header=None)
        self.end_states = end_df.values

        # --- Basic setup ---
        self.max_queue = max_queue
        self.seed(seed)

        # --- Observation and action spaces ---
        obs_dim = 46  # 2 + 21 + 21 + 2 = 46
        self.observation_space = spaces.Box(
            low=-1.0, high=1e6, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_actions)

        # Internal state
        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        self.max_steps = 50

    # ---------------------------------------------------------------
    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ---------------------------------------------------------------
    def _sample_possible_state(self) -> np.ndarray:
        idx = self.np_random.integers(0, self.num_possible_states)
        return self.possible_states[idx]

    # ---------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0

        # 1️⃣ Pick one precomputed state
        base_state = self._sample_possible_state()
        gender = int(base_state[0])
        marital = int(base_state[1])
        prefix = base_state[2:2 + self.num_actions].astype(int)

        # 2️⃣ Random queue lengths (environment info)
        queue_lengths = self.np_random.integers(
            low=0, high=self.max_queue + 1, size=self.num_actions
        ).astype(float)

        # 3️⃣ Compute blood & urine result times
        try:
            blood_idx = self.actions.index("Blood Test")
        except ValueError:
            blood_idx = None
        try:
            urine_idx = self.actions.index("Urine Test")
        except ValueError:
            urine_idx = None

        if blood_idx is not None and prefix[blood_idx] == 1:
            blood_result_time = float(
                self.np_random.uniform(0, self.mean_blood_test_time)
            )
        else:
            blood_result_time = -1.0

        if urine_idx is not None and prefix[urine_idx] == 1:
            urine_result_time = float(
                self.np_random.uniform(0, self.mean_urine_test_time)
            )
        else:
            urine_result_time = -1.0

        # 4️⃣ Assemble final 46-dimensional state
        state_list = (
            [gender, marital]
            + prefix.tolist()
            + queue_lengths.tolist()
            + [blood_result_time, urine_result_time]
        )
        self.state = np.array(state_list, dtype=np.float32)

        return self.state, {}

    # ---------------------------------------------------------------
    def get_reward(self, state, action: int) -> Tuple[float, bool, dict]:
        """
        Tính reward và kiểm tra tính hợp lệ của action theo constraint.
        Trả về: (reward, valid, info)
        """
        gender = int(state[0])
        marital_status = int(state[1])
        prefix = state[2:2 + self.num_actions].astype(int)
        queue_lengths = state[2 + self.num_actions : 2 + 2*self.num_actions].astype(float)
        blood_result_time = float(state[-2])
        urine_result_time = float(state[-1])

        action_name = self.actions[action]
        info = {"action": action_name}
        valid = True

        reward_repeat = 0.0
        reward_constraint = 0.0
        reward_waiting = 0.0

        # 1️⃣ Kiểm tra repeat
        if prefix[action] == 1:
            reward_repeat = -10.0
            info["violation"] = "repeat_action"
            valid = False

        # 2️⃣ Lấy constraint tương ứng demographic (dùng string keys để an toàn)
        all_constraints = self.constraints.get("constraints_by_demographics", {})
        gender_key = str(gender)
        marital_key = str(marital_status)

        if gender_key in all_constraints:
            if gender == 0:
                constraint = all_constraints.get(gender_key, {}).get(marital_key, {})
            else:
                constraint = all_constraints.get(gender_key, {}).get("any", {})
        else:
            # fallback mặc định
            constraint = {}

        must_not_do = constraint.get("must_not_do", [])
        must_do_before_all = constraint.get("must_do_before", {})

        # 3️⃣ Hard constraint: must_not_do
        if action_name in must_not_do:
            reward_constraint = -10.0
            info["violation"] = "must_not_do"
            valid = False

        # 4️⃣ Hard constraint: must_do_before
        prereqs = must_do_before_all.get(action_name, [])
        for pre_action in prereqs:
            if pre_action in self.actions:
                pre_idx = self.actions.index(pre_action)
                if prefix[pre_idx] == 0:
                    reward_constraint = -10.0
                    info["violation"] = "must_do_before"
                    info["missing"] = pre_action
                    valid = False
                    break

        # 5️⃣ Reward chờ đợi (chỉ tính nếu hợp lệ)
        if valid:
            remaining_time = []

            # Copy và an toàn khi gộp specific nodes
            ql = queue_lengths.copy()
            if ql.size > 20:
                # gộp node 4 và 20 nếu tồn tại
                try:
                    combined = ql[4] + ql[20]
                    ql[4] = combined
                    ql[20] = combined
                except Exception:
                    pass

            for idx, value in enumerate(ql):
                # guard: nếu actions thay đổi số lượng
                if idx >= self.num_actions:
                    break
                act_name = self.actions[idx]
                act_info = self.activity_info.get(act_name, {})
                mean_time = act_info.get("mean_time", 1)
                staff = act_info.get("staff", 1)

                if isinstance(mean_time, dict):
                    mean_time = float(np.mean(list(mean_time.values())))

                time_est = float(value) * float(mean_time) / max(int(staff), 1)
                remaining_time.append(time_est)

            # ensure length
            if len(remaining_time) < self.num_actions:
                # pad zeros (shouldn't usually happen)
                remaining_time += [0.0] * (self.num_actions - len(remaining_time))

            # "Conclusion" phụ thuộc test results: dùng last action index nếu cần
            conclusion_idx = min(len(remaining_time) - 1, self.num_actions - 1)
            remaining_time[conclusion_idx] = max(
                remaining_time[conclusion_idx],
                blood_result_time if blood_result_time >= 0 else 0.0,
                urine_result_time if urine_result_time >= 0 else 0.0
            )

            arr = np.array(remaining_time, dtype=float)
            if np.all(arr == 0):
                arr = np.ones_like(arr) * 1e-6
            norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

            waiting_time = norm[action] if action < len(norm) else 1.0
            reward_waiting = 1.0 - waiting_time

        # 6️⃣ Tổng hợp reward
        reward = reward_constraint + reward_repeat + reward_waiting
        return reward, valid, info

    # ---------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), "Invalid action"

        self.step_count += 1

        # --- decode state ---
        gender = int(self.state[0])
        marital = int(self.state[1])
        prefix = self.state[2:2 + self.num_actions].astype(int)
        queue_lengths = self.state[2 + self.num_actions : 2 + 2*self.num_actions].astype(float)
        blood_result_time = float(self.state[-2])
        urine_result_time = float(self.state[-1])

        # --- reward + validity ---
        reward, valid, info = self.get_reward(self.state, action)

        if valid:
            # update prefix
            prefix[action] = 1

            # update test times if action triggers them
            action_name = self.actions[action]
            if action_name == "Blood Test":
                blood_result_time = float(self.np_random.uniform(0, self.mean_blood_test_time))
            elif action_name == "Urine Test":
                urine_result_time = float(self.np_random.uniform(0, self.mean_urine_test_time))

            # update queues: the patient leaves the chosen queue (–1) and others vary slightly
            queue_lengths[action] = max(0.0, queue_lengths[action] - 1.0)
            noise = self.np_random.integers(-1, 2, size=self.num_actions).astype(float)
            queue_lengths = np.clip(queue_lengths + noise, 0.0, float(self.max_queue))

        # else: keep prefix and queue_lengths unchanged

        # --- update state ---
        new_state_list = (
            [gender, marital]
            + prefix.tolist()
            + queue_lengths.tolist()
            + [blood_result_time, urine_result_time]
        )
        self.state = np.array(new_state_list, dtype=np.float32)

        # termination check
        current_prefix = np.concatenate(([gender, marital], prefix))
        terminated = any(np.array_equal(current_prefix, end_state) for end_state in self.end_states)
        truncated = self.step_count >= self.max_steps

        info.update({
            "step": self.step_count,
            "action_name": self.actions[action],
            "valid_action": valid,
            "terminated": terminated,
            "truncated": truncated,
        })

        return self.state, reward, terminated, truncated, info

    # ---------------------------------------------------------------
    def render(self, mode="human"):
        print("Current state ({} features):".format(len(self.state)))
        print(self.state)
