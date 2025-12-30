import numpy as np
import pandas as pd
import json
import os

class PenaltyEnv:
    """
    Environment với penalty-based reward system (không dùng action mask).
    Agent sẽ bị phạt nặng nếu chọn invalid action.
    
    State: [gender, marital, 21 blanks, 21 waiting_times, 1 norm_time] = 45 dims
    """
    def __init__(self, state_size, action_size, data_path, activity_info_path='data/raw/activity_info.json'):
        self.state_size = state_size
        self.action_size = action_size
        
        self.WIN_REWARD = 200.0
        self.PENALTY_REWARD = -50.0  # Phạt khi điền vào ô đã có
        self.RULE_VIOLATION_PENALTY = -60.0  # Phạt khi vi phạm luật
        self.STEP_REWARD = 1.0
        self.total_time = 0.0

        # Load Activity Info
        if not os.path.exists(activity_info_path):
            activity_info_path = os.path.join('..', activity_info_path)
        try:
            with open(activity_info_path, 'r', encoding='utf-8') as f:
                activity_info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find activity_info.json at {activity_info_path}")

        # Load Trace Data
        print(f"📂 PenaltyEnv loading data: {data_path}")
        try:
            df = pd.read_csv(data_path)
            full_trace = df.values
            if full_trace.shape[1] == 22:
                self.queue_trace = full_trace[:, 1:]
            else:
                self.queue_trace = full_trace
        except Exception as e:
            print(f"⚠️ Error loading trace: {e}. Using dummy data.")
            self.queue_trace = np.zeros((1058, 21))

        self.max_trace_len = len(self.queue_trace)

        # Prepare computation arrays
        self.mean_time = []
        self.staff = []
        for key in activity_info.keys():
            self.staff.append(activity_info[key]["staff"])
            if key == "Payment":
                t = (activity_info[key]["mean_time"]["Cash"] + activity_info[key]["mean_time"]["Credit"]) / 2
                self.mean_time.append(t)
            else:
                self.mean_time.append(activity_info[key]["mean_time"])
        
        self.mean_time_arr = np.array(self.mean_time)
        self.staff_arr = np.array(self.staff)

    def _get_state(self):
        """State: [gender, marital, 21 blanks, 21 waiting_times, 1 norm_time]"""
        estimated_wait_time = (self.queues * self.mean_time_arr) / self.staff_arr
        norm_wait_time = estimated_wait_time / 200.0
        norm_current_time = np.array([self.total_time / self.max_trace_len])
        return np.concatenate((self.features, self.blanks, norm_wait_time, norm_current_time))
        
    def _get_queues_at(self, t):
        if t < self.max_trace_len:
            return self.queue_trace[t]
        return np.zeros(self.action_size)

    def _create_goal_mask(self):
        """Tạo goal mask dựa trên features"""
        goal_mask = np.ones(self.action_size)
        if self.features[0]:  # Male
            goal_mask[8] = 0
            goal_mask[9] = 0
        else:  # Female
            if not self.features[1]:
                goal_mask[8] = 0
        return goal_mask

    def reset(self):
        self.total_time = 0.0
        self.features = np.random.randint(0, 2, size=2)
        self.blanks = np.zeros(self.action_size)
        self.done = False
        self.goal_mask = self._create_goal_mask()
        self.start_time_idx = np.random.randint(0, max(1, self.max_trace_len - 100))
        self.queues = self._get_queues_at(self.start_time_idx)
        return self._get_state()

    def _check_rule_violation(self, action):
        """Kiểm tra xem action có vi phạm luật không"""
        # Luật 1: Action bị cấm theo features
        if self.features[0] == 1:  # Male
            if action in {8, 9}:
                return True
        elif self.features[0] == 0 and self.features[1] == 0:  # Female, Single
            if action == 8:
                return True

        # Luật 2: Kiểm tra thứ tự prefix
        prefix_rules = {
            0: [],
            1: [0],
            2: [0, 1],
            3: [0, 1, 2],
            4: [0, 1, 2, 3],
            5: [0, 1, 2, 3, 4],
            6: [0, 1, 2, 3, 4],
            7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4],
            9: [0, 1, 2, 3, 4]
        }

        if action in prefix_rules:
            required = prefix_rules[action]
            if any(self.blanks[i] != 1 for i in required):
                return True

        # Luật 3: Phụ thuộc vào features
        if self.features[0] == 1:  # Male
            if action in range(10, 20):
                if any(self.blanks[i] != 1 for i in range(8)):
                    return True
            if action == 20:
                if any(self.blanks[i] != 1 for i in range(8)):
                    return True
                if any(self.blanks[i] != 1 for i in range(10, 20)):
                    return True
        else:  # Female
            if self.features[1] == 0:  # Single
                if action in range(10, 20):
                    required = list(range(8)) + [9]
                    if any(self.blanks[i] != 1 for i in required):
                        return True
                if action == 20:
                    required = list(range(8)) + [9] + list(range(10, 20))
                    if any(self.blanks[i] != 1 for i in required):
                        return True
            else:  # Married
                if action in range(10, 20):
                    if any(self.blanks[i] != 1 for i in range(10)):
                        return True
                if action == 20:
                    if any(self.blanks[i] != 1 for i in range(20)):
                        return True

        return False

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, self.done

        # Kiểm tra vi phạm luật
        is_rule_violated = self._check_rule_violation(action)
        
        if is_rule_violated:
            reward = self.RULE_VIOLATION_PENALTY
            self.done = True
            return self._get_state(), reward, self.done

        # Kiểm tra ô đã điền
        if self.blanks[action] == 1:
            reward = self.PENALTY_REWARD
            self.done = True
            return self._get_state(), reward, self.done

        # Action hợp lệ
        current_q = self.queues[action]
        time_action = round((current_q * self.mean_time_arr[action]) / self.staff_arr[action], 2)
        
        self.blanks[action] = 1
        self.total_time += time_action
        
        # Time Travel
        current_trace_idx = self.start_time_idx + int(self.total_time)
        self.queues = self._get_queues_at(current_trace_idx)

        # Reward Calculation
        if np.array_equal(self.blanks, self.goal_mask):
            reward = self.WIN_REWARD
            MAX_TIME = 800.0
            total = min(self.total_time, MAX_TIME)
            time_bonus = 100.0 - (total / MAX_TIME) * 100.0
            reward += time_bonus
            self.done = True
        else:
            # Reward dựa trên ranking của waiting time
            all_wait = (self.queues * self.mean_time_arr) / self.staff_arr
            sorted_times = sorted(all_wait)
            ranking = sorted_times.tolist().index(time_action) if time_action in sorted_times else len(sorted_times) // 2
            reward = 2.0 - 0.2 * ranking
            self.done = False

        return self._get_state(), reward, self.done
