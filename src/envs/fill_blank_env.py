import json
import numpy as np

from src.config import ACTION_SIZE, ACTIVITY_INFO_PATH, STATE_SIZE


class FillBlanksEnv:
    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE,
                 data_path: str = ACTIVITY_INFO_PATH):
        self.state_size = state_size
        self.action_size = action_size

        self.WIN_REWARD = 100.0
        self.total_time = 0.0

        with open(data_path, 'r', encoding='utf-8') as f:
            activity_info = json.load(f)

        self.throughput = []
        for key, value in activity_info.items():
            if key == "Payment":
                self.throughput.append((value["mean_time"]["Cash"] + value["mean_time"]["Credit"]) / 2 / value["staff"])
            else:
                self.throughput.append(value["mean_time"]/value["staff"])

        self.mean_time = []
        self.staff = []

        for key in activity_info.keys():
            self.staff.append(activity_info[key]["staff"])
            if key == "Payment":
                time = (activity_info[key]["mean_time"]["Cash"] + activity_info[key]["mean_time"]["Credit"]) / 2
                self.mean_time.append(time)
            else:
                self.mean_time.append(activity_info[key]["mean_time"])

    def _get_state(self):
        return np.concatenate((self.features, self.blanks, self.queues))

    def _create_goal_mask(self):
        goal_mask = np.ones(self.action_size)
        if self.features[0]:
            goal_mask[8] = 0
            goal_mask[9] = 0
        else:
            if not self.features[1]:
                goal_mask[8] = 0
        return goal_mask

    def reset(self):
        self.total_time = 0.0
        self.features = np.random.randint(0, 2, size=2)
        self.blanks = np.zeros(self.action_size)
        self.queues = np.array([np.random.poisson(0.15 * tp) for tp in self.throughput])
        self.done = False
        self.goal_mask = self._create_goal_mask()
        return self._get_state()

    def get_action_mask(self):
        mask = np.ones(self.action_size, dtype=np.float32)
        # Mask for action have done
        for i in range(self.action_size):
            if self.blanks[i] == 1:
                mask[i] = 0.0

        # Mask for Gender and Marital Status
        if self.features[0] == 1:  # Male
            mask[8] = 0.0
            mask[9] = 0.0
        else:  # Female
            if self.features[1] == 0:
                mask[8] = 0.0

        # Mask for constraints
        prefix_rules = {
            0: [], 1: [0], 2: [0,1], 3: [0,1,2], 4: [0,1,2,3],
            5: [0,1,2,3,4], 6: [0,1,2,3,4], 7: [0,1,2,3,4],
            8: [0,1,2,3,4], 9: [0,1,2,3,4]
        }
        for act, deps in prefix_rules.items():
            if any(self.blanks[d] != 1 for d in deps):
                mask[act] = 0.0

        def check_full_1(start, end):
            return all(self.blanks[k] == 1 for k in range(start, end))

        if self.features[0] == 1:  # Male
            if not check_full_1(0,8):
                mask[10:20] = 0.0
            if not (check_full_1(0,8) and check_full_1(10,20)):
                mask[20] = 0.0
        else:  # Female
            if self.features[1] == 0:  # Single
                if not (check_full_1(0,8) and self.blanks[9]==1):
                    mask[10:20] = 0.0
                if not (check_full_1(0,8) and self.blanks[9]==1 and check_full_1(10,20)):
                    mask[20] = 0.0
            else:  # Married
                if not check_full_1(0,10):
                    mask[10:20] = 0.0
                if not check_full_1(0,20):
                    mask[20] = 0.0

        return mask

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, self.done

        self.blanks[action] = 1
        self.queues = np.array([np.random.poisson(0.15 * tp) for tp in self.throughput])

        if np.array_equal(self.blanks, self.goal_mask):
            reward = self.WIN_REWARD
            total = min(self.total_time, 200)
            reward += 50 - (total / 200) * 100
            self.done = True
        else:
            times = [round(self.queues[i] * self.mean_time[i] / self.staff[i], 2) for i in range(len(self.queues))]
            time_action = times[action]
            self.total_time += time_action
            sorted_times = sorted(times)
            ranking = sorted_times.index(time_action)
            reward = 2.0 - 0.2 * ranking
            self.done = False

        next_state = self._get_state()
        return next_state, reward, self.done
