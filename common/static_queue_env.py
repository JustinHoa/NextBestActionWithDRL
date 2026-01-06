import numpy as np
import pandas as pd
import json
import os

class StaticQueueEnv:
    """
    Environment with Static Queue Constraints:
    - Max queue = 2 × resource for each activity
    - Actions become invalid when queue is full
    - Mock action (index 21) when no valid actions available
    - Applies only to concurrent activities (Cluster 1 & 2)
    """
    
    def __init__(self, state_size, action_size, data_path, activity_info_path='data/raw/activity_info.json'):
        self.state_size = state_size  # 66 dimensions
        self.action_size = action_size  # 22 (21 activities + 1 mock)
        self.WIN_REWARD = 100.0
        self.MOCK_PENALTY = -0.5
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
        print(f"📂 StaticQueueEnv loading data: {data_path}")
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
        
        # Static Queue Constraints: max_queue = 2 × staff
        self.max_queue = 2 * self.staff_arr
        
        # Define concurrent activity groups (indices 0-20)
        # Cluster 1: Eye(5), ENT(6), Dental(7), Gynecological(8), Breast(9)
        # Cluster 2: DEXA(10), Chest X-ray(11), In-depth Eye(12), General Ultrasound(13),
        #            Urine Test(14), ENT Endoscopy(15), ECG(16), Spirometry(17), Cardiac Ultrasound(18), Blood Test(19)
        self.cluster1_indices = [5, 6, 7, 8, 9]
        self.cluster2_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.concurrent_indices = self.cluster1_indices + self.cluster2_indices
        
        # Mock action statistics
        self.mock_action_count = 0
    
    def _get_state(self):
        """
        State: [gender, marital, 21 blanks, 21 waiting_times, 21 queue_utilization, 1 norm_time]
        Total: 2 + 21 + 21 + 21 + 1 = 66 dimensions
        """
        estimated_wait_time = (self.queues * self.mean_time_arr) / self.staff_arr
        norm_wait_time = estimated_wait_time / 200.0
        
        # Queue utilization: current_queue / max_queue (0-1)
        queue_utilization = self.queues / self.max_queue
        queue_utilization = np.clip(queue_utilization, 0.0, 1.0)
        
        norm_current_time = np.array([self.total_time / self.max_trace_len])
        
        return np.concatenate((
            self.features,
            self.blanks,
            norm_wait_time,
            queue_utilization,
            norm_current_time
        ))
    
    def _get_queues_at(self, t):
        if t < self.max_trace_len:
            return self.queue_trace[t]
        return np.zeros(21)
    
    def _create_goal_mask(self):
        goal_mask = np.ones(21)
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
        self.blanks = np.zeros(21)
        self.done = False
        self.goal_mask = self._create_goal_mask()
        self.start_time_idx = np.random.randint(0, max(1, self.max_trace_len - 100))
        self.queues = self._get_queues_at(self.start_time_idx)
        self.mock_action_count = 0
        return self._get_state()
    
    def get_action_mask(self):
        """
        Action mask with queue capacity constraints.
        Action 21 (mock) is always available as fallback.
        """
        # Start with 22 actions (21 activities + 1 mock)
        mask = np.ones(22, dtype=np.float32)
        
        # Base masking: already completed actions
        for i in range(21):
            if self.blanks[i] == 1:
                mask[i] = 0.0
        
        # Gender/marital constraints
        if self.features[0] == 1:  # Male
            mask[8] = 0.0
            mask[9] = 0.0
        else:  # Female
            if self.features[1] == 0:  # Single
                mask[8] = 0.0
        
        # Dependency rules (prefix ordering)
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
        
        for act, deps in prefix_rules.items():
            if any(self.blanks[d] != 1 for d in deps):
                mask[act] = 0.0
        
        # Cluster dependencies
        def check_full(s, e):
            return all(self.blanks[k] == 1 for k in range(s, e))
        
        if self.features[0] == 1:  # Male
            if not check_full(0, 8):
                mask[10:20] = 0.0
            if not (check_full(0, 8) and check_full(10, 20)):
                mask[20] = 0.0
        else:  # Female
            if self.features[1] == 0:  # Single
                if not (check_full(0, 8) and self.blanks[9] == 1):
                    mask[10:20] = 0.0
                if not (check_full(0, 8) and self.blanks[9] == 1 and check_full(10, 20)):
                    mask[20] = 0.0
            else:  # Married
                if not check_full(0, 10):
                    mask[10:20] = 0.0
                if not check_full(0, 20):
                    mask[20] = 0.0
        
        # Queue capacity constraints (only for concurrent activities)
        for i in self.concurrent_indices:
            if mask[i] == 1.0 and self.queues[i] >= self.max_queue[i]:
                mask[i] = 0.0
        
        # Mock action (index 21) is always available if no other actions
        if np.sum(mask[:21]) == 0:
            mask[21] = 1.0
        else:
            mask[21] = 0.0
        
        return mask
    
    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, self.done
        
        # Handle mock action (index 21)
        if action == 21:
            self.mock_action_count += 1
            self.total_time += 1.0  # Wait 1 minute
            
            # Update queues after 1 minute
            current_trace_idx = self.start_time_idx + int(self.total_time)
            self.queues = self._get_queues_at(current_trace_idx)
            
            return self._get_state(), self.MOCK_PENALTY, False
        
        # Normal action (0-20)
        mask_before = self.get_action_mask()
        
        current_q = self.queues[action]
        time_action = round((current_q * self.mean_time_arr[action]) / self.staff_arr[action], 2)
        
        self.blanks[action] = 1
        self.total_time += time_action
        
        # Time Travel
        current_trace_idx = self.start_time_idx + int(self.total_time)
        self.queues = self._get_queues_at(current_trace_idx)
        
        # Reward Calculation
        if np.array_equal(self.blanks, self.goal_mask):
            # Terminal reward: normalize by total_time to avoid dominating intermediate rewards
            MAX_TIME = 1200.0
            total = min(self.total_time, MAX_TIME)
            # Scale terminal reward to be comparable to intermediate rewards
            reward = 10.0 - (total / MAX_TIME) * 10.0  # Range: [0, 10]
            self.done = True
        else:
            all_wait = (self.queues * self.mean_time_arr) / self.staff_arr
            valid_idx = np.where(mask_before[:21] == 1.0)[0]
            if len(valid_idx) > 0:
                avg_wait = float(np.mean(all_wait[valid_idx]))
            else:
                avg_wait = float(np.mean(all_wait))
            reward = 0.1 * (avg_wait - time_action) - 0.1
            self.done = False
        
        return self._get_state(), reward, self.done
