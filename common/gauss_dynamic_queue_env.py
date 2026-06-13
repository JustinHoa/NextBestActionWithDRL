import numpy as np
import pandas as pd
import json
import os

class GaussDynamicQueueEnv:
    """
    Environment with Gauss Dynamic Queue Constraints:
    - Capacity for each concurrent activity follows Gaussian distribution
    - Mean = 2 × resource, Std = 0.5 × resource
    - Capacity is sampled at reset and remains fixed during episode
    - Capacity is clipped to [1 × resource, 3 × resource]
    - Applies only to concurrent activities (Cluster 1 & 2)
    - No mock action needed
    """
    
    def __init__(self, state_size, action_size, data_path, activity_info_path='data/raw/activity_info.json'):
        self.state_size = state_size  # 87 dimensions
        self.action_size = action_size  # 21 activities (no mock)
        self.WIN_REWARD = 100.0
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
        print(f"📂 GaussDynamicQueueEnv loading data: {data_path}")
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
        
        # Gauss Dynamic Queue: capacity follows Gaussian distribution
        self.gauss_mean_multiplier = 2.0  # Mean at 2 × resource
        self.gauss_std_multiplier = 0.5   # Std = 0.5 × resource
        self.current_max_queue = np.zeros(21)
        
        # Define concurrent activity groups (indices 0-20)
        self.cluster1_indices = [5, 6, 7, 8, 9]
        self.cluster2_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.concurrent_indices = self.cluster1_indices + self.cluster2_indices
        
        # Statistics
        self.capacity_samples = np.zeros(21)  # Track sampled capacities
    
    def _sample_gauss_capacity(self):
        """
        Sample capacity from Gaussian distribution for concurrent activities.
        Non-concurrent activities use fixed 2 × resource.
        """
        capacity = np.zeros(21)
        
        for i in range(21):
            if i in self.concurrent_indices:
                # Sample from Gaussian: N(2×resource, 0.5×resource)
                mean = self.gauss_mean_multiplier * self.staff_arr[i]
                std = self.gauss_std_multiplier * self.staff_arr[i]
                sampled = np.random.normal(mean, std)
                
                # Clip to [1×resource, 3×resource]
                min_cap = 1.0 * self.staff_arr[i]
                max_cap = 3.0 * self.staff_arr[i]
                capacity[i] = np.clip(sampled, min_cap, max_cap)
            else:
                # Non-concurrent activities: fixed 2 × resource
                capacity[i] = 2.0 * self.staff_arr[i]
        
        return capacity
    
    def _get_state(self):
        """
        State: [gender, marital, 21 blanks, 21 waiting_times, 21 queue_utilization, 21 capacity_ratio, 1 norm_time]
        Total: 2 + 21 + 21 + 21 + 21 + 1 = 87 dimensions
        """
        estimated_wait_time = (self.queues * self.mean_time_arr) / self.staff_arr
        norm_wait_time = estimated_wait_time / 200.0
        
        # Queue utilization: current_queue / current_max_queue (0-1)
        queue_utilization = self.queues / np.maximum(self.current_max_queue, 1.0)
        queue_utilization = np.clip(queue_utilization, 0.0, 1.0)
        
        # Capacity ratio: current_max_queue / (2 × resource) normalized
        # This tells agent how much capacity is available relative to baseline
        capacity_ratio = self.current_max_queue / (2.0 * self.staff_arr)
        capacity_ratio = np.clip(capacity_ratio, 0.5, 1.5)  # [1×, 3×] → [0.5, 1.5]
        capacity_ratio = (capacity_ratio - 0.5) / 1.0  # Normalize to [0, 1]
        
        norm_current_time = np.array([self.total_time / self.max_trace_len])
        
        return np.concatenate((
            self.features,
            self.blanks,
            norm_wait_time,
            queue_utilization,
            capacity_ratio,
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
        
        # Sample Gaussian capacity for this episode
        self.current_max_queue = self._sample_gauss_capacity()
        self.capacity_samples = self.current_max_queue.copy()
        
        return self._get_state()
    
    def get_action_mask(self):
        """
        Action mask with Gaussian capacity (fixed per episode).
        """
        # Start with 21 actions
        mask = np.ones(21, dtype=np.float32)
        
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
        
        # Queue capacity constraints (Gaussian sampled capacity)
        for i in range(21):
            if mask[i] == 1.0:  # Only check available actions
                if self.queues[i] >= self.current_max_queue[i]:
                    mask[i] = 0.0
        
        return mask
    
    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, self.done
        
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
            MAX_TIME = 1200.0
            total = min(self.total_time, MAX_TIME)
            reward = 10.0 - (total / MAX_TIME) * 10.0
            self.done = True
        else:
            all_wait = (self.queues * self.mean_time_arr) / self.staff_arr
            valid_idx = np.where(mask_before == 1.0)[0]
            if len(valid_idx) > 0:
                avg_wait = float(np.mean(all_wait[valid_idx]))
            else:
                avg_wait = float(np.mean(all_wait))
            reward = 0.1 * (avg_wait - time_action) - 0.1
            self.done = False
        
        return self._get_state(), reward, self.done
