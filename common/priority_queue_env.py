import numpy as np
import pandas as pd
import json
import os

class PriorityQueueEnv:
    """
    Environment with Priority Queue for Emergency Patients:
    - Emergency patients appear randomly (use only 10 test actions: indices 11-20)
    - When emergency arrives: current queue → mock queue, emergency gets priority
    - After emergency done: restore mock queue → current queue
    - Penalty for assigning new patients to actions with active emergency
    - Goal: minimize avg time for all patients while ensuring emergency wait time ≈ 0
    """
    
    def __init__(self, state_size, action_size, data_path, activity_info_path='data/raw/activity_info.json'):
        self.state_size = state_size  # 86 dimensions
        self.action_size = action_size  # 22 (21 activities + 1 mock)
        self.WIN_REWARD = 100.0
        self.MOCK_PENALTY = -0.5
        self.EMERGENCY_PENALTY = -2.0  # Penalty for choosing action with emergency (increased)
        self.EMERGENCY_BONUS = 3.0     # Bonus for completing emergency (increased)
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
        print(f"📂 PriorityQueueEnv loading data: {data_path}")
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
        
        # Emergency test actions: indices 11-20 (10 actions)
        # Blood Test(19), Urine Test(14), Chest X-ray(11), CT(not in list), MRI(not in list),
        # General Ultrasound(13), ECG(16), ENT Endoscopy(15), Biopsy(not in list), PET(not in list)
        # Available in our 21 actions: 11,12,13,14,15,16,17,18,19,20
        self.emergency_test_actions = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        # Priority queue tracking
        self.emergency_flags = np.zeros(21, dtype=np.float32)  # 1 if emergency active
        self.mock_queues = np.zeros(21, dtype=np.float32)      # Store original queue when emergency
        
        # Statistics
        self.mock_action_count = 0
        self.emergency_count = 0
        self.emergency_wait_times = []
    
    def _get_state(self):
        """
        State: [gender, marital, 21 blanks, 21 queues, 21 emergency_flags, 21 mock_queues]
        Total: 2 + 21 + 21 + 21 + 21 = 86 dimensions
        """
        return np.concatenate((
            self.features,
            self.blanks,
            self.queues / 10.0,  # Normalize queues
            self.emergency_flags,
            self.mock_queues / 10.0  # Normalize mock queues
        ))
    
    def _get_queues_at(self, t):
        if t < self.max_trace_len:
            return self.queue_trace[t]
        return np.zeros(21)
    
    def _create_goal_mask(self):
        """Create goal mask based on patient features (gender, marital status)."""
        goal = np.ones(21, dtype=np.float32)
        gender = int(self.features[0])
        marital = int(self.features[1])
        
        # Gender-specific exclusions
        if gender == 0:  # Male
            goal[8] = 0.0  # Gynecological
            goal[9] = 0.0  # Breast
        
        # Marital status exclusions
        if marital == 0:  # Single
            goal[8] = 0.0  # Gynecological
        
        return goal
    
    def reset(self):
        """Reset environment for new episode."""
        self.done = False
        self.total_time = 0.0
        self.start_time_idx = 0
        
        # Random patient features
        self.features = np.array([
            np.random.randint(0, 2),  # gender: 0=male, 1=female
            np.random.randint(0, 2)   # marital: 0=single, 1=married
        ], dtype=np.float32)
        
        # Initialize blanks and goal
        self.blanks = np.zeros(21, dtype=np.float32)
        self.goal_mask = self._create_goal_mask()
        
        # Initialize queues
        self.queues = self._get_queues_at(self.start_time_idx).astype(np.float32)
        
        # Reset priority queue tracking
        self.emergency_flags = np.zeros(21, dtype=np.float32)
        self.mock_queues = np.zeros(21, dtype=np.float32)
        
        return self._get_state()
    
    def _simulate_emergency(self):
        """
        Randomly simulate emergency patient arrival.
        5% chance per step that emergency appears on one of the test actions.
        """
        if np.random.random() < 0.05:  # 5% chance
            # Choose random test action for emergency
            emergency_action = np.random.choice(self.emergency_test_actions)
            
            # Only trigger if not already in emergency
            if self.emergency_flags[emergency_action] == 0:
                # Move current queue to mock queue
                self.mock_queues[emergency_action] = self.queues[emergency_action]
                
                # Set emergency queue (1 emergency patient)
                self.queues[emergency_action] = 1.0
                
                # Mark as emergency active
                self.emergency_flags[emergency_action] = 1.0
                
                self.emergency_count += 1
                return emergency_action
        
        return None
    
    def _resolve_emergency(self, action):
        """
        Check if emergency is resolved for given action.
        Emergency resolved when queue becomes 0.
        """
        if self.emergency_flags[action] == 1.0 and self.queues[action] == 0:
            # Restore original queue from mock
            self.queues[action] = self.mock_queues[action]
            self.mock_queues[action] = 0.0
            self.emergency_flags[action] = 0.0
            return True
        return False
    
    def get_action_mask(self):
        """
        Return valid action mask.
        - Mask completed actions (blanks == 1)
        - Mask actions violating dependencies
        - Mask actions violating gender/marital constraints
        - Do NOT mask emergency actions (allow agent to handle them)
        - Enable mock action if all others masked
        """
        mask = np.ones(self.action_size, dtype=np.float32)
        
        # Mask completed actions
        mask[:21] = 1.0 - self.blanks
        
        # Apply goal mask (gender/marital constraints)
        mask[:21] *= self.goal_mask
        
        # Dependency rules (same as StaticQueueEnv)
        # Registration (0) must be done first
        if self.blanks[0] == 0:
            mask[1:21] = 0.0
        
        # Payment (20) must be done last
        if not np.array_equal(self.blanks[:20], self.goal_mask[:20]):
            mask[20] = 0.0
        
        # Mock action (21): only if all real actions masked
        if np.sum(mask[:21]) > 0:
            mask[21] = 0.0
        else:
            mask[21] = 1.0
        
        return mask
    
    def step(self, action):
        """Execute action and return next state, reward, done."""
        action = int(action)
        mask_before = self.get_action_mask()
        
        # Simulate emergency arrival
        emergency_action = self._simulate_emergency()
        
        # Handle mock action
        if action == 21:
            self.total_time += 1.0
            self.mock_action_count += 1
            reward = self.MOCK_PENALTY
            self.done = False
            return self._get_state(), reward, self.done
        
        # Calculate time for chosen action (waiting time + processing time)
        current_q = self.queues[action]
        
        # If emergency active on this action, normal patients must wait for both emergency + mock queue
        if self.emergency_flags[action] == 1.0:
            # Emergency patient: only wait for emergency queue (priority)
            # Normal patient: wait for emergency queue + mock queue (must wait for emergency to finish first)
            # We don't know if current patient is emergency or normal here, so we use worst case (emergency + mock)
            total_queue = current_q + self.mock_queues[action]
        else:
            total_queue = current_q
        
        time_action = round((total_queue * self.mean_time_arr[action]) / self.staff_arr[action], 2)
        
        # Check if choosing action with active emergency
        emergency_penalty = 0.0
        if self.emergency_flags[action] == 1.0:
            emergency_penalty = self.EMERGENCY_PENALTY
        
        # Update state
        self.blanks[action] = 1.0
        self.total_time += time_action
        
        # Time Travel: update queue based on elapsed time
        current_trace_idx = self.start_time_idx + int(self.total_time)
        self.queues = self._get_queues_at(current_trace_idx).astype(np.float32)
        
        # Check if emergency resolved
        emergency_bonus = 0.0
        if self._resolve_emergency(action):
            emergency_bonus = self.EMERGENCY_BONUS
            self.emergency_wait_times.append(0.0)  # Emergency handled immediately
        
        # Reward Calculation
        if np.array_equal(self.blanks, self.goal_mask):
            # Terminal reward
            MAX_TIME = 1200.0
            total = min(self.total_time, MAX_TIME)
            reward = 10.0 - (total / MAX_TIME) * 10.0
            self.done = True
        else:
            # Intermediate reward
            all_wait = (self.queues * self.mean_time_arr) / self.staff_arr
            valid_idx = np.where(mask_before[:21] == 1.0)[0]
            if len(valid_idx) > 0:
                avg_wait = float(np.mean(all_wait[valid_idx]))
            else:
                avg_wait = float(np.mean(all_wait))
            
            reward = 0.1 * (avg_wait - time_action) - 0.1
            reward += emergency_penalty  # Penalty for choosing emergency action
            reward += emergency_bonus    # Bonus for resolving emergency
            self.done = False
        
        return self._get_state(), reward, self.done
