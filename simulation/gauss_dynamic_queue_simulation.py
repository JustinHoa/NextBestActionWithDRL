import simpy
import random
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.utils import DEVICE

# --- CONFIG ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
EVAL_DATA_DIR = os.path.join(ROOT_DIR, "data", "evaluate")
ACTIVITY_INFO_PATH = os.path.join(RAW_DATA_DIR, "activity_info.json")

# Load global simulation data
with open(ACTIVITY_INFO_PATH, 'r', encoding='utf-8') as f:
    activity_info = json.load(f)

id_to_name = {info['id']: name for name, info in activity_info.items()}
name_to_id = {name: info['id'] for name, info in activity_info.items()}
activity_names_list = [id_to_name[i] for i in range(1, 22)]

# Precompute arrays
mean_times_arr = []
staff_arr = []
for name in activity_names_list:
    info = activity_info[name]
    staff_arr.append(info['staff'])
    if name == "Payment":
        m = (info["mean_time"]["Cash"] + info["mean_time"]["Credit"]) / 2
    else:
        m = info["mean_time"]
    mean_times_arr.append(m)
mean_times_arr = np.array(mean_times_arr)
staff_arr = np.array(staff_arr)

# Cluster definitions
CLUSTER_1_INDICES = [5, 6, 7, 8, 9]
CLUSTER_2_INDICES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


class GaussDynamicQueueHealthCenter:
    """Health center with Gauss Dynamic Queue: capacity follows Gaussian distribution."""
    
    def __init__(self, env):
        self.env = env
        self.resources = {name: simpy.Resource(env, data["staff"]) for name, data in activity_info.items()}
        self.current_patient_count = {name: 0 for name in activity_info}
        
        # Gaussian capacity: N(2×resource, 0.5×resource), clipped to [1×, 3×]
        self.current_max_queue = self._sample_gauss_capacity()
        
        self.finished_patient_count = 0
    
    def _sample_gauss_capacity(self):
        """Sample capacity from Gaussian distribution for concurrent activities."""
        capacity = {}
        for name, data in activity_info.items():
            idx = name_to_id[name] - 1
            if idx in CLUSTER_1_INDICES or idx in CLUSTER_2_INDICES:
                # Concurrent activities: Gaussian
                mean = 2.0 * data["staff"]
                std = 0.5 * data["staff"]
                sampled = np.random.normal(mean, std)
                min_cap = 1.0 * data["staff"]
                max_cap = 3.0 * data["staff"]
                capacity[name] = np.clip(sampled, min_cap, max_cap)
            else:
                # Non-concurrent: fixed 2×
                capacity[name] = 2.0 * data["staff"]
        return capacity
    
    def get_queue_status_array(self):
        return np.array([self.current_patient_count[name] for name in activity_names_list])
    
    def get_current_max_queue_array(self):
        return np.array([self.current_max_queue[name] for name in activity_names_list])
    
    def is_queue_full(self, activity_name):
        """Check if queue for this activity is full."""
        return self.current_patient_count[activity_name] >= self.current_max_queue[activity_name]


class GaussDynamicQueueCoordinator:
    def __init__(self, agent=None, total_sim_time=1440):
        self.agent = agent
        self.total_sim_time = total_sim_time
    
    def predict(self, patient, current_queues_arr, current_time, candidates_names: list, center):
        """Predict next action with Gaussian capacity."""
        candidate_indices = [name_to_id[name] - 1 for name in candidates_names]
        
        # Get available candidates (not full)
        available_candidates = [name for name in candidates_names if not center.is_queue_full(name)]
        
        # If no available candidates, choose randomly (shouldn't happen with Gaussian)
        if not available_candidates:
            available_candidates = candidates_names
        
        # Random agent
        if self.agent is None:
            return random.choice(available_candidates)
        
        # Agent Logic
        raw_wait = (current_queues_arr * mean_times_arr) / staff_arr
        norm_wait = raw_wait / 200.0
        
        # Queue utilization
        current_max_queue_arr = center.get_current_max_queue_array()
        queue_utilization = current_queues_arr / np.maximum(current_max_queue_arr, 1.0)
        queue_utilization = np.clip(queue_utilization, 0.0, 1.0)
        
        # Capacity ratio
        capacity_ratio = current_max_queue_arr / (2.0 * staff_arr)
        capacity_ratio = np.clip(capacity_ratio, 0.5, 1.5)
        capacity_ratio = (capacity_ratio - 0.5) / 1.0
        
        norm_time = np.array([current_time / self.total_sim_time])
        
        # Encode Patient Info
        gender_code = 1 if patient.gender == "Male" else 0
        marital_code = 1 if patient.marital == "Married" else 0
        
        # State: [gender, marital, 21 blanks, 21 waiting_times, 21 queue_utilization, 21 capacity_ratio, 1 norm_time]
        state = np.concatenate(([gender_code, marital_code], patient.prefix, norm_wait, queue_utilization, capacity_ratio, norm_time))
        
        # Create Mask (21 actions)
        mask = np.zeros(21)
        available_indices = [name_to_id[name] - 1 for name in available_candidates]
        for idx in available_indices:
            mask[idx] = 1.0
        
        # Agent Act
        action_idx = self.agent.act(state, mask, eps=0.0)
        
        # Validate action
        if action_idx not in available_indices:
            return random.choice(available_candidates)
        
        return activity_names_list[action_idx]


class GaussDynamicQueuePatient:
    def __init__(self, env, center, pid, gender, marital, event_log):
        self.env = env
        self.center = center
        self.id = pid
        self.gender = gender
        self.marital = marital
        self.event_log = event_log
        self.prefix = np.zeros(21)
        self.start_time = 0
        self.end_time = 0
    
    def do_activity(self, activity_name):
        """Perform an activity."""
        self.event_log.append({"CaseID": self.id, "Activity": activity_name, "Timestamp": self.env.now, "Lifecycle": "START"})
        
        res = self.center.resources[activity_name]
        self.center.current_patient_count[activity_name] += 1
        
        with res.request() as req:
            yield req
            
            data = activity_info[activity_name]
            if isinstance(data["mean_time"], dict):
                mean = (data["mean_time"]["Cash"] + data["mean_time"]["Credit"]) / 2
            else:
                mean = data["mean_time"]
            duration = random.triangular(mean * 0.8, mean * 1.2, mean)
            
            yield self.env.timeout(duration)
            
            idx = name_to_id[activity_name] - 1
            self.prefix[idx] = 1
            self.center.current_patient_count[activity_name] -= 1
            
            self.event_log.append({"CaseID": self.id, "Activity": activity_name, "Timestamp": self.env.now, "Lifecycle": "COMPLETE"})
    
    def go_process(self, coordinator):
        self.start_time = self.env.now
        
        CLUSTER_1 = ["Eye Examination", "ENT Examination", "Dental Examination", "Gynecological Examination", "Breast Examination"]
        CLUSTER_2 = ["DEXA Bone Density Scan", "Chest X-ray", "In-depth Eye Examination", "General Ultrasound", 
                     "Urine Test", "ENT Endoscopy", "Electrocardiogram (ECG)", "Post-bronchodilator Spirometry", 
                     "Cardiac Ultrasound", "Blood Test"]
        
        # Filter based on gender/marital
        c1 = [a for a in CLUSTER_1 if not ((a == "Gynecological Examination" and (self.gender == "Male" or self.marital == "Single")) 
                                           or (a == "Breast Examination" and self.gender == "Male"))]
        c2 = list(CLUSTER_2)
        
        # Sequential start
        for act in ["Registration", "Payment", "Get Triage Number", "Measure Vital Signs", "General Medicine Examination"]:
            yield self.env.process(self.do_activity(act))
        
        # Cluster 1 with Gaussian capacity
        while c1:
            q = self.center.get_queue_status_array()
            act = coordinator.predict(self, q, self.env.now, c1, self.center)
            c1.remove(act)
            yield self.env.process(self.do_activity(act))
        
        # Cluster 2 with Gaussian capacity
        while c2:
            q = self.center.get_queue_status_array()
            act = coordinator.predict(self, q, self.env.now, c2, self.center)
            c2.remove(act)
            yield self.env.process(self.do_activity(act))
        
        # Conclusion
        yield self.env.process(self.do_activity("Conclusion"))
        self.end_time = self.env.now
        self.center.finished_patient_count += 1


def run_gauss_dynamic_queue_simulation(num_patients=200, agent=None, version_output="gauss_dynamic_queue", seed=None, model_name="GaussDynamicQueueDQN", gen_id=0):
    """Run simulation with Gaussian dynamic queue constraints."""
    env = simpy.Environment()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    center = GaussDynamicQueueHealthCenter(env)
    coord = GaussDynamicQueueCoordinator(agent, total_sim_time=1440)
    
    event_logs = []
    queue_logs = []
    patients_list = []
    
    def patient_gen(patient_list_ref):
        for i in range(num_patients):
            p = GaussDynamicQueuePatient(env, center, i, random.choice(["Male", "Female"]), random.choice(["Single", "Married"]), event_logs)
            patient_list_ref.append(p)
            env.process(p.go_process(coord))
            yield env.timeout(random.expovariate(1.0 / 2.0))
    
    def monitor():
        while center.finished_patient_count < num_patients:
            status = center.current_patient_count.copy()
            status["Time"] = env.now
            queue_logs.append(status)
            yield env.timeout(1.0)
    
    env.process(patient_gen(patients_list))
    env.process(monitor())
    env.run()
    
    # Save logs
    if len(queue_logs) > 0:
        df_q = pd.DataFrame(queue_logs)
        cols = ['Time'] + [name for name in activity_names_list if name in df_q.columns]
        df_q = df_q[cols]
        q_path = os.path.join(RAW_DATA_DIR, f"queue_log_{num_patients}_{model_name}_gen_{gen_id}.csv")
        df_q.to_csv(q_path, index=False)
        print(f"✅ Saved QueueLog: {q_path}")
    
    if len(event_logs) > 0:
        df_e = pd.DataFrame(event_logs)
        e_path = os.path.join(EVAL_DATA_DIR, f"event_log_{num_patients}_{model_name}_gen_{gen_id}.csv")
        df_e.to_csv(e_path, index=False)
        print(f"✅ Saved EventLog: {e_path}")
    
    # Statistics
    total_times = [p.end_time - p.start_time for p in patients_list if hasattr(p, 'end_time') and p.end_time > 0]
    avg_time = np.mean(total_times) if total_times else 0
    
    # Calculate capacity statistics
    max_queue_arr = center.get_current_max_queue_array()
    capacity_ratios = max_queue_arr / (2.0 * staff_arr)
    
    print(f"✅ Gauss Dynamic Queue Simulation finished.")
    print(f"   Avg Time: {avg_time:.2f} mins")
    print(f"   Capacity Range: [{capacity_ratios.min():.2f}×, {capacity_ratios.max():.2f}×]")
    print(f"   Capacity Mean: {capacity_ratios.mean():.2f}×")
    
    return avg_time
