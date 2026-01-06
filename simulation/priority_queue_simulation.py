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

# Emergency test actions (indices 11-20 in 0-indexed)
EMERGENCY_TEST_ACTIONS = ["Chest X-ray", "In-depth Eye Examination", "General Ultrasound", 
                          "Urine Test", "ENT Endoscopy", "Electrocardiogram (ECG)", 
                          "Post-bronchodilator Spirometry", "Cardiac Ultrasound", "Blood Test", "DEXA Bone Density Scan"]


class PriorityQueueHealthCenter:
    """Health center with emergency priority queue."""
    def __init__(self, env):
        self.env = env
        # Create resources with priority queue (lower priority number = higher priority)
        self.resources = {name: simpy.PriorityResource(env, data["staff"]) for name, data in activity_info.items()}
        self.current_patient_count = {name: 0 for name in activity_info}
        self.finished_patient_count = 0
        self.emergency_count = 0
    
    def get_queue_status_array(self):
        return np.array([self.current_patient_count[name] for name in activity_names_list])


class PriorityQueueCoordinator:
    """Coordinator for priority queue with agent."""
    def __init__(self, agent=None, total_sim_time=1440):
        self.agent = agent
        self.total_sim_time = total_sim_time
    
    def predict(self, patient, current_queues_arr, current_time, candidates_names: list):
        """Predict next action."""
        if not candidates_names:
            return None
        
        # Random agent
        if self.agent is None:
            return random.choice(candidates_names)
        
        # Agent Logic (using gym-style state from PriorityQueueEnv)
        # State: [gender, marital, 21 blanks, 21 queues, 21 emergency_flags, 21 mock_queues]
        gender_code = 1.0 if patient.gender == "Male" else 0.0
        marital_code = 1.0 if patient.marital == "Married" else 0.0
        
        # Normalize queues
        norm_queues = current_queues_arr / 10.0
        
        # Emergency flags (0 for now, will be updated in real-time)
        emergency_flags = np.zeros(21)
        mock_queues = np.zeros(21)
        
        state = np.concatenate(([gender_code, marital_code], patient.prefix, norm_queues, emergency_flags, mock_queues))
        
        # Create mask (22 actions: 21 activities + 1 mock)
        mask = np.zeros(22)
        candidate_indices = [name_to_id[name] - 1 for name in candidates_names]
        for idx in candidate_indices:
            mask[idx] = 1.0
        
        # Agent act
        action_idx = self.agent.act(state, mask, eps=0.0)
        
        # If agent chose mock action (index 21) or invalid action
        if action_idx == 21 or action_idx not in candidate_indices:
            return None  # Will trigger mock wait
        
        return activity_names_list[action_idx]


class PriorityQueuePatient:
    """Patient with emergency priority support."""
    def __init__(self, env, center, pid, gender, marital, event_log, is_emergency=False):
        self.env = env
        self.center = center
        self.id = pid
        self.gender = gender
        self.marital = marital
        self.event_log = event_log
        self.is_emergency = is_emergency
        self.prefix = np.zeros(21)
        self.start_time = 0
        self.end_time = 0
        self.emergency_wait_time = 0.0  # Track waiting time for emergency patients
    
    def do_activity(self, activity_name, coordinator):
        """Perform an activity with priority queue support."""
        self.event_log.append({"CaseID": self.id, "Activity": activity_name, "Timestamp": self.env.now, "Lifecycle": "START"})
        
        res = self.center.resources[activity_name]
        self.center.current_patient_count[activity_name] += 1
        
        # Priority: 0 for emergency (highest), 1 for normal
        priority = 0 if (self.is_emergency and activity_name in EMERGENCY_TEST_ACTIONS) else 1
        
        # Track wait time for emergency patients
        wait_start = self.env.now if self.is_emergency and activity_name in EMERGENCY_TEST_ACTIONS else None
        
        with res.request(priority=priority) as req:
            yield req
            
            # Calculate actual wait time for emergency patients
            if wait_start is not None:
                self.emergency_wait_time += (self.env.now - wait_start)
            
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
    
    def do_mock_wait(self):
        """Wait 1 minute when no valid actions."""
        self.event_log.append({"CaseID": self.id, "Activity": "MOCK_WAIT", "Timestamp": self.env.now, "Lifecycle": "START"})
        yield self.env.timeout(1.0)
        self.event_log.append({"CaseID": self.id, "Activity": "MOCK_WAIT", "Timestamp": self.env.now, "Lifecycle": "COMPLETE"})
    
    def go_process(self, coordinator):
        """Patient workflow with agent coordination."""
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
            yield self.env.process(self.do_activity(act, coordinator))
        
        # Cluster 1 with agent coordination
        while c1:
            q = self.center.get_queue_status_array()
            act = coordinator.predict(self, q, self.env.now, c1)
            
            if act is None:
                yield self.env.process(self.do_mock_wait())
                continue
            
            c1.remove(act)
            yield self.env.process(self.do_activity(act, coordinator))
        
        # Cluster 2 with agent coordination
        while c2:
            q = self.center.get_queue_status_array()
            act = coordinator.predict(self, q, self.env.now, c2)
            
            if act is None:
                yield self.env.process(self.do_mock_wait())
                continue
            
            c2.remove(act)
            yield self.env.process(self.do_activity(act, coordinator))
        
        # Conclusion
        yield self.env.process(self.do_activity("Conclusion", coordinator))
        self.end_time = self.env.now
        self.center.finished_patient_count += 1


def run_priority_queue_simulation(num_patients=200, agent=None, version_output="priority_queue", seed=None, model_name="PriorityQueueDQN", gen_id=0):
    """Run simulation with priority queue for emergency patients."""
    env = simpy.Environment()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    center = PriorityQueueHealthCenter(env)
    coord = PriorityQueueCoordinator(agent, total_sim_time=1440)
    
    event_logs = []
    queue_logs = []
    patients_list = []
    
    def patient_gen(patient_list_ref):
        for i in range(num_patients):
            # 10% chance of emergency patient
            is_emergency = random.random() < 0.10
            if is_emergency:
                center.emergency_count += 1
            
            p = PriorityQueuePatient(env, center, i, random.choice(["Male", "Female"]), 
                                    random.choice(["Single", "Married"]), event_logs, is_emergency)
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
        q_path = os.path.join(RAW_DATA_DIR, f"queue_log_{num_patients}_priorityqueue_{version_output}.csv")
        df_q.to_csv(q_path, index=False)
        print(f"✅ Saved QueueLog: {q_path}")
    
    if len(event_logs) > 0:
        df_e = pd.DataFrame(event_logs)
        e_path = os.path.join(EVAL_DATA_DIR, f"event_log_{num_patients}_priorityqueue_{version_output}.csv")
        df_e.to_csv(e_path, index=False)
        print(f"✅ Saved EventLog: {e_path}")
    
    # Calculate statistics - separate normal vs emergency
    normal_patients = [p for p in patients_list if not p.is_emergency and hasattr(p, 'end_time') and p.end_time > 0]
    emergency_patients = [p for p in patients_list if p.is_emergency and hasattr(p, 'end_time') and p.end_time > 0]
    
    normal_total_times = [p.end_time - p.start_time for p in normal_patients]
    emergency_total_times = [p.end_time - p.start_time for p in emergency_patients]
    emergency_wait_times = [p.emergency_wait_time for p in emergency_patients]
    
    avg_normal_time = np.mean(normal_total_times) if normal_total_times else 0.0
    avg_emergency_time = np.mean(emergency_total_times) if emergency_total_times else 0.0
    avg_emergency_wait = np.mean(emergency_wait_times) if emergency_wait_times else 0.0
    avg_overall_time = np.mean(normal_total_times + emergency_total_times) if (normal_total_times or emergency_total_times) else 0.0
    
    print(f"\n✅ Priority Queue Simulation finished.")
    print(f"📊 RESULTS:")
    print(f"   Overall Avg Time: {avg_overall_time:.2f} mins")
    print(f"   Normal Patients: {len(normal_patients)} | Avg Total Time: {avg_normal_time:.2f} mins")
    print(f"   Emergency Patients: {len(emergency_patients)} | Avg Total Time: {avg_emergency_time:.2f} mins")
    print(f"   Emergency Avg Waiting Time: {avg_emergency_wait:.2f} mins ")
    print(f"   Total Emergency Count: {center.emergency_count}")
    
    # Return metrics dictionary
    return {
        'overall_avg_time': avg_overall_time,
        'normal_avg_time': avg_normal_time,
        'emergency_avg_time': avg_emergency_time,
        'emergency_avg_wait': avg_emergency_wait,
        'normal_count': len(normal_patients),
        'emergency_count': len(emergency_patients)
    }


if __name__ == "__main__":
    # Test simulation
    print("Testing Priority Queue Simulation...")
    avg_time = run_priority_queue_simulation(
        num_patients=200,
        agent=None,
        version_output="test",
        seed=42,
        model_name="Random",
        gen_id=0
    )
    print(f"\nTest completed. Avg time: {avg_time:.2f} mins")
