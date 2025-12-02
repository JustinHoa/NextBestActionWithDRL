import simpy
import random
import time
import json
import pandas as pd
import numpy as np
import torch
import os
import sys

# Thêm đường dẫn root để import được package 'agents' và 'common'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.utils import DEVICE, STATE_SIZE, ACTION_SIZE
# Import tất cả các agent để dễ dàng thay đổi
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_agent import DuelingAgent
from agents.rainbow_agent import RainbowAgent

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

# Precompute arrays for faster calc
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


# --- CLASSES ---
class HealthCheckCenter:
    def __init__(self, env):
        self.env = env
        self.resources = {name: simpy.Resource(env, data["staff"]) for name, data in activity_info.items()}
        self.current_patient_count = {name: 0 for name in activity_info}
        self.finished_patient_count = 0

    def get_queue_status_array(self):
        return np.array([self.current_patient_count[name] for name in activity_names_list])

class Coordinator:
    def __init__(self, agent=None, total_sim_time=1440):
        self.agent = agent
        self.total_sim_time = total_sim_time

    def predict(self, patient, current_queues_arr, current_time, candidates_names):
        candidate_indices = [name_to_id[name] - 1 for name in candidates_names]
        
        # Nếu không có agent, chọn ngẫu nhiên
        if self.agent is None:
            return random.choice(candidates_names)

        # Agent Logic
        raw_wait = (current_queues_arr * mean_times_arr) / staff_arr
        norm_wait = raw_wait / 200.0 
        norm_time = np.array([current_time / self.total_sim_time])
        
        # Encode Patient Info
        gender_code = 1 if patient.gender == "Male" else 0
        marital_code = 1 if patient.marital_status == "Married" else 0
        
        state = np.concatenate(([gender_code, marital_code], patient.prefix, norm_wait, norm_time))
        
        # Create Mask
        mask = np.zeros(ACTION_SIZE)
        for idx in candidate_indices: mask[idx] = 1.0
        
        # Agent Act (eps=0 for pure exploitation)
        action_idx = self.agent.act(state, mask, eps=0.0)
        
        if action_idx not in candidate_indices:
            # Fallback an toàn: nếu agent chọn sai, chọn ngẫu nhiên từ các lựa chọn hợp lệ
            return random.choice(candidates_names)
            
        return activity_names_list[action_idx]

class Patient:
    def __init__(self, env, center, pid, gender, marital, event_log):
        self.env = env
        self.center = center
        self.id = pid
        self.gender = gender
        self.marital = marital
        self.event_log = event_log # List to store logs
        self.prefix = np.zeros(21)
        self.lab_results = {"blood": -1, "urine": -1}

    def do_activity(self, activity_name):
        # Log Start
        self.event_log.append({"CaseID": self.id, "Activity": activity_name, "Timestamp": self.env.now, "Lifecycle": "START"})
        
        res = self.center.resources[activity_name]
        self.center.current_patient_count[activity_name] += 1
        
        with res.request() as req:
            yield req
            # Processing time calculation
            data = activity_info[activity_name]
            if isinstance(data["mean_time"], dict):
                 mean = (data["mean_time"]["Cash"] + data["mean_time"]["Credit"])/2
            else: mean = data["mean_time"]
            duration = random.triangular(mean*0.8, mean*1.2, mean)
            
            yield self.env.timeout(duration)
            
            # Lab results logic
            if activity_name == "Blood Test": self.lab_results["blood"] = self.env.now + 60
            if activity_name == "Urine Test": self.lab_results["urine"] = self.env.now + 45
            
            idx = name_to_id[activity_name] - 1
            self.prefix[idx] = 1
            self.center.current_patient_count[activity_name] -= 1
            
            # Log Complete
            self.event_log.append({"CaseID": self.id, "Activity": activity_name, "Timestamp": self.env.now, "Lifecycle": "COMPLETE"})

    def go_process(self, coordinator):
        # ... (Logic luồng đi khám giữ nguyên như cũ, chỉ rút gọn cho ngắn) ...
        # Định nghĩa Group
        CLUSTER_1 = ["Eye Examination", "ENT Examination", "Dental Examination", "Gynecological Examination", "Breast Examination"]
        CLUSTER_2 = ["Blood Test", "Urine Test", "General Ultrasound", "Cardiac Ultrasound", "Chest X-ray"]
        
        # Filter candidates based on gender/marital
        c1 = [a for a in CLUSTER_1 if not ((a=="Gynecological Examination" and (self.gender=="Male" or self.marital=="Single")) or (a=="Breast Examination" and self.gender=="Male"))]
        c2 = list(CLUSTER_2)

        # 1. Start Seq
        for act in ["Registration", "Payment", "Measure Vital Signs", "General Medicine Examination"]:
            yield self.env.process(self.do_activity(act))

        # 2. Dynamic choice C1
        while c1:
            q = self.center.get_queue_status_array()
            act = coordinator.predict(self, q, self.env.now, c1)
            c1.remove(act)
            yield self.env.process(self.do_activity(act))

        # 3. Dynamic choice C2
        while c2:
            q = self.center.get_queue_status_array()
            act = coordinator.predict(self, q, self.env.now, c2)
            c2.remove(act)
            yield self.env.process(self.do_activity(act))
            
        # 4. End Seq
        yield self.env.process(self.do_activity("Conclusion"))
        self.center.finished_patient_count += 1

def run_simulation(num_patients=200, agent=None, version_output="0", is_model_run=False):
    env = simpy.Environment()
    center = HealthCheckCenter(env)
    coord = Coordinator(agent, total_sim_time=1440)
    
    event_logs = []
    queue_logs = []
    
    # Generator
    def patient_gen():
        for i in range(num_patients):
            p = Patient(env, center, i, random.choice(["Male", "Female"]), random.choice(["Single", "Married"]), event_logs)
            env.process(p.go_process(coord))
            yield env.timeout(random.expovariate(1.0/2.0)) # Arrive every ~2 mins

    # Monitor
    def monitor():
        while center.finished_patient_count < num_patients:
            status = center.current_patient_count.copy()
            status["Time"] = env.now
            queue_logs.append(status)
            yield env.timeout(1.0)

    env.process(patient_gen())
    env.process(monitor())
    env.run()

    # Chỉ lưu queue log nếu đây là lần chạy để sinh data cho thế hệ tiếp theo
    if is_model_run:
        df_q = pd.DataFrame(queue_logs)
        q_path = os.path.join(RAW_DATA_DIR, f"200_queue_log_version_{version_output}.csv")
        df_q.to_csv(q_path, index=False)
        print(f"✅ Saved new QueueLog for next generation: {q_path}")

    # Luôn lưu event log để đánh giá
    if len(event_logs) > 0:
        df_e = pd.DataFrame(event_logs)
        e_path = os.path.join(EVAL_DATA_DIR, f"event_log_version_{version_output}.csv")
        df_e.to_csv(e_path, index=False)
        print(f"✅ Saved EventLog for evaluation: {e_path}")
    
    print(f"✅ Simulation Version '{version_output}' finished.")
    
    # Trả về thời gian khám trung bình để so sánh
    total_times = [p.end_time - p.start_time for p in env.process(patient_gen()) if p.end_time > 0]
    return np.mean(total_times) if total_times else 0

if __name__ == "__main__":
    NUM_PATIENTS = 200
    MODEL_PATH = os.path.join(ROOT_DIR, "logs/DDQN/final_gen3.pth") # <-- THAY ĐỔI MODEL Ở ĐÂY
    
    print("--- RUNNING SIMULATION FOR EVALUATION ---")
    
    # 1. Chạy mô phỏng ngẫu nhiên (Baseline)
    print("\n[1/2] Running RANDOM simulation (Baseline)...")
    start_time = time.time()
    random_avg_time = run_simulation(NUM_PATIENTS, agent=None, version_output="random_baseline")
    print(f"-> Finished in {time.time() - start_time:.2f}s. Average time: {random_avg_time:.2f} mins.")

    # 2. Chạy mô phỏng với Model DRL
    if os.path.exists(MODEL_PATH):
        print(f"\n[2/2] Running DRL AGENT simulation from: {MODEL_PATH}")
        # Tự động xác định agent class từ đường dẫn
        algo_name = MODEL_PATH.split(os.sep)[-2]
        if 'duel' in algo_name.lower(): agent = DuelingAgent(STATE_SIZE, ACTION_SIZE)
        elif 'rainbow' in algo_name.lower(): agent = RainbowAgent(STATE_SIZE, ACTION_SIZE)
        elif 'ddqn' in algo_name.lower(): agent = DDQNAgent(STATE_SIZE, ACTION_SIZE)
        else: agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        
        agent.load(MODEL_PATH)
        
        start_time = time.time()
        model_avg_time = run_simulation(NUM_PATIENTS, agent=agent, version_output="model_run")
        print(f"-> Finished in {time.time() - start_time:.2f}s. Average time: {model_avg_time:.2f} mins.")
        
        # 3. So sánh kết quả
        print("\n--- COMPARISON ---")
        print(f"Random Agent Avg Time: {random_avg_time:.2f} minutes")
        print(f"DRL Agent Avg Time:    {model_avg_time:.2f} minutes")
        improvement = ((random_avg_time - model_avg_time) / random_avg_time) * 100
        print(f"Improvement: {improvement:+.2f}%")
    else:
        print(f"\n⚠️ Model file not found at '{MODEL_PATH}'. Skipping DRL agent simulation.")