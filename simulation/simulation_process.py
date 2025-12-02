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
from agents.per_dqn_agent import PerDqnAgent

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

    def predict(self, patient, current_queues_arr, current_time, candidates_names: list):
        candidate_indices = [name_to_id[name] - 1 for name in candidates_names]
        
        # Nếu không có agent, chọn ngẫu nhiên
        if self.agent is None:
            return random.choice(candidates_names) if candidates_names else None

        # Agent Logic
        raw_wait = (current_queues_arr * mean_times_arr) / staff_arr
        norm_wait = raw_wait / 200.0 
        norm_time = np.array([current_time / self.total_sim_time])
        
        # Encode Patient Info
        gender_code = 1 if patient.gender == "Male" else 0
        marital_code = 1 if patient.marital == "Married" else 0
        
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
        self.start_time = 0
        self.end_time = 0

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
        self.start_time = self.env.now
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
        self.end_time = self.env.now
        self.center.finished_patient_count += 1

def run_simulation(num_patients=200, agent=None, version_output="0", is_model_run=False, seed=None):
    env = simpy.Environment() # type: ignore
    # Cố định seed cho các thư viện ngẫu nhiên để đảm bảo tính lặp lại
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    center = HealthCheckCenter(env)
    coord = Coordinator(agent, total_sim_time=1440)
    
    event_logs = []
    queue_logs = []
    patients_list = [] # Khởi tạo danh sách bệnh nhân ở đây

    # Generator
    def patient_gen(patient_list_ref):
        for i in range(num_patients):
            p = Patient(env, center, i, random.choice(["Male", "Female"]), random.choice(["Single", "Married"]), event_logs)
            patient_list_ref.append(p) # Thêm bệnh nhân vào danh sách được truyền vào
            env.process(p.go_process(coord))
            yield env.timeout(random.expovariate(1.0/2.0)) # Arrive every ~2 mins

    # Monitor
    def monitor():
        while center.finished_patient_count < num_patients:
            status = center.current_patient_count.copy()
            status["Time"] = env.now
            queue_logs.append(status)
            yield env.timeout(1.0)

    env.process(patient_gen(patients_list)) # Truyền danh sách vào generator
    env.process(monitor())
    env.run()

    # Lưu queue log nếu là lần chạy để sinh data cho thế hệ tiếp theo hoặc là lần chạy random đầu tiên
    if is_model_run or agent is None:
        df_q = pd.DataFrame(queue_logs)
        # Đảm bảo các cột được sắp xếp đúng thứ tự
        cols = ['Time'] + [name for name in activity_names_list if name in df_q.columns]
        df_q = df_q[cols]
        q_path = os.path.join(RAW_DATA_DIR, f"200_queue_log_version_{version_output}.csv")
        df_q.to_csv(q_path, index=False)
        print(f"✅ Saved QueueLog: {q_path}")

    # Luôn lưu event log để đánh giá
    if len(event_logs) > 0:
        df_e = pd.DataFrame(event_logs)
        e_path = os.path.join(EVAL_DATA_DIR, f"event_log_version_{version_output}.csv")
        df_e.to_csv(e_path, index=False)
        print(f"✅ Saved EventLog for evaluation: {e_path}")
    
    print(f"✅ Simulation Version '{version_output}' finished.")
    
    # Trả về thời gian khám trung bình để so sánh
    # Tính toán thời gian từ danh sách bệnh nhân đã được thu thập
    total_times = [p.end_time - p.start_time for p in patients_list if hasattr(p, 'end_time') and p.end_time > 0]
    return np.mean(total_times) if total_times else 0

# --- HELPER FUNCTION: Tự động chọn class Agent ---
def get_agent(algo_name, model_path, state_size, action_size):
    algo_name = algo_name.lower()
    if 'rainbow' in algo_name:
        return RainbowAgent(state_size, action_size)
    elif 'duel' in algo_name:
        return DuelingAgent(state_size, action_size)
    elif 'per' in algo_name:
        return PerDqnAgent(state_size, action_size)
    elif 'ddqn' in algo_name:
        return DDQNAgent(state_size, action_size)
    else:
        # Mặc định là DQN
        return DQNAgent(state_size, action_size)

if __name__ == "__main__":
    # 1. CẤU HÌNH CHẠY TEST
    NUM_PATIENTS = 200        # Số bệnh nhân mỗi lần test
    TEST_SEED = 42            # <--- QUAN TRỌNG: Cố định seed để công bằng
    MODEL_FILENAME = "final_gen3.pth" # Tên file model bạn muốn test
    
    # Danh sách các thuật toán cần test (tên folder trong logs/)
    ALGO_LIST = ["DQN", "DDQN", "Dueling", "PerDQN", "Rainbow"] 
    
    print(f"--- STARTING EVALUATION (SEED={TEST_SEED}) ---")

    # 2. CHẠY RANDOM (BASELINE) - CHỈ CHẠY 1 LẦN VỚI SEED CỐ ĐỊNH
    print(f"\n[BASELINE] Running Random Agent with Seed {TEST_SEED}...")
    start_time = time.time()
    # Lưu ý: Truyền seed vào đây để đảm bảo môi trường giống hệt các lần sau
    random_avg_time = run_simulation(NUM_PATIENTS, agent=None, version_output="random_base", seed=TEST_SEED)
    print(f"-> Random Avg Time: {random_avg_time:.2f} mins")

    # 3. VÒNG LẶP TEST TỪNG MODEL
    results = []
    
    for algo in ALGO_LIST:
        model_path = os.path.join(ROOT_DIR, "logs", algo, MODEL_FILENAME)
        
        print(f"\n[{algo}] Testing model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"⚠️ File not found: {model_path}. Skipping...")
            continue
            
        # Load Agent
        try:
            agent = get_agent(algo, model_path, STATE_SIZE, ACTION_SIZE)
            agent.load(model_path)
            
            # Chạy Simulation với CÙNG MỘT SEED như Random
            t0 = time.time()
            model_avg_time = run_simulation(NUM_PATIENTS, agent=agent, version_output=algo.lower(), seed=TEST_SEED)
            duration = time.time() - t0
            
            # Tính toán độ cải thiện
            improvement = ((random_avg_time - model_avg_time) / random_avg_time) * 100
            
            print(f"-> {algo} Avg Time: {model_avg_time:.2f} mins | Improvement: {improvement:+.2f}%")
            
            results.append({
                "Algorithm": algo,
                "Avg Time (min)": model_avg_time,
                "Improvement (%)": improvement,
                "Real Time (s)": duration
            })
            
        except Exception as e:
            print(f"❌ Error testing {algo}: {e}")

    # 4. TỔNG HỢP KẾT QUẢ
    print("\n" + "="*40)
    print(f"{'SUMMARY REPORT':^40}")
    print("="*40)
    print(f"Baseline (Random): {random_avg_time:.2f} mins")
    print("-" * 40)
    print(f"{'Algorithm':<10} | {'Time':<10} | {'Improve':<10}")
    print("-" * 40)
    
    # Sắp xếp kết quả từ tốt nhất đến tệ nhất
    results.sort(key=lambda x: x["Avg Time (min)"])
    
    for res in results:
        print(f"{res['Algorithm']:<10} | {res['Avg Time (min)']:<10.2f} | {res['Improvement (%)']:<+8.2f}%")
    print("="*40)