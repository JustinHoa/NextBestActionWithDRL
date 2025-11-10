import simpy
import random
import json
import os
import pandas as pd

# -----------------------------
# Load activity info
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, '../../data/raw_data/')
output_dir = os.path.join(current_dir, '../../data/evaluate_data/')
model_dir = os.path.join(current_dir, '../../models/')

with open(os.path.join(input_dir, "general_check_up_activities_info.json"), 'r', encoding='utf-8') as f:
    activity_info = json.load(f)

# -----------------------------
# Define clusters & fixed sequences
# -----------------------------
FIXED_SEQ_START = ["Registration", "Payment", "Get Triage Number", "Measure Vital Signs", "General Medicine Examination"]
FIXED_SEQ_END = ["Conclusion"]
CLUSTER_1_CANDIDATES = ["Eye Examination", "ENT Examination", "Dental Examination", "Gynecological Examination", "Breast Examination"]
CLUSTER_2_CANDIDATES = ["Blood Test", "Urine Test", "In-depth Eye Examination", "ENT Endoscopy",
                        "Electrocardiogram (ECG)", "Post-bronchodilator Spirometry", "General Ultrasound",
                        "Cardiac Ultrasound", "Chest X-ray", "DEXA Bone Density Scan"]

# -----------------------------
# HealthCheckCenter
# -----------------------------
class HealthCheckCenter:
    def __init__(self, env):
        self.env = env
        # Resource: tạo riêng cho mỗi activity
        self.resources = {}
        for name, data in activity_info.items():
            # GME + Conclusion dùng chung resource
            if name in ["General Medicine Examination", "Conclusion"]:
                if "GME_Conclusion" not in self.resources:
                    self.resources["GME_Conclusion"] = simpy.Resource(env, data["staff"])
            else:
                self.resources[name] = simpy.Resource(env, data["staff"])
        
        # Queue count: số bệnh nhân đang request + processing
        self.current_patient_count = {}
        for name in activity_info.keys():
            if name in ["General Medicine Examination", "Conclusion"]:
                self.current_patient_count["GME_Conclusion"] = 0
            else:
                self.current_patient_count[name] = 0

    def perform_activity(self, activity_name: str):
        data = activity_info[activity_name]
        mean_time = data["mean_time"] if not isinstance(data["mean_time"], dict) else random.choices(
            [data["mean_time"]["Cash"], data["mean_time"]["Credit"]],
            weights=[0.8,0.2]
        )[0]
        duration = random.triangular(mean_time*0.8, mean_time*1.2, mean_time)
        yield self.env.timeout(duration)

    def get_queue_status(self):
        return self.current_patient_count.copy()

# -----------------------------
# Patient
# -----------------------------
class Patient:
    def __init__(self, env, center: HealthCheckCenter, pid: int, gender: str, marital_status: str):
        self.env = env
        self.center = center
        self.id = pid
        self.gender = gender
        self.marital_status = marital_status
        self.prefix = [0] * len(activity_info)  # nhị phân theo activity_info id
        # Lab test init -1.0
        self.lab_results = {"blood": -1.0, "urine": -1.0}
        self.event_log = []

    # def do_activity(self, activity_name: str):
    #     # Map resource: GME + Conclusion dùng chung
    #     res_name = "GME_Conclusion" if activity_name in ["General Medicine Examination","Conclusion"] else activity_name
    #     resource = self.center.resources[res_name]

    #     # --- Request time ---
    #     request_time = self.env.now  # thời điểm patient gửi request
    #     with resource.request() as req:
    #         yield req
    #         # Increment queue count
    #         self.center.current_patient_count[res_name] += 1

    #         # --- Start time ---
    #         start_time = self.env.now
    #         yield self.env.process(self.center.perform_activity(activity_name))
    #         # --- End time ---
    #         end_time = self.env.now

    #         # Update lab results
    #         if activity_name == "Blood Test":
    #             mean_test_time = activity_info["Blood Test"]["mean_test_time"]
    #             epsilon = random.uniform(0,5)
    #             self.lab_results["blood"] = end_time + mean_test_time + epsilon
    #         elif activity_name == "Urine Test":
    #             mean_test_time = activity_info["Urine Test"]["mean_test_time"]
    #             epsilon = random.uniform(0,5)
    #             self.lab_results["urine"] = end_time + mean_test_time + epsilon

    #         # --- Append event log trực tiếp ---
    #         self.event_log.append({
    #             "patient_id": self.id,
    #             "activity_name": activity_name,
    #             "request_timestamp": request_time,
    #             "start_timestamp": start_time,
    #             "end_timestamp": end_time,
    #             "gender": self.gender if activity_name=="Registration" else "",
    #             "marital_status": self.marital_status if activity_name=="Registration" else "",
    #             "blood_result_ready": self.lab_results["blood"] if activity_name=="Blood Test" else "",
    #             "urine_result_ready": self.lab_results["urine"] if activity_name=="Urine Test" else ""
    #         })

    #         # Update prefix
    #         idx = activity_info[activity_name]["id"] - 1
    #         self.prefix[idx] = 1

    #         # Decrement queue count
    #         self.center.current_patient_count[res_name] -= 1

    def do_activity(self, activity_name: str):
        # Map resource: GME + Conclusion dùng chung
        res_name = "GME_Conclusion" if activity_name in ["General Medicine Examination","Conclusion"] else activity_name
        resource = self.center.resources[res_name]

        # --- Request time ---
        request_time = self.env.now  # thời điểm patient gửi request
        with resource.request() as req:
            yield req
            # Increment queue count
            self.center.current_patient_count[res_name] += 1

            # --- Snapshot queue status trước khi bắt đầu activity ---
            queue_snapshot = self.center.get_queue_status()

            # --- Start time ---
            start_time = self.env.now
            yield self.env.process(self.center.perform_activity(activity_name))
            # --- End time ---
            end_time = self.env.now

            # Update lab results
            if activity_name == "Blood Test":
                mean_test_time = activity_info["Blood Test"]["mean_test_time"]
                epsilon = random.uniform(0,5)
                self.lab_results["blood"] = end_time + mean_test_time + epsilon
            elif activity_name == "Urine Test":
                mean_test_time = activity_info["Urine Test"]["mean_test_time"]
                epsilon = random.uniform(0,5)
                self.lab_results["urine"] = end_time + mean_test_time + epsilon

            # --- Append event log trực tiếp ---
            self.event_log.append({
                "patient_id": self.id,
                "activity_name": activity_name,
                "request_timestamp": request_time,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "gender": self.gender if activity_name=="Registration" else "",
                "marital_status": self.marital_status if activity_name=="Registration" else "",
                "blood_result_ready": self.lab_results["blood"] if activity_name=="Blood Test" else "",
                "urine_result_ready": self.lab_results["urine"] if activity_name=="Urine Test" else "",
                "queue_status": json.dumps(queue_snapshot)  # 💥 thêm dòng này
            })

            # Update prefix
            idx = activity_info[activity_name]["id"] - 1
            self.prefix[idx] = 1

            # Decrement queue count
            self.center.current_patient_count[res_name] -= 1

    def go_through_process(self, coordinator):
        arrival_time = self.env.now

        # Step 1: prepare clusters
        cluster1 = []
        for act in CLUSTER_1_CANDIDATES:
            if act=="Gynecological Examination" and self.gender=="Female" and self.marital_status=="Married":
                cluster1.append(act)
            elif act=="Breast Examination" and self.gender=="Female":
                cluster1.append(act)
            elif act not in ["Gynecological Examination","Breast Examination"]:
                cluster1.append(act)
        cluster2 = list(CLUSTER_2_CANDIDATES)

        # Step 2: fixed start sequence
        for act in FIXED_SEQ_START:
            yield self.env.process(self.do_activity(act))

        # Step 3: Cluster 1 (greedy)
        while cluster1:
            queue_status = self.center.get_queue_status()
            next_act = coordinator.predict(self, queue_status, cluster1)
            cluster1.remove(next_act)
            yield self.env.process(self.do_activity(next_act))

        # Step 4: Cluster 2 (greedy)
        while cluster2:
            queue_status = self.center.get_queue_status()
            next_act = coordinator.predict(self, queue_status, cluster2)
            cluster2.remove(next_act)
            yield self.env.process(self.do_activity(next_act))

        # Step 5: Conclusion (wait lab nếu cần)
        test_ready_times = [t for t in [self.lab_results["blood"], self.lab_results["urine"]] if t>0]
        if test_ready_times:
            latest_ready = max(test_ready_times)
            yield self.env.timeout(max(0, latest_ready - self.env.now))
        for act in FIXED_SEQ_END:
            yield self.env.process(self.do_activity(act))

        return self.env.now - arrival_time

# -----------------------------
# NextBestAction (greedy-epsilon, dqn, dbcq)
# -----------------------------
class NextBestAction:
    def __init__(self, mode="greedy-epsilon", epsilon=0.3, model=None):
        self.mode = mode
        self.epsilon = epsilon
        self.model = model

    def predict(self, patient: Patient, queue_status: dict, candidates: list) -> str:
        # Greedy-epsilon
        if self.mode == "greedy-epsilon":
            # 1 phần bệnh nhân vẫn đi đúng theo thứ tự: 1->21. (30% bệnh nhân)
            if random.random() < self.epsilon:
                return candidates[0]
            
            # Còn lại chọn theo greedy.
            min_count = float('inf')
            selected = None
            for act in candidates:
                key = "GME_Conclusion" if act in ["General Medicine Examination","Conclusion"] else act
                if queue_status.get(key,0) < min_count:
                    min_count = queue_status.get(key,0)
                    selected = act
            return selected
        # DQN
        elif self.mode == "dqn":
            return
        
        # Discrete BCQ
        else: 
            return

# -----------------------------
# Patient generator (giữ nguyên logic arrival time)
# -----------------------------
def patient_generator(env, center: HealthCheckCenter, num_patients: int, coordinator: NextBestAction, all_patients: list):
    """
    Sinh bệnh nhân theo thời gian đến ngẫu nhiên (phân phối mũ)
    và cho họ tham gia mô phỏng.
    all_patients: danh sách để lưu reference tới các đối tượng Patient
    """
    for pid in range(num_patients):
        gender = random.choice(["Male", "Female"])
        marital_status = random.choice(["Single", "Married"])
        patient = Patient(env, center, pid, gender, marital_status)
        all_patients.append(patient)

        # Mỗi bệnh nhân được đưa vào mô phỏng dưới dạng process
        env.process(patient.go_through_process(coordinator))

        # Khoảng thời gian giữa hai bệnh nhân đến (mean = 5 phút)
        yield env.timeout(random.expovariate(1.0 / 5))

# -----------------------------
# Run simulation (chuẩn SimPy)
# -----------------------------
def run_simulation(num_patients=50):
    env = simpy.Environment()
    center = HealthCheckCenter(env)
    coordinator = NextBestAction("greedy", 0.3)
    all_patients = []

    # Gọi patient_generator để sinh bệnh nhân rải rác theo thời gian
    env.process(patient_generator(env, center, num_patients, coordinator, all_patients))

    # Chạy mô phỏng
    env.run(until=10000) # Chạy mô phỏng 10 000 phút.
    print("Simulation finished")

    # Gom event logs từ tất cả bệnh nhân
    df_list = [pd.DataFrame(p.event_log) for p in all_patients if p.event_log]
    df_all = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    # Xuất CSV
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "greedy_event_log.csv")
    df_all.to_csv(out_file, index=False)
    print(f"Event log saved to {out_file}")

    # Tính các chỉ số hiệu năng cơ bản
    total_times = []
    for p in all_patients:
        if p.event_log:
            arrival = p.event_log[0]["start_timestamp"]
            finish = p.event_log[-1]["end_timestamp"]
            total_times.append(finish - arrival)
    metrics = {}
    if total_times:
        metrics = {
            "avg_time": sum(total_times) / len(total_times),
            "max_time": max(total_times),
            "throughput": len(total_times) / max(total_times)
        }

    print("Metrics:", metrics)
    return df_all, metrics

if __name__ == "__main__":
    df, metrics = run_simulation(20)