import simpy
import random
import statistics
import pandas as pd

##### Setup Environment #####

# --- Activity Configuration ---
ACTIVITIES = {
    # --- A Area: Greeting ---
    "Registration": {"id": 1, "mean_time": 5, "staff": 2, "location": "A1"},
    "Payment": {"id": 2, "mean_time": {"Cash": 3, "Credit": 1}, "staff": 1, "location": "A2"},
    "Get Triage Number": {"id": 3, "mean_time": 1, "staff": 1, "location": "A3"},

    # --- B Area: Clinical ---
    "Measure Vital Signs": {"id": 4, "mean_time": 5, "staff": 1, "location": "B1"},
    "General Medicine Examination": {"id": 5, "mean_time": 20, "staff": 2, "location": "B2"},
    "Conclusion": {"id": 21, "mean_time": 20, "staff": 1, "location": "B2"}, 
    
    # CLUSTER 1 (Khám chuyên khoa: thứ tự tùy ý, cần hoàn thành xong Cụm 1 mới sang Cụm 2)
    "Eye Examination": {"id": 6, "mean_time": 15, "staff": 1, "location": "B3"},
    "ENT Examination": {"id": 7, "mean_time": 15, "staff": 1, "location": "B4"},
    "Dental Examination": {"id": 8, "mean_time": 15, "staff": 1, "location": "B5"},
    "Gynecological Examination": {"id": 9, "mean_time": 15, "staff": 1, "location": "B6"}, # Female Married only
    "Breast Examination": {"id": 10, "mean_time": 15, "staff": 1, "location": "B7"}, # Female only

    # CLUSTER 2 (Cận lâm sàng: thứ tự tùy ý, cần hoàn thành xong Cụm 2 mới sang Conclusion)
    "Blood Test": {"id": 11, "mean_time": 10, "mean_test_time": 90, "staff": 1, "location": "C10"},
    "Urine Test": {"id": 12, "mean_time": 10, "mean_test_time": 60, "staff": 1, "location": "C5"},
    "In-depth Eye Examination": {"id": 13, "mean_time": 30, "staff": 1, "location": "C3"},
    "ENT Endoscopy": {"id": 14, "mean_time": 20, "staff": 1, "location": "C6"},
    "Electrocardiogram (ECG)": {"id": 15, "mean_time": 20, "staff": 1, "location": "C7"},
    "Post-bronchodilator Spirometry": {"id": 16, "mean_time": 30, "staff": 1, "location": "C8"},
    "General Ultrasound": {"id": 17, "mean_time": 45, "staff": 1, "location": "C4"},
    "Cardiac Ultrasound": {"id": 18, "mean_time": 30, "staff": 1, "location": "C9"},
    "Chest X-ray": {"id": 19, "mean_time": 20, "staff": 1, "location": "C2"},
    "DEXA Bone Density Scan": {"id": 20, "mean_time": 20, "staff": 1, "location": "C1"},
}

# Định nghĩa các cụm hoạt động
FIXED_SEQ_START = [
    "Registration", "Payment", "Get Triage Number", 
    "Measure Vital Signs", "General Medicine Examination"
]

CLUSTER_1_CANDIDATES = [
    "Eye Examination", "ENT Examination", "Dental Examination", 
    "Gynecological Examination", "Breast Examination"
]

CLUSTER_2_CANDIDATES = [
    "Blood Test", "Urine Test", "In-depth Eye Examination", "ENT Endoscopy",
    "Electrocardiogram (ECG)", "Post-bronchodilator Spirometry", "General Ultrasound",
    "Cardiac Ultrasound", "Chest X-ray", "DEXA Bone Density Scan"
]

FIXED_SEQ_END = ["Conclusion"]


# Global event log and performance metrics
event_log = []
wait_times = []

# --- Health Check Center ---
class HealthCheckCenter:
    def __init__(self, env):
        self.env = env
        self.resources = {
            name: simpy.Resource(env, data["staff"]) for name, data in ACTIVITIES.items()
        }

    def perform_activity(self, name):
        """Simulate the processing time for a given activity."""
        data = ACTIVITIES[name]

        # Choose payment type randomly
        if isinstance(data["mean_time"], dict):
            mean_time = random.choices(
                [data["mean_time"]["Cash"], data["mean_time"]["Credit"]],
                weights=[0.8, 0.2],
            )[0]
        else:
            mean_time = data["mean_time"]

        # Simulate execution time (some randomness)
        # Dùng phân phối tam giác để giữ thời gian thực hiện ở mức hợp lý
        yield self.env.timeout(random.triangular(mean_time * 0.8, mean_time * 1.2, mean_time))

# --- Patient ---
class Patient:
    def __init__(self, env, center, pid, gender, marital_status):
        self.env = env
        self.center = center
        self.id = pid
        self.gender = gender
        self.marital_status = marital_status

        # Track when lab results become available
        self.blood_result_ready = None
        self.urine_result_ready = None

    def _log_event(self, act, start_time, end_time):
        """Helper function to log a single event."""
        event = {
            "patient_id": self.id,
            "activity_name": act,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "gender": self.gender if act == "Registration" else "",
            "marital_status": self.marital_status if act == "Registration" else "",
            "result_ready_time": (
                self.blood_result_ready if act == "Blood Test"
                else self.urine_result_ready if act == "Urine Test"
                else ""
            ),
        }
        event_log.append(event)
        
    def _do_activity(self, act):
        """Process a single activity and log the event."""
        resource = self.center.resources[act]
        
        with resource.request() as req:
            yield req
            start_time = self.env.now
            yield self.env.process(self.center.perform_activity(act))
            end_time = self.env.now
            
            # Cập nhật thời gian có kết quả xét nghiệm
            if act == "Blood Test":
                self.blood_result_ready = end_time + ACTIVITIES["Blood Test"]["mean_test_time"]
            elif act == "Urine Test":
                self.urine_result_ready = end_time + ACTIVITIES["Urine Test"]["mean_test_time"]
                
            self._log_event(act, start_time, end_time)

    def go_through_process(self):
        """Simulate a patient's full check-up process."""
        arrival_time = self.env.now

        # 1. --- Xây dựng luồng hoạt động cụ thể cho bệnh nhân ---
        
        # Lọc các hoạt động Cụm 1 theo giới tính/tình trạng hôn nhân
        patient_cluster_1 = []
        for act in CLUSTER_1_CANDIDATES:
            if act == "Gynecological Examination" and self.gender == "Female" and self.marital_status == "Married":
                patient_cluster_1.append(act)
            elif act == "Breast Examination" and self.gender == "Female":
                patient_cluster_1.append(act)
            elif act not in ["Gynecological Examination", "Breast Examination"]:
                patient_cluster_1.append(act)
        
        # Cbụm 2: Không có giới hạn nhân khẩu học, lấy toàn ộ
        patient_cluster_2 = CLUSTER_2_CANDIDATES
        
        # Xáo trộn thứ tự thực hiện ngẫu nhiên trong mỗi cụm song song
        random.shuffle(patient_cluster_1)
        random.shuffle(patient_cluster_2)
        
        # 2. --- Thực hiện các Hoạt động Tuần tự Bắt buộc (Phần đầu) ---
        for act in FIXED_SEQ_START:
            yield self.env.process(self._do_activity(act))
            
        # 3. --- Thực hiện Cụm 1 (Song song) ---
        # Bệnh nhân thực hiện các hoạt động trong Cụm 1 (thứ tự tùy ý)
        
        # Tạo list các process cho Cụm 1
        cluster_1_processes = [self.env.process(self._do_activity(act)) for act in patient_cluster_1]
        
        # Chờ TẤT CẢ các process trong Cụm 1 hoàn thành
        if cluster_1_processes:
            yield self.env.all_of(cluster_1_processes)
            
        # 4. --- Thực hiện Cụm 2 (Song song) ---
        # Bệnh nhân thực hiện các hoạt động trong Cụm 2 (thứ tự tùy ý)
        
        # Tạo list các process cho Cụm 2
        cluster_2_processes = [self.env.process(self._do_activity(act)) for act in patient_cluster_2]
        
        # Chờ TẤT CẢ các process trong Cụm 2 hoàn thành
        if cluster_2_processes:
            yield self.env.all_of(cluster_2_processes)

        # 5. --- Thực hiện Hoạt động Tuần tự Bắt buộc (Conclusion) ---
        for act in FIXED_SEQ_END:
            # Riêng bước Conclusion, cần kiểm tra thời gian có kết quả xét nghiệm
            test_ready_times = [
                t for t in [self.blood_result_ready, self.urine_result_ready] if t is not None
            ]
            if test_ready_times:
                latest_ready = max(test_ready_times)
                # Chờ cho đến khi kết quả xét nghiệm (Blood Test/Urine Test) sẵn sàng
                yield self.env.timeout(max(0, latest_ready - self.env.now))
                
            yield self.env.process(self._do_activity(act))
            
        # 6. Ghi nhận tổng thời gian hoàn thành
        total_time = self.env.now - arrival_time
        wait_times.append(total_time)


# --- Patient Generator ---
def patient_generator(env, center, num_patients, male_ratio, married_female_ratio):
    """Continuously generate a fixed number of patients with random demographics."""
    
    # Tính toán số lượng bệnh nhân theo tỉ lệ
    N_MALE = int(num_patients * male_ratio)
    N_FEMALE_MARRIED = int(num_patients * married_female_ratio)
    N_FEMALE_SINGLE = num_patients - N_MALE - N_FEMALE_MARRIED

    # Tạo danh sách nhân khẩu học
    demographics = (
        [("Male", None)] * N_MALE +
        [("Female", "Married")] * N_FEMALE_MARRIED +
        [("Female", "Single")] * N_FEMALE_SINGLE
    )
    random.shuffle(demographics)
    
    # Tỉ lệ đến ngẫu nhiên (sử dụng arrival_rate=5 từ hàm main)
    arrival_rate = 5 # minutes
    
    for pid in range(num_patients):
        gender, marital_status = demographics[pid]
        
        patient = Patient(env, center, pid, gender, marital_status)
        env.process(patient.go_through_process())
        
        # Khoảng thời gian đến giữa các bệnh nhân (phân phối Poisson)
        yield env.timeout(random.expovariate(1.0 / arrival_rate))

# --- Run Simulation ---
def main():
    # Sử dụng seed cố định cho khả năng tái tạo
    random.seed(42) 
    
    # Giả lập 100 bệnh nhân, tỉ lệ 50:20:30 như bạn yêu cầu (Nam: Nữ Married: Nữ Single)
    NUM_PATIENTS_SIM = 100
    MALE_RATIO = 0.5
    MARRIED_FEMALE_RATIO = 0.2
    
    env = simpy.Environment()
    center = HealthCheckCenter(env)

    # Khởi tạo quá trình tạo bệnh nhân
    env.process(patient_generator(
        env, center, 
        num_patients=NUM_PATIENTS_SIM, 
        male_ratio=MALE_RATIO, 
        married_female_ratio=MARRIED_FEMALE_RATIO
    ))
    
    # Chạy mô phỏng cho đến khi tất cả 100 bệnh nhân hoàn thành
    # (Tăng thời gian chạy lên để đảm bảo 100 bệnh nhân hoàn thành)
    env.run(until=5000) 

    # --- Kết quả và Xuất file ---
    
    # 1. Hiển thị hiệu suất
    actual_completed_patients = len(wait_times)
    if actual_completed_patients > 0:
        avg_wait = statistics.mean(wait_times)
        print(f"Tổng bệnh nhân hoàn thành: {actual_completed_patients}/{NUM_PATIENTS_SIM}")
        print(f"Thời gian xử lý trung bình: {avg_wait:.2f} phút.")
    else:
        print("Không có bệnh nhân nào hoàn thành trong thời gian mô phỏng.")

    # 2. Xuất event log
    df = pd.DataFrame(event_log)
    
    # Điền thông tin nhân khẩu học (gender, marital_status) cho tất cả các dòng
    demography_map = df[df['activity_name'] == 'Registration'][['patient_id', 'gender', 'marital_status']]
    df = df.drop(columns=['gender', 'marital_status'])
    df = pd.merge(df, demography_map, on='patient_id', how='left')

    # Định dạng lại thời gian
    df['start_timestamp'] = df['start_timestamp'].round(2)
    df['end_timestamp'] = df['end_timestamp'].round(2)
    
    # Sắp xếp theo thời gian bắt đầu
    df = df.sort_values(by="start_timestamp").reset_index(drop=True)
    
    # Lưu file
    df.to_csv("event_log_simpy.csv", index=False)
    print("Event log saved as 'event_log_simpy.csv'.")
    print("\n--- Event Log Head ---")
    print(df.head(10))

if __name__ == "__main__":
    main()


def go_through_process(self):
    """Simulate a patient's full check-up process (randomized cluster order, sequential execution)."""
    arrival_time = self.env.now

    # 1. --- Xây dựng luồng hoạt động cụ thể cho bệnh nhân ---

    # Cụm 1: lọc theo giới tính / tình trạng hôn nhân
    patient_cluster_1 = []
    for act in CLUSTER_1_CANDIDATES:
        if act == "Gynecological Examination" and self.gender == "Female" and self.marital_status == "Married":
            patient_cluster_1.append(act)
        elif act == "Breast Examination" and self.gender == "Female":
            patient_cluster_1.append(act)
        elif act not in ["Gynecological Examination", "Breast Examination"]:
            patient_cluster_1.append(act)

    # Cụm 2: không giới hạn nhân khẩu học
    patient_cluster_2 = list(CLUSTER_2_CANDIDATES)

    # Xáo trộn thứ tự trong mỗi cụm
    random.shuffle(patient_cluster_1)
    random.shuffle(patient_cluster_2)

    # 2. --- Ghép toàn bộ luồng hoạt động theo thứ tự ---
    activities = FIXED_SEQ_START + patient_cluster_1 + patient_cluster_2 + FIXED_SEQ_END

    # 3. --- Thực hiện tuần tự từng hoạt động ---
    for act in activities:
        resource = self.center.resources[act]

        # Nếu là bước Conclusion thì phải chờ kết quả xét nghiệm sẵn sàng
        if act == "Conclusion":
            test_ready_times = [
                t for t in [self.blood_result_ready, self.urine_result_ready] if t is not None
            ]
            if test_ready_times:
                latest_ready = max(test_ready_times)
                yield self.env.timeout(max(0, latest_ready - self.env.now))

        # Request tài nguyên và thực hiện
        with resource.request() as req:
            yield req
            start_time = self.env.now
            yield self.env.process(self.center.perform_activity(act))
            end_time = self.env.now

        # Cập nhật thời gian có kết quả xét nghiệm
        if act == "Blood Test":
            self.blood_result_ready = end_time + ACTIVITIES["Blood Test"]["mean_test_time"]
        elif act == "Urine Test":
            self.urine_result_ready = end_time + ACTIVITIES["Urine Test"]["mean_test_time"]

        # --- Ghi log event ---
        event = {
            "patient_id": self.id,
            "activity_name": act,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "gender": self.gender if act == "Registration" else "",
            "marital_status": self.marital_status if act == "Registration" else "",
            "result_ready_time": (
                self.blood_result_ready if act == "Blood Test"
                else self.urine_result_ready if act == "Urine Test"
                else ""
            ),
        }
        event_log.append(event)

    # 4. --- Tổng thời gian hoàn thành ---
    total_time = self.env.now - arrival_time
    wait_times.append(total_time)

