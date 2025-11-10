# import simpy
# import random
# import statistics
# import pandas as pd
# import os
# import json

# current_dir = os.path.dirname(os.path.abspath(__file__))
# input_dir = os.path.join(current_dir, '../../../data/raw_data/')
# model_dir = os.path.join(current_dir, '../../model/')

# with open(input_dir + "general_check_up_activities_info.json", 'r', encoding='utf-8') as f:
#     activity_info = json.load(f)

# # Các hoạt động tuần tự: id từ 1->5
# # Khám lâm sàng chuyên khoa: song song: 6->10
# # Khám cận lâm sàng: song song: 11->20
# # Kết luận: 21.

# # Chạy giả lập với simpy, ghi lại thông tin event log (patient_id, activity, request_timestamp, start_timestamp, end_timestamp, wait_time (= start_timestamp - request_timestamp))

# # Kiến trúc như sau:
# # Class bệnh viện: khai báo nhân lực, thời gian xử lí trung bình,... (thông tin trong activity_info). Ngoài ra với mỗi activity phải có biến đếm số bệnh nhân tại mỗi node, cập nhật liên tục theo thời gian thực. (đặc biệt quan trọng)
# # Class bệnh nhân: tạo bệnh nhân ngẫu nhiên gồm patient_id, gender, marital_status. 
# # Class điều phối (hay là next best action) đó: thì cứ với mỗi đầu vào là gender, marital status, prefix (chuỗi hoạt động, cái nào bật lên 1 thì bệnh nhân đã đi cái đó), số bệnh nhân tại mỗi node (lấy từ cái class bệnh viện đó), thời gian chờ blood result time, urine result time.
# # -> Dự đoán ra next best action. Thì đang có 3 cơ chế dự đoán. 
# # Cách 1: Greedy-epsilon: cái nào ít thì nhảy vô. (thực tế nè) tuy nhiên có 1 bộ phận thì vẫn đi đúng từ 1->21. không greedy. Greedy thì chỉ ở Khám lâm sàng chuyên khoa (5 cái) và Khám cận lâm sàng (10 cái). Oke he.
# # Cách 2: Dùng model dqn (đã train rồi nè) thì cứ đẩy đầu vào vào model -> nó dự đoán ra next best action rồi đi thôi. Cứ có 1 bệnh nhân xong 1 action. -> Request model -> Next best action -> Chọn action đó.
# # Cách 3: Cũng như cách 2 nhưng mà dùng model khác là Discrete Batch Constraint Q Learning.
# # 2 cái model thì đang để trong model_dir rồi. Có nên viết code của Greedy-epsilon và lưu vào model luôn không nhỉ? Rồi load ra thôi.

# # Sau khi chạy giả lập với 3 model -> 3 event log. Kiểm tra 3 event log này để xem cái nào oke hơn.

# # Nhận xét tính khả thi của của việc giả lập này.

import simpy
import random
import pandas as pd

# ============================================
# CLASS: HealthCheckCenter
# Mục tiêu: Quản lý tài nguyên của từng activity, track số bệnh nhân tại node
# ============================================
class HealthCheckCenter:
    def __init__(self, env):
        """
        env: simpy.Environment
        self.resources: dict[activity_name -> simpy.Resource]
        """
        self.env = env
        self.resources = {}  # tạo resource từ ACTIVITIES
        self.current_patient_count = {}  # dict[activity_name -> int]
    
    def perform_activity(self, activity_name: str) -> float:
        """
        Mô phỏng thời gian thực hiện 1 activity.
        Returns: duration (float)
        """
        pass  # implement timeout + randomization

    def get_queue_status(self) -> dict:
        """
        Trả về số bệnh nhân hiện tại ở mỗi node
        """
        return self.current_patient_count.copy()

# ============================================
# CLASS: Patient
# Mục tiêu: Thể hiện 1 bệnh nhân, lưu trữ thông tin cá nhân và lịch sử hoạt động
# ============================================
class Patient:
    def __init__(self, env, center: HealthCheckCenter, pid: int, gender: str, marital_status: str):
        """
        env: simpy.Environment
        center: HealthCheckCenter
        pid: patient_id
        gender, marital_status
        """
        self.env = env
        self.center = center
        self.id = pid
        self.gender = gender
        self.marital_status = marital_status
        self.prefix = []  # list[int] các activity đã thực hiện
        self.lab_results = {"blood": None, "urine": None}
        self.event_log = []

    def log_event(self, activity_name: str, start_time: float, end_time: float):
        """
        Ghi lại event log của 1 activity
        """
        pass

    def do_activity(self, activity_name: str):
        """
        Thực hiện 1 activity: request resource, simulate duration, update prefix và lab_results
        """
        pass

    def go_through_process(self, coordinator) -> float:
        """
        Input: coordinator (NextBestAction object)
        Chạy toàn bộ luồng khám:
            1. Hoạt động tuần tự
            2. Cluster 1 (song song)
            3. Cluster 2 (song song)
            4. Conclusion
        Output: total_time (float)
        """
        pass

# ============================================
# CLASS: NextBestAction
# Mục tiêu: Dự đoán next activity dựa trên 3 cơ chế
# ============================================
class NextBestAction:
    def __init__(self, mode: str, model_path: str = None):
        """
        mode: 'greedy', 'dqn', 'cql'
        model_path: nếu mode=='dqn' hoặc 'cql', đường dẫn model
        """
        self.mode = mode
        self.model_path = model_path
        self.model = None  # load model nếu cần

    def predict(self, patient: Patient, queue_status: dict) -> str:
        """
        Input: patient (Patient object), queue_status (dict[activity -> số bệnh nhân])
        Output: next_activity_name (str)
        """
        pass  # implement theo mode

# ============================================
# FUNCTION: patient_generator
# Mục tiêu: tạo và khởi chạy quá trình Simpy cho N bệnh nhân
# ============================================
def patient_generator(env, center: HealthCheckCenter, num_patients: int, coordinator: NextBestAction):
    """
    Input: env, center, num_patients, coordinator
    Output: tạo env.process cho từng patient
    """
    pass

# ============================================
# FUNCTION: run_simulation
# Mục tiêu: Chạy mô phỏng với 1 cơ chế NextBestAction
# ============================================
def run_simulation(num_patients: int, mode: str, model_path: str = None):
    """
    Input: num_patients, mode, model_path
    Output: event_log (list[dict]), metrics (dict)
    """
    env = simpy.Environment()
    center = HealthCheckCenter(env)
    coordinator = NextBestAction(mode, model_path)

    env.process(patient_generator(env, center, num_patients, coordinator))
    env.run(until=10000)  # hoặc until all patients done

    # Tính metrics: throughput, avg_wait_time, max_wait_time...
    metrics = {}
    return [], metrics

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    NUM_PATIENTS = 100
    
    # Ví dụ chạy 3 cơ chế
    for mode in ['greedy', 'dqn', 'cql']:
        event_log, metrics = run_simulation(NUM_PATIENTS, mode, model_path=f"model/{mode}.pt")
        print(f"Mode: {mode}, Metrics: {metrics}")
        # Lưu event_log ra CSV

