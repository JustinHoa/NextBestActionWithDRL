# Dự Án Điều Phối Bệnh Nhân Dùng Deep Reinforcement Learning

## 1. Giới thiệu (Overview)

Dự án này xây dựng một hệ thống AI sử dụng Deep Reinforcement Learning (DRL) để điều phối luồng bệnh nhân trong một trung tâm khám sức khỏe tổng quát. Mục tiêu là giảm thiểu thời gian chờ đợi của bệnh nhân và tối ưu hóa việc sử dụng tài nguyên (y bác sĩ, phòng khám).

Điểm đặc biệt của dự án là quy trình Training Loop khép kín:

1.  **Simulation**: Mô phỏng môi trường bệnh viện để sinh dữ liệu hàng chờ (queue log). Dữ liệu ban đầu được tạo ra bằng cách cho bệnh nhân chọn phòng một cách ngẫu nhiên.
2.  **Training**: Huấn luyện một Agent (bộ não AI) dựa trên dữ liệu hàng chờ đã sinh ra. Agent sẽ học cách chọn phòng khám tiếp theo sao cho thời gian chờ dự kiến là thấp nhất.
3.  **Evaluation & Refinement**: Áp dụng Agent đã huấn luyện vào lại môi trường mô phỏng. Agent lúc này sẽ đưa ra quyết định "thông minh" hơn, tạo ra một bộ dữ liệu hàng chờ mới hiệu quả hơn. Quá trình này lặp lại, giúp Agent ngày càng được tinh chỉnh (fine-tuning) qua các "thế hệ".

## 2. Các thuật toán hỗ trợ (Algorithms)

Dự án được thiết kế theo hướng Module hóa (Modular Design) để dễ dàng thử nghiệm và mở rộng nhiều biến thể của DQN. Tất cả các thuật toán đều được tích hợp cơ chế **Action Masking** để đảm bảo AI không bao giờ chọn các hành động không hợp lệ (ví dụ: nam giới khám phụ khoa, hoặc khám lại phòng đã khám).

- [x] **DQN** (Deep Q-Network): Thuật toán DRL nền tảng.
- [x] **Double DQN (DDQN)**: Cải tiến của DQN, giúp giảm việc đánh giá quá cao giá trị Q (Overestimation bias), làm cho việc học ổn định hơn.
- [x] **Dueling DQN**: Sử dụng kiến trúc mạng đặc biệt, tách luồng ước tính giá trị của trạng thái (State Value) và lợi thế của hành động (Advantage), giúp học hiệu quả hơn trong các môi trường có nhiều hành động.
- [x] **Multi-step Learning**: Cho phép agent nhìn xa hơn một bước trong tương lai khi cập nhật giá trị Q, giúp tăng tốc độ học.
- [x] **Prioritized Experience Replay (PER)**: Thay vì lấy mẫu kinh nghiệm một cách ngẫu nhiên, PER ưu tiên những kinh nghiệm "gây bất ngờ" hoặc có lỗi dự đoán cao, giúp agent tập trung học vào những điều khó.
- [x] **Rainbow DQN**: Một thuật toán tổng hợp, kết hợp tất cả các cải tiến trên để đạt được hiệu suất vượt trội.

## 3. Cấu Trúc Thư Mục (Project Structure)

```
rl_project/
│
├── main.py              # 🚀 FILE CHẠY CHÍNH: Điều khiển luồng Training
├── requirements.txt     # Các thư viện cần thiết
├── README.md            # Tài liệu hướng dẫn dự án
│
├── agents/              # 🧠 TRÍ TUỆ NHÂN TẠO (AI LOGIC)
│   ├── __init__.py
│   ├── base_agent.py    # Class cha, chứa logic act() và Masking
│   ├── dqn_agent.py     # Cài đặt DQN
│   ├── ddqn_agent.py    # Cài đặt Double DQN
│   ├── dueling_agent.py # Cài đặt Dueling DQN
│   └── rainbow_agent.py # Cài đặt Rainbow (DDQN, Dueling, PER, Multi-step)
│
├── common/              # 🛠️ CÔNG CỤ DÙNG CHUNG (UTILITIES)
│   ├── __init__.py
│   ├── env.py           # Môi trường RL (Gym-like) đọc Queue Log
│   ├── buffers.py       # Replay Buffer và Prioritized Replay Buffer
│   └── utils.py         # Hàm vẽ đồ thị, config, save/load
│
├── networks/            # 🕸️ KIẾN TRÚC MẠNG NEURAL (PYTORCH)
│   ├── __init__.py
│   ├── dqn_net.py       # Mạng Fully Connected cơ bản
│   └── duel_net.py      # Mạng Dueling (Value & Advantage streams)
│
├── simulation/          # 🌍 MÔ PHỎNG QUY TRÌNH (SIMPY)
│   ├── __init__.py
│   └── simulation_process.py # Chạy SimPy: Sinh Event Log & Queue Log
│
├── data/                # 💾 DỮ LIỆU
│   ├── raw/             # Input cho Training (Queue Logs, Activity Info)
│   │   ├── activity_info.json
│   │   └── ...
│   └── evaluate/        # Output của Simulation (Event Logs)
│       └── ...
│
└── logs/                # 📊 KẾT QUẢ TRAINING (CHECKPOINTS & PLOTS)
    ├── DQN/
    ├── DDQN/
    └── ...
```

## 4. Quy trình vận hành (Pipeline)

Hệ thống hoạt động theo mô hình 3 thế hệ (Generations) để tinh chỉnh dần dần:

1.  **Generation 1 (Học từ đầu)**:

    - **Data**: Dùng dữ liệu `queue_log_version_0.csv` được tạo ra từ mô phỏng chạy hoàn toàn ngẫu nhiên.
    - **Training**: Agent học cách "né" những phòng khám quá đông và tuân thủ các quy tắc cơ bản.
    - **Output**: Model `final_gen1.pth`.

2.  **Generation 2 (Tinh chỉnh)**:

    - **Simulation**: Chạy mô phỏng với Model Gen 1 để tạo ra bộ dữ liệu mới `queue_log_version_1.csv`. Môi trường lúc này đã trật tự hơn.
    - **Training**: Agent được huấn luyện tiếp (fine-tune) trên dữ liệu mới này.
    - **Output**: Model `final_gen2.pth`.

3.  **Generation 3 (Tối ưu hóa)**:
    - **Simulation & Training**: Lặp lại quy trình với Learning Rate thấp hơn để tối ưu hóa sâu hơn nữa.
    - **Output**: Model `final_gen3.pth`.

## 5. Cách chạy dự án

### Bước 1: Cài đặt thư viện

Đảm bảo bạn đã cài đặt Python 3.8+ và các thư viện cần thiết.

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Training

Mở file `main.py`, tìm và sửa biến `ALGO_TO_RUN` thành thuật toán bạn muốn huấn luyện (ví dụ: `'DQN'`, `'DDQN'`, `'Dueling'`, `'Rainbow'`).

Sau đó, chạy lệnh sau từ terminal:

```bash
python main.py
```

Hệ thống sẽ tự động thực hiện chuỗi training 3 thế hệ. Các model đã huấn luyện (`.pth`) và biểu đồ kết quả (`.png`) sẽ được lưu vào thư mục `logs/<Tên thuật toán>/`.

### Bước 3: Chạy Simulation & Đánh giá

Sau khi đã có model, bạn có thể dùng nó để chạy mô phỏng và xem hiệu quả thực tế.

1.  Mở file `simulation/simulation_process.py`.
2.  Chỉnh sửa biến `MODEL_PATH` để trỏ đến file model bạn muốn đánh giá (ví dụ: `"logs/DDQN/final_gen3.pth"`).
3.  Chạy file:
    ```bash
    python simulation/simulation_process.py
    ```

File này sẽ chạy 2 kịch bản: một là mô phỏng với agent ngẫu nhiên (để làm baseline), hai là mô phỏng với agent DRL của bạn. Kết quả so sánh hiệu suất (thời gian khám trung bình) sẽ được in ra và một biểu đồ so sánh sẽ được lưu trong `data/evaluate/`.

Đồng thời, một file `event_log_version_XXX.csv` sẽ được tạo ra, bạn có thể sử dụng file này với các công cụ Process Mining (như Celonis, PM4Py) để phân tích quy trình chi tiết.
