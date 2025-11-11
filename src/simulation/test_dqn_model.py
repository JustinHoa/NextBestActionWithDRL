# import os
# import numpy as np
# import torch
# from stable_baselines3 import DQN
# from src.online_training.next_best_action_env.env import NextBestActionEnv

# # ================================
# # Load model
# # ================================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, "../../models/dqn_model_best.zip")

# print(f"Loading model from {model_path}...")
# model = DQN.load(model_path)
# print("✅ Model loaded successfully.")

# # ================================
# # Create environment
# # ================================
# env = NextBestActionEnv()

# # Reset environment to get initial state
# state, _ = env.reset()
# print(state)
# print("\n=== INITIAL STATE ===")
# print("Gender ", state[0])
# print("Marital Status ", state[1])
# print("Prefix ")
# print(state[2:23])
# print("Number patient at node ")
# print(state[23:44])
# print("Blood & Urine Result Time ", state[-2], " ", state[-1])
# print("State shape:", np.shape(state))

# # ================================
# # Predict next action (deterministic)
# # ================================
# action, _ = model.predict(state, deterministic=True)

# # ================================
# # Get all Q-values
# # ================================
# with torch.no_grad():
#     q_values = model.q_net(torch.tensor(state).float().unsqueeze(0))

# q_values = q_values.cpu().numpy().flatten()
# print("\n=== Q-VALUES ===")
# for i, q in enumerate(q_values):
#     act_name = env.activity_names[i] if hasattr(env, "activity_names") else f"Action {i}"
#     marker = "⬅️ SELECTED" if i == action else ""
#     print(f"{i:02d}: {act_name:<35} Q = {q:.4f} {marker}")

# print("\n=== MODEL PREDICTION ===")
# print(f"Predicted action index: {action}")
# if hasattr(env, "activity_names"):
#     print(f"Corresponding activity: {env.activity_names[action]}")

# # ================================
# # Thực hiện action để xem reward
# # ================================
# next_state, reward, done, truncated, info = env.step(action)

# print("reward", reward)

import os
import numpy as np
import torch
from stable_baselines3 import DQN
from src.online_training.next_best_action_env.env import NextBestActionEnv

# ================================
# Hàm generate test state
# ================================
def generate_test_state(gender, marital_status, prefix, num_actions=21, 
                        max_queue=5, mean_blood_test_time=10, mean_urine_test_time=5, seed=None):
    rng = np.random.default_rng(seed)
    queue_lengths = rng.integers(low=0, high=max_queue+1, size=num_actions).astype(float)

    actions = [
        "Registration", "Payment", "Get Triage Number", "Measure Vital Signs",
        "General Medicine Examination", "Eye Examination", "ENT Examination",
        "Dental Examination", "Gynecological Examination", "Breast Examination",
        "Blood Test", "Urine Test", "In-depth Eye Examination", "ENT Endoscopy",
        "Electrocardiogram (ECG)", "Post-bronchodilator Spirometry",
        "General Ultrasound", "Cardiac Ultrasound", "Chest X-ray",
        "DEXA Bone Density Scan", "Conclusion"
    ]

    try:
        blood_idx = actions.index("Blood Test")
    except ValueError:
        blood_idx = None
    try:
        urine_idx = actions.index("Urine Test")
    except ValueError:
        urine_idx = None

    blood_result_time = float(rng.uniform(0, mean_blood_test_time)) if blood_idx is not None and prefix[blood_idx]==1 else -1.0
    urine_result_time = float(rng.uniform(0, mean_urine_test_time)) if urine_idx is not None and prefix[urine_idx]==1 else -1.0

    state_list = [gender, marital_status] + prefix + queue_lengths.tolist() + [blood_result_time, urine_result_time]
    return np.array(state_list, dtype=np.float32)

# ================================
# Prefixes theo từng nhóm
# ================================
prefixes_for_male = [
    [0]*21, [1,0]+[0]*19, [1,1,1,1,1]+[0]*16,
    [1]*8+[0,0]+[0]*11,
    [1,1,1,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0],
    [1]*8 + [0]*2 + [1]*10 + [0]
]

prefixes_for_female_married = [
    [0]*21, [1,0]+[0]*19, [1,1,1,1,1]+[0]*16,
    [1]*10 + [0]*11,
    [1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1]*20 + [0]
]

prefixes_for_female_single = [
    [0]*21, [1,0]+[0]*19, [1,1,1,1,1]+[0]*16,
    [1]*9 + [0] + [0]*11,
    [1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0],
    [1]*8 + [0,1] + [1]*10 + [0]
]

# ================================
# Tạo tất cả states test
# ================================
all_states = []
for prefix in prefixes_for_male:
    all_states.append(generate_test_state(gender=1, marital_status=1, prefix=prefix))
for prefix in prefixes_for_male:
    all_states.append(generate_test_state(gender=1, marital_status=0, prefix=prefix))
for prefix in prefixes_for_female_married:
    all_states.append(generate_test_state(gender=0, marital_status=1, prefix=prefix))
for prefix in prefixes_for_female_single:
    all_states.append(generate_test_state(gender=0, marital_status=0, prefix=prefix))

print(f"Generated {len(all_states)} test states")

# ================================
# Load model
# ================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../../models/dqn_model.zip")

print(f"Loading model from {model_path}...")
model = DQN.load(model_path)
print("✅ Model loaded successfully.")

# ================================
# Create environment
# ================================
env = NextBestActionEnv()

# ================================
# Test DQN trên tất cả states
# ================================
for idx, state in enumerate(all_states):
    print(f"\n=== TEST STATE {idx+1} ===")
    print("Gender:", state[0])
    print("Marital Status:", state[1])
    print("Prefix:", state[2:23])
    print("Number patient at node:", state[23:44])
    print("Blood & Urine Result Time:", state[-2], state[-1])

    # Predict
    action, _ = model.predict(state, deterministic=True)
    print(f"Predicted action index: {action}")
    if hasattr(env, "activity_names"):
        print(f"Corresponding activity: {env.activity_names[action]}")

    # Q-values
    with torch.no_grad():
        q_values = model.q_net(torch.tensor(state).float().unsqueeze(0)).cpu().numpy().flatten()

    print("Q-values:")
    all_acts = env.activity_names if hasattr(env, "activity_names") else [f"Action {i}" for i in range(len(q_values))]
    for i, q in enumerate(q_values):
        marker = "⬅️ SELECTED" if i == action else ""
        print(f"{i:02d}: {all_acts[i]:<35} Q = {q:.4f} {marker}")
    print("=====================================")
