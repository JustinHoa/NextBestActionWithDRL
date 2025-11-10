import os
import numpy as np
import torch
from stable_baselines3 import DQN
from src.online_training.next_best_action_env.env import NextBestActionEnv

# ================================
# Load model
# ================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../../models/dqn_model_best.zip")

print(f"Loading model from {model_path}...")
model = DQN.load(model_path)
print("✅ Model loaded successfully.")

# ================================
# Create environment
# ================================
env = NextBestActionEnv()

# Reset environment to get initial state
state, _ = env.reset()
print(state)
print("\n=== INITIAL STATE ===")
print("Gender ", state[0])
print("Marital Status ", state[1])
print("Prefix ")
print(state[2:23])
print("Number patient at node ")
print(state[23:44])
print("Blood & Urine Result Time ", state[-2], " ", state[-1])
print("State shape:", np.shape(state))

# ================================
# Predict next action (deterministic)
# ================================
action, _ = model.predict(state, deterministic=True)

# ================================
# Get all Q-values
# ================================
with torch.no_grad():
    q_values = model.q_net(torch.tensor(state).float().unsqueeze(0))

q_values = q_values.cpu().numpy().flatten()
print("\n=== Q-VALUES ===")
for i, q in enumerate(q_values):
    act_name = env.activity_names[i] if hasattr(env, "activity_names") else f"Action {i}"
    marker = "⬅️ SELECTED" if i == action else ""
    print(f"{i:02d}: {act_name:<35} Q = {q:.4f} {marker}")

print("\n=== MODEL PREDICTION ===")
print(f"Predicted action index: {action}")
if hasattr(env, "activity_names"):
    print(f"Corresponding activity: {env.activity_names[action]}")

# ================================
# Thực hiện action để xem reward
# ================================
next_state, reward, done, truncated, info = env.step(action)

print("reward", reward)