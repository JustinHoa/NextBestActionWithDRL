import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from src.online_training.next_best_action_env.env import NextBestActionEnv

# =========================================
# 1️⃣ Tạo environment (bọc Monitor)
# =========================================
env = Monitor(NextBestActionEnv())

# =========================================
# 2️⃣ Khởi tạo model DQN
# =========================================
model = DQN(
    policy="MlpPolicy",   # Mạng policy dạng MLP (fully-connected NN)
    env=env,               # Custom Gymnasium environment
    learning_rate=1e-3,    # Tốc độ học
    buffer_size=50000,     # Replay buffer (max 50.000 mẫu)
    learning_starts=1000,  # Bắt đầu train sau khi có 1.000 bước đầu tiên
    batch_size=64,         # Batch size cho mỗi lần update
    tau=0.01,              # Soft update cho target network
    gamma=0.99,            # Discount factor
    target_update_interval=1000,  # Đồng bộ target network mỗi 1000 bước
    train_freq=4,          # Cứ 4 bước thì train 1 lần
    exploration_fraction=0.1,     # 10% đầu cho thăm dò epsilon-greedy
    exploration_final_eps=0.05,   # Sau đó epsilon = 0.05
    verbose=1,             # In log trong quá trình train
)

# =========================================
# 3️⃣ Train model (ngắn để test)
# =========================================
print("\n🚀 Training started...\n")
model.learn(total_timesteps=200000)
print("\n✅ Training finished!\n")

# =========================================
# 4️⃣ Lưu model
# =========================================
save_path = os.path.abspath("dqn_model_test")
model.save(save_path)
print(f"✅ Model saved to: {save_path}.zip")

# =========================================
# 5️⃣ Đánh giá model
# =========================================
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"\n📊 Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}")