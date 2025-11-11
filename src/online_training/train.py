import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.online_training.next_best_action_env.env import NextBestActionEnv

# =========================================
# 0️⃣ Chọn chế độ test hay full training
# =========================================
TEST_MODE = False   # True: chạy nhanh test, False: train full
print(f"TEST_MODE = {TEST_MODE}, training {'fast test' if TEST_MODE else 'full'}")

# =========================================
# 1️⃣ Tạo environment (bọc VecNormalize + Monitor)
# =========================================
env = DummyVecEnv([lambda: Monitor(NextBestActionEnv())])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=1.0)

# =========================================
# 2️⃣ Khởi tạo model DQN
# =========================================
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=100000,
    learning_starts=2000,
    batch_size=64,
    gamma=0.98,
    tau=0.01,
    target_update_interval=1000,
    train_freq=4,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
)

# =========================================
# 3️⃣ Train model với early stopping / test nhanh
# =========================================
print("\n🚀 Training started...\n")

eval_rewards = []
total_steps_done = 0
early_stop = False

# Cấu hình cho test nhanh hoặc train full
if TEST_MODE:
    step_increment = 1
    timesteps_per_step = 2000
    max_steps = 5  # test nhanh chạy vài block
else:
    step_increment = 20000
    timesteps_per_step = 20000
    max_steps = 200000

for step in range(0, max_steps, step_increment):
    model.learn(total_timesteps=timesteps_per_step, reset_num_timesteps=False)
    total_steps_done += timesteps_per_step

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=2 if TEST_MODE else 10)
    eval_rewards.append(mean_reward)
    print(f"Step {total_steps_done:6d} | Mean reward: {mean_reward:.3f}")

    # Early stopping nếu reward ổn định và > 0.8 (chỉ dùng khi train full)
    if not TEST_MODE and len(eval_rewards) > 3:
        recent = eval_rewards[-3:]
        if max(recent) - min(recent) < 0.02 and mean_reward > 0.8:
            print(f"\n✅ Reward has stabilized. Stopping training at step {total_steps_done}.\n")
            early_stop = True
            break

if not early_stop:
    print(f"\n✅ Training finished after {total_steps_done} steps.\n")

# =========================================
# 4️⃣ Lưu model
# =========================================
save_path = os.path.abspath(os.path.join("models", "dqn_model_test" if TEST_MODE else "dqn_model"))
model.save(save_path)
print(f"✅ Model saved to: {save_path}.zip")

# =========================================
# 5️⃣ Đánh giá model
mean_eval_episodes = 2 if TEST_MODE else 10
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=mean_eval_episodes)
print(f"\n📊 Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}")
