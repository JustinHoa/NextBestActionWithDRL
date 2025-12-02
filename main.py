import os
from collections import deque
import sys

import numpy as np
import torch
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_agent import DuelingAgent
from agents.rainbow_agent import RainbowAgent
from agents.per_dqn_agent import PerDqnAgent
from common.env import FillBlanksEnv
from common.utils import (
    ACTION_SIZE,
    STATE_SIZE,
    TRAIN_CONFIG,
    append_to_pickle,
    ensure_dir,
    plot_training_status,
)
from simulation.simulation_process import run_simulation


def train(agent, algo_name: str, version_id: int):
    """Hàm training chính cho một thế hệ."""
    config = TRAIN_CONFIG[version_id]
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)

    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING: [{algo_name}] - VERSION {version_id}")
    print(f"   Description: {config['description']}")
    print(f"   Data: {config['data_file']}")
    print(f"{'='*60}\n")

    env = FillBlanksEnv(STATE_SIZE, ACTION_SIZE, data_path=config["data_file"])

    # Load model từ thế hệ trước (nếu có)
    if config["load_model"]:
        load_path = os.path.join(log_dir, config["load_model"])
        if os.path.exists(load_path):
            print(f"🔄 Loading pre-trained model from: {load_path}")
            agent.load(load_path)
        else:
            print(f"⚠️ Warning: Model file not found at {load_path}. Training from scratch.")

    # Cập nhật learning rate
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = config["lr"]

    scores_window = deque(maxlen=100)
    all_scores, all_losses = [], []
    eps = config["eps_start"]

    # Rainbow không dùng epsilon
    use_epsilon = not isinstance(agent, RainbowAgent)

    tqdm_bar = tqdm(range(1, config["episodes"] + 1), desc=f"Training V{version_id}")
    for i_episode in tqdm_bar:
        state = env.reset()
        score = 0
        for _ in range(30):  # Max steps per episode
            mask = env.get_action_mask()
            
            current_eps = eps if use_epsilon else 0.0
            action = agent.act(state, mask, current_eps)
            
            next_state, reward, done = env.step(action)
            next_mask = env.get_action_mask()

            loss = agent.step(state, action, reward, next_state, done, next_mask)
            if loss is not None:
                all_losses.append(loss)

            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        all_scores.append(score)
        if use_epsilon:
            eps = max(config["eps_end"], eps * config["eps_decay"])

        if i_episode % 100 == 0:
            tqdm_bar.set_postfix(avg_score=f"{np.mean(scores_window):.2f}", eps=f"{eps:.3f}")

    # Lưu model cuối cùng
    save_path = os.path.join(log_dir, config["save_name"])
    agent.save(save_path)

    # Vẽ biểu đồ cuối cùng
    plot_training_status(all_scores, all_losses, algo_name, version_id, save_dir=log_dir)
    print(f"\n✅ Finished Version {version_id}. Model saved to {save_path}")

    # Chạy simulation để sinh data cho thế hệ tiếp theo
    if version_id < 3:
        print(f"\n🌍 Running simulation with new model to generate data for Gen {version_id + 1}...")
        run_simulation(num_patients=200, agent=agent, version_output=str(version_id), is_model_run=True)

SUPPORTED_ALGOS = ["DQN", "DDQN", "PerDQN", "Dueling", "Rainbow"]

if __name__ == "__main__":
    # --- 1. Lấy tham số từ command line ---
    if len(sys.argv) < 2:
        print("❌ Bạn chưa truyền thuật toán.")
        print("   Ví dụ cách chạy:")
        print("   python main.py DQN")
        print("   python main.py DDQN")
        print("   python main.py Dueling")
        print("   python main.py Rainbow")
        sys.exit(1)

    ALGO_TO_RUN = sys.argv[1].strip()

    def get_agent(algo_name: str):
        """Factory function để lấy agent tương ứng."""
        if algo_name == "DQN":
            return DQNAgent(STATE_SIZE, ACTION_SIZE)
        if algo_name == "DDQN":
            return DDQNAgent(STATE_SIZE, ACTION_SIZE)
        if algo_name == "Dueling":
            return DuelingAgent(STATE_SIZE, ACTION_SIZE)
        if algo_name == "PerDQN":
            return PerDqnAgent(STATE_SIZE, ACTION_SIZE)
        if algo_name == "Rainbow":
            return RainbowAgent(STATE_SIZE, ACTION_SIZE)
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # --- 2. Kiểm tra thuật toán hợp lệ ---
    if ALGO_TO_RUN not in SUPPORTED_ALGOS:
        print(f"❌ Thuật toán không hợp lệ: {ALGO_TO_RUN}")
        print("   Hỗ trợ:", SUPPORTED_ALGOS)
        sys.exit(1)

    print(f"🚀 ALGO_TO_RUN = {ALGO_TO_RUN}\n")

    agent = get_agent(ALGO_TO_RUN)

    # --- 3. Kiểm tra file dữ liệu ban đầu ---
    initial_data_path = os.path.join("data", "raw", "200_queue_log_version_0.csv")

    if not os.path.exists(initial_data_path):
        print("--- Initial data (version 0) not found. ---")
        print("🌍 Running RANDOM simulation to generate initial queue log...")

        run_simulation(
            num_patients=200,
            agent=None,             # Random mode
            version_output="0",
            is_model_run=False
        )

        print("--- Initial data generated successfully. ---\n")

    # --- 4. Chạy đầy đủ 3 thế hệ training ---
    print(f"--- Starting Full Training Cycle for [{ALGO_TO_RUN}] ---")

    train(agent, ALGO_TO_RUN, 1)
    train(agent, ALGO_TO_RUN, 2)
    train(agent, ALGO_TO_RUN, 3)

    print("\n🎉🎉🎉 All training generations completed! 🎉🎉🎉")