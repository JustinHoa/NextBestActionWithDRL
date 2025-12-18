import os
import pickle
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- GLOBAL CONSTANTS ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STATE_SIZE = 45
ACTION_SIZE = 21

# --- TRAINING CONFIG ---
TRAIN_CONFIG = {
    1: {
        "description": "Gen 1 - Train from scratch on random data",
        "data_file": "data/raw/200_queue_log_version_0.csv",
        "episodes": 10000,
        "lr": 1e-4,
        "eps_start": 1.0,
        "eps_decay": 0.9996,
        "eps_end": 0.1,
        "load_model": None,
        "save_name": "final_gen1.pth",
    },
    2: {
        "description": "Gen 2 - Fine-tune on Gen 1 agent's data",
        "data_file": "data/raw/200_queue_log_version_1.csv",
        "episodes": 6000,
        "lr": 1e-5,
        "eps_start": 0.1,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_gen1.pth",
        "save_name": "final_gen2.pth",
    },
    3: {
        "description": "Gen 3 - Aggressive fine-tune with smaller LR",
        "data_file": "data/raw/200_queue_log_version_2.csv",
        "episodes": 4000,
        "lr": 1e-5,
        "eps_start": 0.05,
        "eps_decay": 0.99,
        "eps_end": 0.001,
        "load_model": "final_gen2.pth",
        "save_name": "final_gen3.pth",
    },
    4: {
        "description": "Gen 4 - Continue fine-tune on Gen 3 agent's data",
        "data_file": "data/raw/200_queue_log_version_3.csv",
        "episodes": 3000,
        "lr": 1e-5,
        "eps_start": 0.05,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_gen3.pth",
        "save_name": "final_gen4.pth",
    },
    5: {
        "description": "Gen 5 - Final fine-tune (max generations)",
        "data_file": "data/raw/200_queue_log_version_4.csv",
        "episodes": 3000,
        "lr": 1e-5,
        "eps_start": 0.05,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_gen4.pth",
        "save_name": "final_gen5.pth",
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_to_pickle(file_path: str, new_data: List[float]) -> None:
    """Nối dữ liệu mới vào cuối file pickle."""
    data = load_pickle(file_path, default=[])
    data.extend(new_data)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: str, default: Optional[List[float]] = None) -> List[float]:
    """Tải dữ liệu từ file pickle."""
    if default is None:
        default = []
    if not os.path.exists(file_path):
        return default
    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_training_status(
    scores: List[float],
    losses: List[float],
    algo_name: str,
    version_id: int,
    save_dir: str
) -> None:
    """Vẽ và lưu biểu đồ training (reward và loss)."""
    if not scores or not losses:
        print("⚠️ No data to plot.")
        return

    ensure_dir(save_dir)
    plt.figure(figsize=(14, 6))

    # --- Biểu đồ Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(losses, color="tab:red", alpha=0.5, label="Raw Loss")
    if len(losses) > 100:
        ma_loss = np.convolve(losses, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(losses)), ma_loss, color="darkred", linewidth=1.5, label="MA Loss (50)")
    plt.title(f"[{algo_name}] Training Loss - Version {version_id}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- Biểu đồ Reward ---
    plt.subplot(1, 2, 2)
    plt.plot(scores, color="tab:blue", alpha=0.3, label="Episode Score")
    if len(scores) >= 100:
        ma_score = np.convolve(scores, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, len(scores)), ma_score, color="orange", linewidth=2, label="Avg Score (100 eps)")
    plt.title(f"[{algo_name}] Rewards - Version {version_id}")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = os.path.join(save_dir, f"monitoring_v{version_id}_final.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Plot saved to {plot_path}")
