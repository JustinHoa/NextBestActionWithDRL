import os
import copy
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- GLOBAL CONSTANTS ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STATE_SIZE = 45
ACTION_SIZE = 21

# Queue-constrained variants
STATE_SIZE_STATIC_QUEUE = 66  # [gender, marital, 21 blanks, 21 waiting_times, 21 queue_utilization, 1 norm_time]
ACTION_SIZE_STATIC_QUEUE = 22  # 21 activities + 1 mock action

STATE_SIZE_DYNAMIC_QUEUE = 87  # [gender, marital, 21 blanks, 21 waiting_times, 21 queue_utilization, 21 is_expanded, 1 norm_time]
ACTION_SIZE_DYNAMIC_QUEUE = 21  # 21 activities (no mock action)

STATE_SIZE_PRIORITY_QUEUE = 86  # [gender, marital, 21 blanks, 21 queues, 21 emergency_flags, 21 mock_queues]
ACTION_SIZE_PRIORITY_QUEUE = 22  # 21 activities + 1 mock action

# --- TRAINING CONFIG ---
_BASE_TRAIN_CONFIG = {
    1: {
        "description": "Gen 1 - Train from scratch on random data",
        "data_file": "data/raw/200_queue_log_version_random_base.csv",
        "episodes": 6000, # 6000
        "lr": 1e-4,
        "eps_start": 1.0,
        "eps_decay": 0.9994,
        "eps_end": 0.1,
        "load_model": None,
        "save_name": "final_gen1.pth",
    },
    2: {
        "description": "Gen 2 - Fine-tune on Gen 1 agent's data",
        "data_file": "data/raw/200_queue_log_version_1.csv",
        "episodes": 10000, # 10 000
        "lr": 3e-5,
        "eps_start": 0.5,
        "eps_decay": 0.9994,
        "eps_end": 0.05,
        "load_model": "final_gen1.pth",
        "save_name": "final_gen2.pth",
    },
    3: {
        "description": "Gen 3 - Aggressive fine-tune with smaller LR",
        "data_file": "data/raw/200_queue_log_version_2.csv",
        "episodes": 8000,
        "lr": 1e-5,
        "eps_start": 0.8,
        "eps_decay": 0.9994,
        "eps_end": 0.1,
        "load_model": "final_gen2.pth",
        "save_name": "final_gen3.pth",
    },
    4: {
        "description": "Gen 4 - Continue fine-tune on Gen 3 agent's data",
        "data_file": "data/raw/200_queue_log_version_3.csv",
        "episodes": 8000,
        "lr": 1e-5,
        "eps_start": 0.3,
        "eps_decay": 0.9992,
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

_PENALTY_DQN_CONFIG = {
    1: {
        "description": "PenaltyDQN - Train from scratch with penalty-based rewards",
        "data_file": "data/raw/queue_log_200_random_base.csv",
        "episodes": 100000,
        "lr": 1e-4,
        "eps_start": 1.0,
        "eps_decay": 0.9999,
        "eps_end": 0.01,
        "load_model": None,
        "save_name": "final_200_gen_1.pth",
    },
}

_STATIC_QUEUE_DQN_CONFIG = {
    1: {
        "description": "StaticQueueDQN Gen 1 - Train from scratch with static queue constraints",
        "data_file": "data/raw/queue_log_200_staticqueue_random_base.csv",
        "episodes": 1000,
        "lr": 1e-4,
        "eps_start": 1.0,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": None,
        "save_name": "final_200_gen_1.pth",
    },
    2: {
        "description": "StaticQueueDQN Gen 2 - Fine-tune on Gen 1 data",
        "data_file": "data/raw/queue_log_200_staticqueue_gen_1.csv",
        "episodes": 800,
        "lr": 3e-5,
        "eps_start": 0.5,
        "eps_decay": 0.999,
        "eps_end": 0.05,
        "load_model": "final_200_gen_1.pth",
        "save_name": "final_200_gen_2.pth",
    },
    3: {
        "description": "StaticQueueDQN Gen 3 - Aggressive fine-tune",
        "data_file": "data/raw/queue_log_200_staticqueue_gen_2.csv",
        "episodes": 600,
        "lr": 1e-5,
        "eps_start": 0.3,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_200_gen_2.pth",
        "save_name": "final_200_gen_3.pth",
    },
}

_DYNAMIC_QUEUE_DQN_CONFIG = {
    1: {
        "description": "DynamicQueueDQN Gen 1 - Train from scratch with dynamic queue expansion",
        "data_file": "data/raw/queue_log_200_dynamicqueue_random_base.csv",
        "episodes": 1000,
        "lr": 1e-4,
        "eps_start": 1.0,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": None,
        "save_name": "final_200_gen_1.pth",
    },
    2: {
        "description": "DynamicQueueDQN Gen 2 - Fine-tune on Gen 1 data",
        "data_file": "data/raw/queue_log_200_dynamicqueue_gen_1.csv",
        "episodes": 800,
        "lr": 3e-5,
        "eps_start": 0.5,
        "eps_decay": 0.999,
        "eps_end": 0.05,
        "load_model": "final_200_gen_1.pth",
        "save_name": "final_200_gen_2.pth",
    },
    3: {
        "description": "DynamicQueueDQN Gen 3 - Aggressive fine-tune",
        "data_file": "data/raw/queue_log_200_dynamicqueue_gen_2.csv",
        "episodes": 600,
        "lr": 1e-5,
        "eps_start": 0.3,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_200_gen_2.pth",
        "save_name": "final_200_gen_3.pth",
    },
}

_PRIORITY_QUEUE_DQN_CONFIG = {
    1: {
        "description": "Gen 1: Train from scratch with priority queue",
        "data_file": "data/raw/queue_log_200_random_base.csv",
        "episodes": 1000,
        "lr": 1e-4,
        "eps_start": 1.0,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": None,
        "save_name": "final_200_gen_1.pth",
    },
    2: {
        "description": "Gen 2: Fine-tune on Gen 1 data",
        "data_file": "data/raw/queue_log_200_priorityqueue_gen_1.csv",
        "episodes": 800,
        "lr": 3e-5,
        "eps_start": 0.3,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_200_gen_1.pth",
        "save_name": "final_200_gen_2.pth",
    },
    3: {
        "description": "Gen 3: Aggressive fine-tune",
        "data_file": "data/raw/queue_log_200_priorityqueue_gen_2.csv",
        "episodes": 600,
        "lr": 1e-5,
        "eps_start": 0.1,
        "eps_decay": 0.999,
        "eps_end": 0.01,
        "load_model": "final_200_gen_2.pth",
        "save_name": "final_200_gen_3.pth",
    },
}

TRAIN_CONFIGS = {
    "DQN": copy.deepcopy(_BASE_TRAIN_CONFIG),
    "DDQN": copy.deepcopy(_BASE_TRAIN_CONFIG),
    "Dueling": copy.deepcopy(_BASE_TRAIN_CONFIG),
    "PerDQN": copy.deepcopy(_BASE_TRAIN_CONFIG),
    "MultiStepDQN": copy.deepcopy(_BASE_TRAIN_CONFIG),
    "Rainbow": copy.deepcopy(_BASE_TRAIN_CONFIG),
    "PenaltyDQN": copy.deepcopy(_PENALTY_DQN_CONFIG),
    # Static Queue Variants
    "StaticQueueDQN": copy.deepcopy(_STATIC_QUEUE_DQN_CONFIG),
    "StaticQueueDDQN": copy.deepcopy(_STATIC_QUEUE_DQN_CONFIG),
    "StaticQueueDueling": copy.deepcopy(_STATIC_QUEUE_DQN_CONFIG),
    "StaticQueuePerDQN": copy.deepcopy(_STATIC_QUEUE_DQN_CONFIG),
    "StaticQueueRainbow": copy.deepcopy(_STATIC_QUEUE_DQN_CONFIG),
    "StaticQueueMultiStepDQN": copy.deepcopy(_STATIC_QUEUE_DQN_CONFIG),
    # Dynamic Queue Variants
    "DynamicQueueDQN": copy.deepcopy(_DYNAMIC_QUEUE_DQN_CONFIG),
    "DynamicQueueDDQN": copy.deepcopy(_DYNAMIC_QUEUE_DQN_CONFIG),
    "DynamicQueueDueling": copy.deepcopy(_DYNAMIC_QUEUE_DQN_CONFIG),
    "DynamicQueuePerDQN": copy.deepcopy(_DYNAMIC_QUEUE_DQN_CONFIG),
    "DynamicQueueRainbow": copy.deepcopy(_DYNAMIC_QUEUE_DQN_CONFIG),
    "DynamicQueueMultiStepDQN": copy.deepcopy(_DYNAMIC_QUEUE_DQN_CONFIG),
    # Priority Queue Variants
    "PriorityQueueDQN": copy.deepcopy(_PRIORITY_QUEUE_DQN_CONFIG),
    "PriorityQueueDDQN": copy.deepcopy(_PRIORITY_QUEUE_DQN_CONFIG),
    "PriorityQueueDueling": copy.deepcopy(_PRIORITY_QUEUE_DQN_CONFIG),
    "PriorityQueuePerDQN": copy.deepcopy(_PRIORITY_QUEUE_DQN_CONFIG),
    "PriorityQueueRainbow": copy.deepcopy(_PRIORITY_QUEUE_DQN_CONFIG),
    "PriorityQueueMultiStepDQN": copy.deepcopy(_PRIORITY_QUEUE_DQN_CONFIG),
}


def get_train_config(algo_name: str, num_patients: int = 200) -> Dict[int, Dict[str, object]]:
    """Get training config with dynamic paths based on num_patients."""
    config = copy.deepcopy(TRAIN_CONFIGS[algo_name])
    
    # Determine queue type prefix for data files
    queue_prefix = ""
    if algo_name.startswith("StaticQueue"):
        queue_prefix = "staticqueue"
    elif algo_name.startswith("DynamicQueue"):
        queue_prefix = "dynamicqueue"
    elif algo_name.startswith("PriorityQueue"):
        queue_prefix = "priorityqueue"
    
    # Update data_file, load_model, and save_name to use num_patients
    for gen_id in config:
        # Update data_file
        if gen_id == 1:
            # Gen 1 uses random_base or queue-specific random_base
            if queue_prefix:
                config[gen_id]["data_file"] = f"data/raw/queue_log_{num_patients}_{queue_prefix}_random_base.csv"
            else:
                config[gen_id]["data_file"] = f"data/raw/queue_log_{num_patients}_random_base.csv"
        else:
            # Gen 2+ uses previous generation's data
            prev_gen = gen_id - 1
            if queue_prefix:
                config[gen_id]["data_file"] = f"data/raw/queue_log_{num_patients}_{queue_prefix}_gen_{prev_gen}.csv"
            else:
                config[gen_id]["data_file"] = f"data/raw/queue_log_{num_patients}_{algo_name}_gen_{prev_gen}.csv"
        
        # Update load_model
        if config[gen_id]["load_model"] is not None:
            prev_gen = gen_id - 1
            config[gen_id]["load_model"] = f"final_{num_patients}_gen_{prev_gen}.pth"
        
        # Update save_name
        config[gen_id]["save_name"] = f"final_{num_patients}_gen_{gen_id}.pth"
    
    return config


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
    save_dir: str,
    num_patients: int = 200
) -> None:
    """Vẽ và lưu biểu đồ training (reward và loss)."""
    if not scores or not losses:
        print("⚠️ No data to plot.")
        return

    ensure_dir(save_dir)
    plt.figure(figsize=(14, 6))

    # --- Biểu đồ Loss ---
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(losses, color="tab:red", alpha=0.5, label="Raw Loss")
    if len(losses) > 100:
        ma_loss = np.convolve(losses, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(losses)), ma_loss, color="darkred", linewidth=1.5, label="MA Loss (50)")
    plt.title(f"[{algo_name}] Training Loss - Version {version_id}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    ax1.legend(loc="upper left")
    plt.grid(alpha=0.3)
    
    # Add inset plot for Gen 1 and Gen 2 to zoom into the latter half
    if version_id in [1, 2] and len(losses) > 100:
        # Create inset axes (position: [left, bottom, width, height] in figure coordinates)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax1, width="40%", height="40%", loc='upper right', borderpad=2)
        
        # Zoom into the latter 50% of the data
        start_idx = len(losses) // 2
        losses_zoom = losses[start_idx:]
        axins.plot(range(start_idx, len(losses)), losses_zoom, color="tab:red", alpha=0.5)
        if len(losses_zoom) > 50:
            ma_loss_zoom = np.convolve(losses_zoom, np.ones(50) / 50, mode="valid")
            axins.plot(range(start_idx + 49, len(losses)), ma_loss_zoom, color="darkred", linewidth=1.5)
        
        axins.set_xlabel("Steps", fontsize=8)
        axins.set_ylabel("Loss", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3)
        axins.set_title("Zoomed (50% latter)", fontsize=8)

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

    plot_path = os.path.join(save_dir, f"monitoring_{num_patients}_gen_{version_id}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Plot saved to {plot_path}")
