import os
from collections import deque
import sys
import time
import glob
import shutil
import random

import numpy as np
import torch
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_agent import DuelingAgent
from agents.rainbow_agent import RainbowAgent
from agents.per_dqn_agent import PerDqnAgent
from agents.multi_step_dqn_agent import MultiStepDqnAgent
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

# === REPRODUCIBILITY SETTINGS ===
TRAIN_SEED = 123  # Fixed seed for training reproducibility
EVAL_SEED = 42    # Fixed seed for evaluation/simulation

def set_deterministic_mode():
    """Set all random number generators to deterministic mode for reproducibility."""
    # Set seeds for all RNG sources
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    
    # GPU deterministic settings (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This may reduce performance but ensures reproducibility
    
    print(f" Deterministic mode enabled with TRAIN_SEED={TRAIN_SEED}")

def save_rng_state():
    """Save current RNG states before simulation calls."""
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def restore_rng_state(state_dict):
    """Restore RNG states after simulation calls to prevent interference."""
    random.setstate(state_dict['random'])
    np.random.set_state(state_dict['numpy'])
    torch.set_rng_state(state_dict['torch'])
    if torch.cuda.is_available() and state_dict['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state_dict['torch_cuda'])

def run_simulation_isolated(num_patients, agent, version_output, is_model_run=False, seed=None):
    """Run simulation with isolated RNG state to prevent training interference."""
    # Save current training RNG state
    training_state = save_rng_state()
    
    try:
        # Run simulation with its own seed
        result = run_simulation(num_patients, agent, version_output, is_model_run, seed)
        return result
    finally:
        # Always restore training RNG state
        restore_rng_state(training_state)


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
    if algo_name == "MultiStepDQN":
        return MultiStepDqnAgent(STATE_SIZE, ACTION_SIZE)
    if algo_name == "Rainbow":
        return RainbowAgent(STATE_SIZE, ACTION_SIZE)
    raise ValueError(f"Unknown algorithm: {algo_name}")


def _append_note(log_dir: str, text: str) -> None:
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, "training_notes.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)


def _extract_episode_from_ckpt_name(filename: str) -> int:
    base = os.path.basename(filename)
    try:
        ep_part = base.split("_ep", 1)[1]
        return int(ep_part.split(".pth", 1)[0])
    except Exception:
        return -1


def _evaluate_checkpoints(algo_name: str, gen_id: int, log_dir: str, seed: int, baseline_avg_time: float):
    pattern = os.path.join(log_dir, f"checkpoint_v{gen_id}_ep*.pth")
    ckpt_paths = sorted(glob.glob(pattern), key=_extract_episode_from_ckpt_name)
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found for Gen {gen_id} at: {pattern}")

    results = []
    best = None

    _append_note(
        log_dir,
        f"\nGen {gen_id}:\n"
        f"Simulation Process Avg Time:\n"
        f"Random (Baseline): {baseline_avg_time:.2f}\n",
    )

    for ckpt_path in ckpt_paths:
        ep = _extract_episode_from_ckpt_name(ckpt_path)
        agent = get_agent(algo_name)
        agent.load(ckpt_path)
        avg_time = run_simulation_isolated(num_patients=200, agent=agent, version_output=f"eval_gen{gen_id}_ep{ep}", seed=seed)
        results.append((ep, ckpt_path, float(avg_time)))
        _append_note(log_dir, f"Checkpoint (ep={ep}): {avg_time:.2f}\n")

        if best is None or avg_time < best[2]:
            best = (ep, ckpt_path, float(avg_time))

    assert best is not None
    _append_note(log_dir, f"=> Choose Checkpoint (ep={best[0]}) for Gen {gen_id}.\n")
    return best, results


def train_one_generation(agent, algo_name: str, gen_id: int):
    """Train một thế hệ và trả về danh sách checkpoint paths được tạo ra."""
    config = TRAIN_CONFIG[gen_id]
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)

    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING: [{algo_name}] - GEN {gen_id}")
    print(f"   Description: {config['description']}")
    print(f"   Data: {config['data_file']}")
    print(f"{'='*60}\n")

    env = FillBlanksEnv(STATE_SIZE, ACTION_SIZE, data_path=config["data_file"])

    if config["load_model"]:
        load_path = os.path.join(log_dir, config["load_model"])
        if os.path.exists(load_path):
            print(f"🔄 Loading pre-trained model from: {load_path}")
            agent.load(load_path)
        else:
            print(f"⚠️ Warning: Model file not found at {load_path}. Training from scratch.")

    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = config["lr"]

    scores_window = deque(maxlen=100)
    all_scores, all_losses = [], []
    eps = config["eps_start"]
    use_epsilon = not isinstance(agent, RainbowAgent)

    ckpt_paths = []
    t0 = time.time()

    tqdm_bar = tqdm(range(1, config["episodes"] + 1), desc=f"Training Gen {gen_id}")
    for i_episode in tqdm_bar:
        state = env.reset()
        score = 0
        for _ in range(30):
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

        if i_episode % 1000 == 0:
            ckpt_name = f"checkpoint_v{gen_id}_ep{i_episode}.pth"
            ckpt_path = os.path.join(log_dir, ckpt_name)
            agent.save(ckpt_path)
            ckpt_paths.append(ckpt_path)
            tqdm_bar.write(f"💾 Checkpoint saved: {ckpt_name}")

    train_seconds = time.time() - t0
    train_minutes = train_seconds / 60.0

    plot_training_status(all_scores, all_losses, algo_name, gen_id, save_dir=log_dir)

    _append_note(
        log_dir,
        f"Episode: {config['episodes']} episode\n"
        f"Training Time: {train_minutes:.2f} minutes\n",
    )

    return ckpt_paths


SUPPORTED_ALGOS = ["DQN", "DDQN", "PerDQN", "Dueling", "Rainbow", "MultiStepDQN"]

if __name__ == "__main__":
    # --- 1. Lấy tham số từ command line ---
    if len(sys.argv) < 2:
        print("❌ Bạn chưa truyền thuật toán.")
        print("   Ví dụ cách chạy:")
        print("   python main.py DQN")
        print("   python main.py DDQN")
        print("   python main.py Dueling")
        print("   python main.py Rainbow")
        print("   python main.py MultiStepDQN")
        sys.exit(1)

    ALGO_TO_RUN = sys.argv[1].strip()

    # --- 2. Kiểm tra thuật toán hợp lệ ---
    if ALGO_TO_RUN not in SUPPORTED_ALGOS:
        print(f"❌ Thuật toán không hợp lệ: {ALGO_TO_RUN}")
        print("   Hỗ trợ:", SUPPORTED_ALGOS)
        sys.exit(1)

    print(f"🚀 ALGO_TO_RUN = {ALGO_TO_RUN}\n")
    
    # === ENABLE DETERMINISTIC MODE FOR REPRODUCIBILITY ===
    set_deterministic_mode()

    agent = get_agent(ALGO_TO_RUN)

    print(f"--- Starting Full Training Cycle for [{ALGO_TO_RUN}] ---")

    log_dir = os.path.join("logs", ALGO_TO_RUN)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {ALGO_TO_RUN} ===\n")

    TEST_SEED = 42
    baseline_avg_time = run_simulation_isolated(num_patients=200, agent=None, version_output="random_base", seed=TEST_SEED)

    prev_best_avg_time = None

    for gen_id in range(1, 6):
        if gen_id not in TRAIN_CONFIG:
            break

        ckpt_paths = train_one_generation(agent, ALGO_TO_RUN, gen_id)

        best, _ = _evaluate_checkpoints(
            algo_name=ALGO_TO_RUN,
            gen_id=gen_id,
            log_dir=log_dir,
            seed=TEST_SEED,
            baseline_avg_time=baseline_avg_time,
        )

        best_ep, best_ckpt_path, best_avg_time = best
        final_name = TRAIN_CONFIG[gen_id]["save_name"]
        final_path = os.path.join(log_dir, final_name)
        shutil.copyfile(best_ckpt_path, final_path)
        _append_note(log_dir, f"Final Model: {final_name} (from ep={best_ep}) | Avg Time: {best_avg_time:.2f}\n")

        if prev_best_avg_time is not None and best_avg_time >= prev_best_avg_time:
            _append_note(
                log_dir,
                f"STOP: Gen {gen_id} Avg Time ({best_avg_time:.2f}) >= Gen {gen_id - 1} Avg Time ({prev_best_avg_time:.2f})\n",
            )
            print(f"🛑 STOP: Gen {gen_id} did not improve over Gen {gen_id - 1}.")
            break

        prev_best_avg_time = best_avg_time

        if gen_id < 5 and (gen_id + 1) in TRAIN_CONFIG:
            agent_for_data = get_agent(ALGO_TO_RUN)
            agent_for_data.load(final_path)
            run_simulation_isolated(num_patients=200, agent=agent_for_data, version_output=str(gen_id), is_model_run=True, seed=TEST_SEED)

    print("\n🎉🎉🎉 Training cycle finished! 🎉🎉🎉")