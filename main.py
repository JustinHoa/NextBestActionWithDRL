import os
from collections import deque
import sys
import time
import glob
import shutil
import random
import json

import numpy as np
import torch
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_agent import DuelingAgent
from agents.rainbow_agent import RainbowAgent
from agents.per_dqn_agent import PerDqnAgent
from agents.multi_step_dqn_agent import MultiStepDqnAgent
from agents.learning_to_act_agent import LearningToActAgent
from agents.forlaps import run_forlaps_once as run_forlaps_once_forlaps
from common.env import FillBlanksEnv
from common.utils import (
    ACTION_SIZE,
    STATE_SIZE,
    DEVICE,
    get_train_config,
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
    if algo_name == "FORLAPS":
        # FORLAPS uses a standard DQN network; training loop is offline and lives in agents/forlaps.py
        return DQNAgent(STATE_SIZE, ACTION_SIZE)
    if algo_name == "LearningToAct":
        return LearningToActAgent(STATE_SIZE, ACTION_SIZE)
    raise ValueError(f"Unknown algorithm: {algo_name}")


def _load_activity_name_to_id(activity_info_path: str = os.path.join("data", "raw", "activity_info.json")):
    if not os.path.exists(activity_info_path):
        activity_info_path = os.path.join("..", activity_info_path)
    with open(activity_info_path, "r", encoding="utf-8") as f:
        activity_info = json.load(f)
    # In this repo, ids appear to be 1..21; map to 0..20
    name_to_id = {name: int(info["id"]) - 1 for name, info in activity_info.items()}
    return name_to_id


def _build_lta_episode_from_event_log(df_case, name_to_id):
    """Build an episode as list of (state_key, action_id, reward).

    State_key = prefix bitmask BEFORE executing the action.
    Reward = -duration(action) using START/COMPLETE timestamps.
    """
    # Keep only START/COMPLETE rows for safety
    df_case = df_case.sort_values("Timestamp")

    # Pair START->COMPLETE per activity in order.
    start_time = {}
    episode = []
    prefix_mask = 0

    for _, row in df_case.iterrows():
        act = row["Activity"]
        lifecycle = row["Lifecycle"]
        ts = float(row["Timestamp"])
        if lifecycle == "START":
            # If START already exists for same activity, overwrite (shouldn't happen often).
            start_time[act] = ts
        elif lifecycle == "COMPLETE":
            if act not in start_time:
                continue
            dur = max(0.0, ts - float(start_time[act]))
            a = name_to_id.get(act)
            if a is None:
                continue
            s_key = int(prefix_mask)
            r = -dur
            episode.append((s_key, int(a), float(r)))
            # update prefix after completion
            prefix_mask |= 1 << int(a)
            # cleanup
            del start_time[act]

    return episode


def _mc_policy_from_event_log(
    event_log_path: str,
    gamma: float = 0.99,
    min_visits: int = 1,
):
    """Monte Carlo policy estimation from historical event log.

    Returns:
      policy: dict[state_key] = best_action_id
    """
    import pandas as pd

    if not os.path.exists(event_log_path):
        raise FileNotFoundError(f"Event log not found: {event_log_path}")

    df = pd.read_csv(event_log_path)
    required_cols = {"CaseID", "Activity", "Timestamp", "Lifecycle"}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(f"Event log missing columns. Need {required_cols}, got {set(df.columns)}")

    name_to_id = _load_activity_name_to_id()

    # Q(s,a) via sums/counts
    q_sum = {}
    q_cnt = {}

    for case_id, df_case in df.groupby("CaseID"):
        episode = _build_lta_episode_from_event_log(df_case, name_to_id)
        if not episode:
            continue

        # Backward return
        G = 0.0
        for (s_key, a, r) in reversed(episode):
            G = r + gamma * G
            key = (int(s_key), int(a))
            q_sum[key] = q_sum.get(key, 0.0) + float(G)
            q_cnt[key] = q_cnt.get(key, 0) + 1

    # Derive greedy policy
    best_by_state = {}
    for (s_key, a), s in q_sum.items():
        c = q_cnt[(s_key, a)]
        if c < min_visits:
            continue
        q = s / c
        if s_key not in best_by_state or q > best_by_state[s_key][0]:
            best_by_state[s_key] = (q, a)

    policy = {int(s_key): int(qa[1]) for s_key, qa in best_by_state.items()}
    return policy


def run_learning_to_act_once():
    """Entry point for: python main.py LearningToAct

    Offline-only: learn policy from baseline event log (random) via Monte Carlo,
    then run simulation evaluation.
    """
    algo_name = "LearningToAct"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, "\n=== Train model: LearningToAct (offline MC policy) ===\n")

    # 1) Ensure baseline random queue log exists (used by FillBlanksEnv in other parts)
    random_queue_log = os.path.join("data", "raw", "200_queue_log_version_random_base.csv")
    if not os.path.exists(random_queue_log):
        _append_note(log_dir, "Random queue log missing; generating via simulation.\n")
        run_simulation_isolated(num_patients=200, agent=None, version_output="random_base", seed=EVAL_SEED)

    # 2) Ensure baseline event log exists
    event_log_path = os.path.join("data", "evaluate", "event_log_version_random_base.csv")
    if not os.path.exists(event_log_path):
        _append_note(log_dir, "Baseline event log missing; generating via simulation.\n")
        run_simulation_isolated(num_patients=200, agent=None, version_output="random_base", seed=EVAL_SEED)

    # 3) Offline MC policy learning
    policy = _mc_policy_from_event_log(event_log_path, gamma=0.99, min_visits=1)
    _append_note(log_dir, f"Learned policy states: {len(policy)}\n")

    policy_path = os.path.join(log_dir, "policy.pkl")
    agent = get_agent("LearningToAct")
    agent.policy = policy
    agent.save(policy_path)
    _append_note(log_dir, f"Saved policy: {policy_path}\n")

    # 4) Evaluate by simulation
    baseline_avg_time = run_simulation_isolated(
        num_patients=200,
        agent=None,
        version_output="random_base",
        seed=EVAL_SEED,
    )
    avg_time = run_simulation_isolated(
        num_patients=200,
        agent=agent,
        version_output="learningtoact",
        is_model_run=False,
        seed=EVAL_SEED,
    )
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Baseline Avg Time: {baseline_avg_time:.2f}\n")
    _append_note(log_dir, f"LearningToAct Avg Time: {avg_time:.2f} | Improvement: {improvement:+.2f}%\n")
    print(f"LearningToAct Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")

def train_one_generation(agent, algo_name: str, gen_id: int, train_config):
    """Train một thế hệ và trả về danh sách checkpoint paths được tạo ra."""
    config = train_config[gen_id]
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


SUPPORTED_ALGOS = ["DQN", "DDQN", "PerDQN", "Dueling", "Rainbow", "MultiStepDQN", "FORLAPS", "LearningToAct"]

def _append_note(log_dir: str, text: str) -> None:
    """Append training/evaluation notes to a file under log_dir."""
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, "training_notes.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)

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

    # Special-case FORLAPS: offline only, no training generations.
    if ALGO_TO_RUN == "FORLAPS":
        run_forlaps_once_forlaps(train_seed=TRAIN_SEED, eval_seed=EVAL_SEED)
        sys.exit(0)

    # Special-case LearningToAct: offline only, no training generations.
    if ALGO_TO_RUN == "LearningToAct":
        run_learning_to_act_once()
        sys.exit(0)

    agent = get_agent(ALGO_TO_RUN)

    print(f"--- Starting Full Training Cycle for [{ALGO_TO_RUN}] ---")

    log_dir = os.path.join("logs", ALGO_TO_RUN)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {ALGO_TO_RUN} ===\n")

    TEST_SEED = 42
    baseline_avg_time = run_simulation_isolated(num_patients=200, agent=None, version_output="random_base", seed=TEST_SEED)

    prev_best_avg_time = None

    train_config = get_train_config(ALGO_TO_RUN)

    for gen_id in range(1, 6):
        if gen_id not in train_config:
            break

        ckpt_paths = train_one_generation(agent, ALGO_TO_RUN, gen_id, train_config)

        best, _ = _evaluate_checkpoints(
            algo_name=ALGO_TO_RUN,
            gen_id=gen_id,
            log_dir=log_dir,
            seed=TEST_SEED,
            baseline_avg_time=baseline_avg_time,
        )

        best_ep, best_ckpt_path, best_avg_time = best
        final_name = train_config[gen_id]["save_name"]
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

        if gen_id < 5 and (gen_id + 1) in train_config:
            agent_for_data = get_agent(ALGO_TO_RUN)
            agent_for_data.load(final_path)
            run_simulation_isolated(num_patients=200, agent=agent_for_data, version_output=str(gen_id), is_model_run=True, seed=TEST_SEED)

    print("\n🎉🎉🎉 Training cycle finished! 🎉🎉🎉")