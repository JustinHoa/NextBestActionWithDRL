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
from agents.penalty_dqn_agent import PenaltyDQNAgent
from agents.forlaps import run_forlaps_once as run_forlaps_once_forlaps
from common.env import FillBlanksEnv
from common.penalty_env import PenaltyEnv
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

def run_simulation_isolated(num_patients, agent, version_output, is_model_run=False, seed=None, model_name="", gen_id=0):
    """Run simulation with isolated RNG state to prevent training interference."""
    # Save current training RNG state
    training_state = save_rng_state()
    
    try:
        # Run simulation with its own seed
        result = run_simulation(num_patients, agent, version_output, is_model_run, seed, model_name, gen_id)
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
    if algo_name == "PenaltyDQN":
        return PenaltyDQNAgent(STATE_SIZE, ACTION_SIZE)
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


def run_learning_to_act_once(num_patients: int = 200):
    """Entry point for: python main.py LearningToAct

    Offline-only: learn policy from baseline event log (random) via Monte Carlo,
    then run simulation evaluation.
    """
    algo_name = "LearningToAct"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, "\n=== Train model: LearningToAct ===\n", num_patients)

    # 1) Ensure baseline random queue log exists (used by FillBlanksEnv in other parts)
    random_queue_log = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    if not os.path.exists(random_queue_log):
        _append_note(log_dir, "Random queue log missing; generating via simulation.\n", num_patients)
        run_simulation_isolated(num_patients=num_patients, agent=None, version_output="random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)

    # 2) Ensure baseline event log exists
    event_log_path = os.path.join("data", "evaluate", f"event_log_{num_patients}_random_base.csv")
    if not os.path.exists(event_log_path):
        _append_note(log_dir, "Baseline event log missing; generating via simulation.\n", num_patients)
        run_simulation_isolated(num_patients=num_patients, agent=None, version_output="random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)

    # 3) Offline MC policy learning
    t0 = time.time()
    policy = _mc_policy_from_event_log(event_log_path, gamma=0.99, min_visits=1)
    train_seconds = time.time() - t0
    train_minutes = train_seconds / 60.0
    
    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    agent = get_agent("LearningToAct")
    agent.policy = policy
    agent.save(model_path)

    # 4) Evaluate by simulation
    baseline_avg_time = run_simulation_isolated(
        num_patients=num_patients,
        agent=None,
        version_output="random_base",
        seed=EVAL_SEED,
        model_name="Random",
        gen_id=0,
    )
    
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients)
    _append_note(log_dir, "====\n", num_patients)
    _append_note(log_dir, f"Episode: {len(policy)} states\n", num_patients)
    _append_note(log_dir, f"Training Time: {train_minutes:.2f} minutes\n", num_patients)
    
    avg_time = run_simulation_isolated(
        num_patients=num_patients,
        agent=agent,
        version_output="learningtoact",
        is_model_run=True,
        seed=EVAL_SEED,
        model_name="LearningToAct",
        gen_id=0,
    )
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"LearningToAct Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")

def train_penalty_dqn(agent, num_patients: int):
    """Train PenaltyDQN với penalty-based reward system."""
    algo_name = "PenaltyDQN"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    
    train_config = get_train_config(algo_name, num_patients)
    config = train_config[1]
    
    smoke_episodes_raw = os.environ.get("SMOKE_EPISODES")
    smoke_episodes = int(smoke_episodes_raw) if smoke_episodes_raw else None
    # Default: train full episodes. Allow optional override for debugging.
    episodes_to_run = config["episodes"] if smoke_episodes is None else min(config["episodes"], smoke_episodes)
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING: [PenaltyDQN]")
    print(f"   Description: {config['description']}")
    print(f"   Episodes: {episodes_to_run}")
    print(f"   Data: {config['data_file']}")
    print(f"{'='*60}\n")
    
    env = PenaltyEnv(STATE_SIZE, ACTION_SIZE, data_path=config["data_file"])
    
    scores_window = deque(maxlen=100)
    all_scores, all_losses = [], []
    eps = config["eps_start"]
    
    ckpt_paths = []
    t0 = time.time()
    
    tqdm_bar = tqdm(range(1, episodes_to_run + 1), desc=f"Training PenaltyDQN")
    for i_episode in tqdm_bar:
        state = env.reset()
        score = 0
        for _ in range(30):
            # PenaltyDQN không dùng mask
            action = agent.act(state, mask=None, eps=eps)
            next_state, reward, done = env.step(action)
            
            loss = agent.step(state, action, reward, next_state, done, next_mask=None)
            if loss is not None:
                all_losses.append(loss)
            
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        all_scores.append(score)
        eps = max(config["eps_end"], eps * config["eps_decay"])
        
        if i_episode % 100 == 0:
            tqdm_bar.set_postfix(avg_score=f"{np.mean(scores_window):.2f}", eps=f"{eps:.3f}")
        
        # Save checkpoints every 10k episodes
        if i_episode % 10000 == 0:
            ckpt_name = f"checkpoint_{num_patients}_gen_1_{i_episode}.pth"
            ckpt_path = os.path.join(log_dir, ckpt_name)
            agent.save(ckpt_path)
            ckpt_paths.append(ckpt_path)
            tqdm_bar.write(f"💾 Checkpoint saved: {ckpt_name}")
    
    train_seconds = time.time() - t0
    train_minutes = train_seconds / 60.0
    
    plot_training_status(all_scores, all_losses, algo_name, 1, save_dir=log_dir, num_patients=num_patients)
    
    _append_note(
        log_dir,
        f"Episode: {config['episodes']} episode\n"
        f"Training Time: {train_minutes:.2f} minutes\n",
        num_patients=num_patients,
    )
    
    return ckpt_paths


def _validate_penalty_checkpoint(ckpt_path: str, num_patients: int):
    import pandas as pd

    possible_states_path = os.path.join("data", "raw", "possible_states.csv")
    df_states = pd.read_csv(possible_states_path, header=None)
    possible_states = df_states.values

    queue_log_path = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    df_queue = pd.read_csv(queue_log_path)
    if df_queue.shape[1] == 22:
        queue_trace = df_queue.iloc[:, 1:].values
    else:
        queue_trace = df_queue.values

    env = PenaltyEnv(STATE_SIZE, ACTION_SIZE, data_path=queue_log_path)

    agent_eval = get_agent("PenaltyDQN")
    agent_eval.load(ckpt_path)

    total = int(len(possible_states))
    invalid = 0

    for row in possible_states:
        gender = int(row[0])
        marital = int(row[1])
        prefix = row[2:23].astype(np.float32)

        env.features = np.array([gender, marital], dtype=np.int64)
        env.blanks = prefix.copy()
        env.done = False
        env.total_time = 0.0
        env.start_time_idx = 0
        env.goal_mask = env._create_goal_mask()

        q_idx = np.random.randint(0, env.max_trace_len)
        env.queues = queue_trace[q_idx]

        state = env._get_state()
        action = agent_eval.act(state, mask=None, eps=0.0)
        _, reward, _ = env.step(int(action))
        if reward < 0:
            invalid += 1

    valid = total - invalid
    valid_pct = (valid / total) * 100.0 if total else 0.0
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_pct": valid_pct,
    }


def train_one_generation(agent, algo_name: str, gen_id: int, train_config, num_patients: int):
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
            ckpt_name = f"checkpoint_{num_patients}_gen_{gen_id}_{i_episode}.pth"
            ckpt_path = os.path.join(log_dir, ckpt_name)
            agent.save(ckpt_path)
            ckpt_paths.append(ckpt_path)
            tqdm_bar.write(f"💾 Checkpoint saved: {ckpt_name}")

    train_seconds = time.time() - t0
    train_minutes = train_seconds / 60.0

    plot_training_status(all_scores, all_losses, algo_name, gen_id, save_dir=log_dir, num_patients=num_patients)

    _append_note(
        log_dir,
        f"Episode: {config['episodes']} episode\n"
        f"Training Time: {train_minutes:.2f} minutes\n",
        num_patients=num_patients,
    )

    return ckpt_paths


def _parse_checkpoint_episode(path: str) -> int:
    """Parse episode number from checkpoint filename."""
    import re
    m = re.search(r"_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _evaluate_checkpoints(algo_name: str, gen_id: int, log_dir: str, seed: int, baseline_avg_time: float, num_patients: int):
    """Evaluate all checkpoints of a generation via simulation and select the best (min avg time)."""
    ckpt_glob = os.path.join(log_dir, f"checkpoint_{num_patients}_gen_{gen_id}_*.pth")
    ckpt_paths = sorted(glob.glob(ckpt_glob), key=_parse_checkpoint_episode)
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found for Gen {gen_id} at {ckpt_glob}")

    results = []
    best = None

    for ckpt_path in ckpt_paths:
        ep = _parse_checkpoint_episode(ckpt_path)
        agent_eval = get_agent(algo_name)
        agent_eval.load(ckpt_path)

        avg_time = run_simulation_isolated(
            num_patients=num_patients,
            agent=agent_eval,
            version_output=f"{algo_name.lower()}_g{gen_id}_ep{ep}",
            is_model_run=False,
            seed=seed,
            model_name=algo_name,
            gen_id=gen_id,
        )

        improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time else 0.0
        results.append((ep, ckpt_path, avg_time, improvement))
        if best is None or avg_time < best[2]:
            best = (ep, ckpt_path, avg_time)

    _append_note(log_dir, f"Gen {gen_id}: evaluated {len(results)} checkpoints\n", num_patients=num_patients)
    if best is not None:
        _append_note(log_dir, f"Gen {gen_id} best checkpoint: ep={best[0]} | Avg Time: {best[2]:.2f}\n", num_patients=num_patients)

    return best, results


SUPPORTED_ALGOS = ["DQN", "DDQN", "PerDQN", "Dueling", "Rainbow", "MultiStepDQN", "FORLAPS", "LearningToAct", "PenaltyDQN"]


def _append_note(log_dir: str, text: str, num_patients: int = 200) -> None:
    """Append training/evaluation notes to a file under log_dir."""
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, f"training_notes_{num_patients}.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    # --- 1. Lấy tham số từ command line ---
    if len(sys.argv) < 2:
        print("❌ Bạn chưa truyền thuật toán.")
        print("   Ví dụ cách chạy:")
        print("   python main.py DQN 200")
        print("   python main.py DDQN 200")
        print("   python main.py Dueling 200")
        print("   python main.py Rainbow 200")
        print("   python main.py MultiStepDQN 200")
        sys.exit(1)

    ALGO_TO_RUN = sys.argv[1].strip()
    NUM_PATIENTS = int(sys.argv[2].strip()) if len(sys.argv) >= 3 else 200

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
        run_forlaps_once_forlaps(train_seed=TRAIN_SEED, eval_seed=EVAL_SEED, num_patients=NUM_PATIENTS)
        sys.exit(0)

    # Special-case LearningToAct: offline only, no training generations.
    if ALGO_TO_RUN == "LearningToAct":
        run_learning_to_act_once(num_patients=NUM_PATIENTS)
        sys.exit(0)
    
    # Special-case PenaltyDQN: single generation with penalty-based training
    if ALGO_TO_RUN == "PenaltyDQN":
        agent = get_agent(ALGO_TO_RUN)
        log_dir = os.path.join("logs", ALGO_TO_RUN)
        ensure_dir(log_dir)
        _append_note(log_dir, f"\n=== Train model: {ALGO_TO_RUN} ===\n", num_patients=NUM_PATIENTS)
        
        # Get baseline
        random_base_queue_log = os.path.join("data", "raw", f"queue_log_{NUM_PATIENTS}_random_base.csv")
        if os.path.exists(random_base_queue_log):
            print(f"✅ Found existing random_base queue log: {random_base_queue_log}")
            print("   Skipping random_base simulation...")
            # Still need to run once to get baseline_avg_time for comparison
            baseline_avg_time = run_simulation_isolated(num_patients=NUM_PATIENTS, agent=None, version_output="random_base_eval", seed=EVAL_SEED, model_name="Random", gen_id=0)
        else:
            baseline_avg_time = run_simulation_isolated(num_patients=NUM_PATIENTS, agent=None, version_output="random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)

        # Log baseline avg time
        _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients=NUM_PATIENTS)
        _append_note(log_dir, "====\n", num_patients=NUM_PATIENTS)
        
        # Train
        ckpt_paths = train_penalty_dqn(agent, NUM_PATIENTS)

        if not ckpt_paths:
            _append_note(log_dir, "No checkpoints were produced in this run (episodes < checkpoint interval).\n", num_patients=NUM_PATIENTS)
            print("✅ PenaltyDQN Training Complete!")
            print("   No checkpoints produced; skipping checkpoint validation/simulation.")
            sys.exit(0)

        _append_note(log_dir, "\n====\nCHECKPOINT VALIDATION (possible_states.csv)\n", num_patients=NUM_PATIENTS)

        best_valid_ckpt = None
        best_valid_avg_time = None
        best_valid_ep = None

        for ckpt_path in ckpt_paths:
            ep = _parse_checkpoint_episode(ckpt_path)
            stats = _validate_penalty_checkpoint(ckpt_path, NUM_PATIENTS)
            _append_note(
                log_dir,
                f"Checkpoint: {os.path.basename(ckpt_path)} | valid={stats['valid']}/{stats['total']} ({stats['valid_pct']:.2f}%) | invalid={stats['invalid']}\n",
                num_patients=NUM_PATIENTS,
            )

            if stats["invalid"] != 0:
                continue

            avg_time = run_simulation_isolated(
                num_patients=NUM_PATIENTS,
                agent=get_agent(ALGO_TO_RUN),
                version_output=f"penaltydqn_valid_ep{ep}",
                is_model_run=False,
                seed=EVAL_SEED,
                model_name=ALGO_TO_RUN,
                gen_id=1,
            )

            _append_note(
                log_dir,
                f"  Simulation: Avg Time={avg_time:.2f} | Improvement={((baseline_avg_time - avg_time) / baseline_avg_time) * 100:+.2f}%\n",
                num_patients=NUM_PATIENTS,
            )

            if best_valid_avg_time is None or avg_time < best_valid_avg_time:
                best_valid_avg_time = avg_time
                best_valid_ckpt = ckpt_path
                best_valid_ep = ep

        if best_valid_ckpt is None:
            _append_note(log_dir, "No checkpoint reached 100% valid actions; no simulation winner selected.\n", num_patients=NUM_PATIENTS)
            print("✅ PenaltyDQN Training Complete!")
            print("   No checkpoint achieved 100% valid actions.")
            sys.exit(0)

        final_name = f"final_{NUM_PATIENTS}_gen_1.pth"
        final_path = os.path.join(log_dir, final_name)
        shutil.copyfile(best_valid_ckpt, final_path)

        improvement_pct = ((baseline_avg_time - best_valid_avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
        _append_note(
            log_dir,
            f"Final Model: {final_name} (from ep={best_valid_ep}) | Avg Time: {best_valid_avg_time:.2f} | Improve Percentage: {improvement_pct:+.2f}%\n",
            num_patients=NUM_PATIENTS,
        )

        print(f"✅ PenaltyDQN Training Complete!")
        print(f"   Best VALID checkpoint ep={best_valid_ep} | Avg Time: {best_valid_avg_time:.2f} | Improvement: {improvement_pct:+.2f}%")
        sys.exit(0)

    agent = get_agent(ALGO_TO_RUN)

    print(f"--- Starting Full Training Cycle for [{ALGO_TO_RUN}] with {NUM_PATIENTS} patients ---")

    log_dir = os.path.join("logs", ALGO_TO_RUN)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {ALGO_TO_RUN} ===\n", num_patients=NUM_PATIENTS)

    TEST_SEED = 42
    
    # Check if random_base queue log already exists
    random_base_queue_log = os.path.join("data", "raw", f"queue_log_{NUM_PATIENTS}_random_base.csv")
    if os.path.exists(random_base_queue_log):
        print(f"✅ Found existing random_base queue log: {random_base_queue_log}")
        print("   Skipping random_base simulation...")
        # Still need to run once to get baseline_avg_time for comparison
        baseline_avg_time = run_simulation_isolated(num_patients=NUM_PATIENTS, agent=None, version_output="random_base_eval", seed=TEST_SEED, model_name="Random", gen_id=0)
    else:
        print(f"⚠️ Random_base queue log not found. Generating...")
        baseline_avg_time = run_simulation_isolated(num_patients=NUM_PATIENTS, agent=None, version_output="random_base", seed=TEST_SEED, model_name="Random", gen_id=0)

    # Log baseline avg time
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients=NUM_PATIENTS)
    
    prev_best_avg_time = None

    train_config = get_train_config(ALGO_TO_RUN, NUM_PATIENTS)

    for gen_id in range(1, 6):
        if gen_id not in train_config:
            break

        # Add separator before each generation
        _append_note(log_dir, "====\n", num_patients=NUM_PATIENTS)
        
        ckpt_paths = train_one_generation(agent, ALGO_TO_RUN, gen_id, train_config, NUM_PATIENTS)

        best, _ = _evaluate_checkpoints(
            algo_name=ALGO_TO_RUN,
            gen_id=gen_id,
            log_dir=log_dir,
            seed=TEST_SEED,
            baseline_avg_time=baseline_avg_time,
            num_patients=NUM_PATIENTS,
        )

        best_ep, best_ckpt_path, best_avg_time = best
        final_name = f"final_{NUM_PATIENTS}_gen_{gen_id}.pth"
        final_path = os.path.join(log_dir, final_name)
        shutil.copyfile(best_ckpt_path, final_path)
        
        # Calculate improvement percentage
        improvement_pct = ((baseline_avg_time - best_avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
        _append_note(log_dir, f"Final Model: {final_name} (from ep={best_ep}) | Avg Time: {best_avg_time:.2f} | Improve Percentage: {improvement_pct:+.2f}%\n", num_patients=NUM_PATIENTS)

        if prev_best_avg_time is not None and best_avg_time >= prev_best_avg_time:
            _append_note(
                log_dir,
                f"STOP: Gen {gen_id} Avg Time ({best_avg_time:.2f}) >= Gen {gen_id - 1} Avg Time ({prev_best_avg_time:.2f})\n",
                num_patients=NUM_PATIENTS,
            )
            print(f"🛑 STOP: Gen {gen_id} did not improve over Gen {gen_id - 1}.")
            break

        prev_best_avg_time = best_avg_time

        if gen_id < 5 and (gen_id + 1) in train_config:
            agent_for_data = get_agent(ALGO_TO_RUN)
            agent_for_data.load(final_path)
            run_simulation_isolated(num_patients=NUM_PATIENTS, agent=agent_for_data, version_output=str(gen_id), is_model_run=True, seed=TEST_SEED, model_name=ALGO_TO_RUN, gen_id=gen_id)

    print("\n🎉🎉🎉 Training cycle finished! 🎉🎉🎉")