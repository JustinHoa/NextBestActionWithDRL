"""
Main script for Unlimited Queue training and evaluation.
Supports: DQN, DDQN, Dueling, PerDQN, Rainbow, MultiStepDQN, FORLAPS, LearningToAct, FCFS, Greedy
"""
import os
from collections import deque
import sys
import time
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
from agents.next_best_action_agent import NextBestActionAgent
from agents.fcfs_agent import FCFSAgent
from agents.greedy_agent import GreedyAgent
from agents.forlaps import run_forlaps_once as run_forlaps_once_forlaps
from agents.tabular_ql_agent import TabularQLAgent
from agents.tabular_sarsa_agent import TabularSARSAAgent
from agents.soliman_dqn_agent import SolimanDQNAgent
from common.env import FillBlanksEnv
from common.utils import (
    ACTION_SIZE,
    STATE_SIZE,
    DEVICE,
    get_train_config,
    ensure_dir,
    plot_training_status,
)
from simulation.simulation_process import run_simulation

# === REPRODUCIBILITY SETTINGS ===
TRAIN_SEED = 1230000
EVAL_SEED = 420000

def set_deterministic_mode():
    """Set all random number generators to deterministic mode."""
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Deterministic mode enabled with TRAIN_SEED={TRAIN_SEED}")

def save_rng_state():
    """Save current RNG states."""
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def restore_rng_state(state_dict):
    """Restore RNG states."""
    random.setstate(state_dict['random'])
    np.random.set_state(state_dict['numpy'])
    torch.set_rng_state(state_dict['torch'])
    if torch.cuda.is_available() and state_dict['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state_dict['torch_cuda'])

def run_simulation_isolated(num_patients, agent, version_output, is_model_run=False, seed=None, model_name="", gen_id=0):
    """Run simulation with isolated RNG state."""
    training_state = save_rng_state()
    try:
        result = run_simulation(num_patients, agent, version_output, is_model_run, seed, model_name, gen_id)
        return result
    finally:
        restore_rng_state(training_state)

def get_agent(algo_name: str, seed: int = 0):
    """Factory function for agents."""
    if algo_name == "DQN":
        return DQNAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "DDQN":
        return DDQNAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "Dueling":
        return DuelingAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "PerDQN":
        return PerDqnAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "MultiStepDQN":
        return MultiStepDqnAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "Rainbow":
        return RainbowAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "FORLAPS":
        return DQNAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "LearningToAct":
        return LearningToActAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "FCFS":
        return FCFSAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "Greedy":
        return GreedyAgent(STATE_SIZE, ACTION_SIZE, seed=seed)
    if algo_name == "HundoganQL":
        return TabularQLAgent(STATE_SIZE, ACTION_SIZE, alpha=0.2, gamma=0.2, seed=seed)
    if algo_name == "HundoganSARSA":
        return TabularSARSAAgent(STATE_SIZE, ACTION_SIZE, alpha=0.2, gamma=0.2, seed=seed)
    if algo_name == "SolimanQL":
        return TabularQLAgent(STATE_SIZE, ACTION_SIZE, alpha=0.2, gamma=0.99, seed=seed)
    if algo_name == "SolimanDQN":
        return SolimanDQNAgent(action_size=ACTION_SIZE, seed=seed)
    raise ValueError(f"Unknown algorithm: {algo_name}")

def _append_note(log_dir: str, text: str, num_patients: int = 200) -> None:
    """Append training/evaluation notes."""
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, f"training_notes_{num_patients}.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)

def _load_activity_name_to_id(activity_info_path: str = os.path.join("data", "raw", "activity_info.json")):
    if not os.path.exists(activity_info_path):
        activity_info_path = os.path.join("..", activity_info_path)
    with open(activity_info_path, "r", encoding="utf-8") as f:
        activity_info = json.load(f)
    name_to_id = {name: int(info["id"]) - 1 for name, info in activity_info.items()}
    return name_to_id

def _build_lta_episode_from_event_log(df_case, name_to_id):
    """Build episode from event log for LearningToAct."""
    df_case = df_case.sort_values("Timestamp")
    start_time = {}
    episode = []
    prefix_mask = 0

    for _, row in df_case.iterrows():
        act = row["Activity"]
        lifecycle = row["Lifecycle"]
        ts = float(row["Timestamp"])
        if lifecycle == "START":
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
            prefix_mask |= 1 << int(a)
            del start_time[act]

    return episode

def _mc_policy_from_event_log(event_log_path: str, gamma: float = 0.99, min_visits: int = 1):
    """Monte Carlo policy estimation from event log."""
    import pandas as pd

    if not os.path.exists(event_log_path):
        raise FileNotFoundError(f"Event log not found: {event_log_path}")

    df = pd.read_csv(event_log_path)
    required_cols = {"CaseID", "Activity", "Timestamp", "Lifecycle"}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(f"Event log missing columns. Need {required_cols}, got {set(df.columns)}")

    name_to_id = _load_activity_name_to_id()

    q_sum = {}
    q_cnt = {}

    for case_id, df_case in df.groupby("CaseID"):
        episode = _build_lta_episode_from_event_log(df_case, name_to_id)
        if not episode:
            continue

        G = 0.0
        for (s_key, a, r) in reversed(episode):
            G = r + gamma * G
            key = (int(s_key), int(a))
            q_sum[key] = q_sum.get(key, 0.0) + float(G)
            q_cnt[key] = q_cnt.get(key, 0) + 1

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
    """Entry point for LearningToAct."""
    algo_name = "LearningToAct"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, "\n=== Train model: LearningToAct ===\n", num_patients)

    random_queue_log = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    if not os.path.exists(random_queue_log):
        _append_note(log_dir, "Random queue log missing; generating via simulation.\n", num_patients)
        run_simulation_isolated(num_patients=num_patients, agent=None, version_output="random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)

    event_log_path = os.path.join("data", "evaluate", f"event_log_{num_patients}_random_base.csv")
    if not os.path.exists(event_log_path):
        _append_note(log_dir, "Baseline event log missing; generating via simulation.\n", num_patients)
        run_simulation_isolated(num_patients=num_patients, agent=None, version_output="random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)

    t0 = time.time()
    policy = _mc_policy_from_event_log(event_log_path, gamma=0.99, min_visits=1)
    train_seconds = time.time() - t0
    train_minutes = train_seconds / 60.0
    
    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    agent = get_agent("LearningToAct")
    agent.policy = policy
    agent.save(model_path)

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
    print(f"✅ LearningToAct Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")

def run_fcfs_once(num_patients: int = 200):
    """Entry point for FCFS."""
    algo_name = "FCFS"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, "\n=== Baseline: FCFS ===\n", num_patients)
    
    agent = get_agent("FCFS")
    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    agent.save(model_path)
    
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
    
    avg_time = run_simulation_isolated(
        num_patients=num_patients,
        agent=agent,
        version_output="fcfs",
        is_model_run=True,
        seed=EVAL_SEED,
        model_name="FCFS",
        gen_id=0,
    )
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"✅ FCFS Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")

def run_greedy_once(num_patients: int = 200):
    """Entry point for Greedy."""
    algo_name = "Greedy"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, "\n=== Baseline: Greedy ===\n", num_patients)
    
    agent = get_agent("Greedy")
    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    agent.save(model_path)
    
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
    
    avg_time = run_simulation_isolated(
        num_patients=num_patients,
        agent=agent,
        version_output="greedy",
        is_model_run=True,
        seed=EVAL_SEED,
        model_name="Greedy",
        gen_id=0,
    )
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"✅ Greedy Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")

def _build_td_transitions_from_event_log(event_log_path: str):
    """Build (s_key, a, r, s_next_key, a_next, done, mask_next) tuples from event log.

    Used for offline tabular RL training (Hundogan et al., 2025).
    s_key  : 21-bit bitmask of completed activities BEFORE the action.
    r      : negative activity duration (time-cost proxy).
    mask_next: simplified — bit i clear in s_next_key means activity i is still valid.
    a_next : actual next logged action (on-policy signal for SARSA).
    """
    import pandas as pd

    df = pd.read_csv(event_log_path)
    name_to_id = _load_activity_name_to_id()
    all_transitions = []

    for _, df_case in df.groupby("CaseID"):
        df_case = df_case.sort_values("Timestamp")
        start_time_map: dict = {}
        episode = []
        prefix_mask = 0

        for _, row in df_case.iterrows():
            act = row["Activity"]
            lifecycle = row["Lifecycle"]
            ts = float(row["Timestamp"])
            if lifecycle == "START":
                start_time_map[act] = ts
            elif lifecycle == "COMPLETE":
                if act not in start_time_map:
                    continue
                dur = max(0.0, ts - float(start_time_map[act]))
                a = name_to_id.get(act)
                if a is None:
                    continue
                episode.append((int(prefix_mask), int(a), -dur))
                prefix_mask |= 1 << int(a)
                del start_time_map[act]

        for i, (s_key, a, r) in enumerate(episode):
            s_next_key = s_key | (1 << a)
            done = (i == len(episode) - 1)
            mask_next = np.array(
                [0.0 if (s_next_key >> j) & 1 else 1.0 for j in range(ACTION_SIZE)],
                dtype=np.float32,
            )
            a_next = episode[i + 1][1] if not done else 0
            all_transitions.append((s_key, a, r, s_next_key, a_next, done, mask_next))

    return all_transitions


def _run_hundogan_tabular_once(algo_name: str, num_patients: int = 200,
                                n_agents: int = 3, n_epochs: int = 20):
    """Train tabular RL agent (QL or SARSA) offline from random-baseline event log.

    Replicates the Hundogan et al. (2025) protocol:
    - Train n_agents with fixed hyperparameters (alpha=0.2, gamma=0.2).
    - Select best agent by validation simulation (stability criterion).
    """
    assert algo_name in ("HundoganQL", "HundoganSARSA")

    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {algo_name} ===\n", num_patients)

    event_log_path = os.path.join("data", "evaluate", f"event_log_{num_patients}_random_base.csv")
    random_queue_log = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    if not os.path.exists(event_log_path) or not os.path.exists(random_queue_log):
        _append_note(log_dir, "Baseline logs missing; generating via simulation.\n", num_patients)
        run_simulation_isolated(
            num_patients=num_patients, agent=None, version_output="random_base",
            seed=EVAL_SEED, model_name="Random", gen_id=0,
        )

    print(f"📊 Building TD transitions from: {event_log_path}")
    transitions = _build_td_transitions_from_event_log(event_log_path)
    print(f"   Total transitions: {len(transitions)}")

    baseline_avg_time = run_simulation_isolated(
        num_patients=num_patients, agent=None, version_output="random_base",
        seed=EVAL_SEED, model_name="Random", gen_id=0,
    )
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients)
    _append_note(log_dir, "====\n", num_patients)

    t0 = time.time()
    best_agent = None
    best_val_time = None

    for agent_idx in range(n_agents):
        seed = TRAIN_SEED + agent_idx
        agent = get_agent(algo_name, seed=seed)
        np.random.seed(seed)

        print(f"\n🔄 Training candidate {agent_idx + 1}/{n_agents} ({algo_name}, seed={seed})...")
        shuffled = list(transitions)
        for epoch in range(n_epochs):
            np.random.shuffle(shuffled)
            for (s_key, a, r, s_next_key, a_next, done, mask_next) in shuffled:
                if algo_name == "HundoganQL":
                    agent.update(s_key, a, r, s_next_key, mask_next, done)
                else:
                    agent.update(s_key, a, r, s_next_key, a_next, done)

            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch + 1}/{n_epochs} | Q-table size: {len(agent.q_table)}")

        val_time = run_simulation_isolated(
            num_patients=num_patients, agent=agent,
            version_output=f"{algo_name.lower()}_val{agent_idx}",
            is_model_run=False, seed=EVAL_SEED,
            model_name=algo_name, gen_id=0,
        )
        print(f"   Candidate {agent_idx + 1} validation avg time: {val_time:.2f} mins")
        _append_note(log_dir, f"Candidate {agent_idx + 1}: val_time={val_time:.2f}\n", num_patients)

        if best_val_time is None or val_time < best_val_time:
            best_val_time = val_time
            best_agent = agent

    train_minutes = (time.time() - t0) / 60.0

    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    best_agent.save(model_path)
    _append_note(log_dir, f"Training Time: {train_minutes:.2f} minutes\n", num_patients)
    _append_note(log_dir, f"Q-table size (best): {len(best_agent.q_table)} states\n", num_patients)

    avg_time = run_simulation_isolated(
        num_patients=num_patients, agent=best_agent,
        version_output=algo_name.lower(),
        is_model_run=True, seed=EVAL_SEED,
        model_name=algo_name, gen_id=0,
    )
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"✅ {algo_name} Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")
    print(f"   Q-table states seen: {len(best_agent.q_table)}")


def run_hundogan_ql_once(num_patients: int = 200):
    """Entry point for HundoganQL (tabular Q-learning)."""
    _run_hundogan_tabular_once("HundoganQL", num_patients)


# ============================================================
# Soliman et al. (2025) — transition-probability reward
# ============================================================

def _compute_transition_probs_from_event_log(event_log_path: str) -> dict:
    """Compute P(action | bitmask_state) from event log.

    Returns {s_key (int): {action (int): probability (float)}}.
    Used as the reward signal for both SolimanQL and SolimanDQN.
    """
    import pandas as pd

    df = pd.read_csv(event_log_path)
    name_to_id = _load_activity_name_to_id()
    counts: dict = {}

    for _, df_case in df.groupby("CaseID"):
        df_case = df_case.sort_values("Timestamp")
        start_map: dict = {}
        prefix_mask = 0

        for _, row in df_case.iterrows():
            act, lc = row["Activity"], row["Lifecycle"]
            if lc == "START":
                start_map[act] = row["Timestamp"]
            elif lc == "COMPLETE" and act in start_map:
                a = name_to_id.get(act)
                if a is None:
                    continue
                s_key = int(prefix_mask)
                counts.setdefault(s_key, {})
                counts[s_key][a] = counts[s_key].get(a, 0) + 1
                prefix_mask |= 1 << int(a)
                del start_map[act]

    trans_probs = {}
    for s_key, ac in counts.items():
        total = sum(ac.values())
        trans_probs[s_key] = {a: c / total for a, c in ac.items()}
    return trans_probs


def _build_soliman_ql_transitions_from_event_log(event_log_path: str,
                                                  trans_probs: dict) -> list:
    """Build (s_key, a, r, s_next_key, a_next, done, mask_next) for SolimanQL.

    Reward = transition probability for valid steps, +10 for terminal.
    """
    import pandas as pd

    df = pd.read_csv(event_log_path)
    name_to_id = _load_activity_name_to_id()
    all_transitions = []

    for _, df_case in df.groupby("CaseID"):
        df_case = df_case.sort_values("Timestamp")
        start_map: dict = {}
        episode = []
        prefix_mask = 0

        for _, row in df_case.iterrows():
            act, lc = row["Activity"], row["Lifecycle"]
            if lc == "START":
                start_map[act] = row["Timestamp"]
            elif lc == "COMPLETE" and act in start_map:
                a = name_to_id.get(act)
                if a is None:
                    continue
                s_key = int(prefix_mask)
                r = trans_probs.get(s_key, {}).get(a, 0.0)
                episode.append((s_key, int(a), float(r)))
                prefix_mask |= 1 << int(a)
                del start_map[act]

        for i, (s_key, a, r) in enumerate(episode):
            s_next_key = s_key | (1 << a)
            done = (i == len(episode) - 1)
            r_final = 10.0 if done else r   # terminal reward from paper (QL version)
            mask_next = np.array(
                [0.0 if (s_next_key >> j) & 1 else 1.0 for j in range(ACTION_SIZE)],
                dtype=np.float32,
            )
            a_next = episode[i + 1][1] if not done else 0
            all_transitions.append((s_key, a, r_final, s_next_key, a_next, done, mask_next))

    return all_transitions


def _run_soliman_ql_once(num_patients: int = 200,
                          n_agents: int = 3, n_epochs: int = 20):
    """Offline tabular Q-learning with transition-probability reward (Soliman QL)."""
    algo_name = "SolimanQL"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {algo_name} ===\n", num_patients)

    event_log_path = os.path.join("data", "evaluate", f"event_log_{num_patients}_random_base.csv")
    if not os.path.exists(event_log_path):
        _append_note(log_dir, "Baseline event log missing; generating.\n", num_patients)
        run_simulation_isolated(num_patients, None, "random_base",
                                seed=EVAL_SEED, model_name="Random", gen_id=0)

    print("📊 Computing transition probabilities from event log...")
    trans_probs = _compute_transition_probs_from_event_log(event_log_path)
    print(f"   States with observations: {len(trans_probs)}")

    print("📊 Building SolimanQL transitions...")
    transitions = _build_soliman_ql_transitions_from_event_log(event_log_path, trans_probs)
    print(f"   Total transitions: {len(transitions)}")

    baseline_avg_time = run_simulation_isolated(
        num_patients, None, "random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients)
    _append_note(log_dir, "====\n", num_patients)

    t0 = time.time()
    best_agent, best_val_time = None, None

    for idx in range(n_agents):
        seed = TRAIN_SEED + idx
        agent = get_agent("SolimanQL", seed=seed)
        np.random.seed(seed)

        print(f"\n🔄 SolimanQL candidate {idx + 1}/{n_agents} (seed={seed})...")
        shuffled = list(transitions)
        for epoch in range(n_epochs):
            np.random.shuffle(shuffled)
            for (s_key, a, r, s_next_key, _, done, mask_next) in shuffled:
                agent.update(s_key, a, r, s_next_key, mask_next, done)
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch + 1}/{n_epochs} | Q-table: {len(agent.q_table)} states")

        val_time = run_simulation_isolated(
            num_patients, agent, f"solimanql_val{idx}",
            is_model_run=False, seed=EVAL_SEED, model_name=algo_name, gen_id=0)
        print(f"   Candidate {idx + 1} val time: {val_time:.2f} mins")
        _append_note(log_dir, f"Candidate {idx + 1}: val_time={val_time:.2f}\n", num_patients)

        if best_val_time is None or val_time < best_val_time:
            best_val_time = val_time
            best_agent = agent

    train_minutes = (time.time() - t0) / 60.0
    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    best_agent.save(model_path)
    _append_note(log_dir, f"Training Time: {train_minutes:.2f} minutes\n", num_patients)
    _append_note(log_dir, f"Q-table size (best): {len(best_agent.q_table)} states\n", num_patients)

    avg_time = run_simulation_isolated(
        num_patients, best_agent, "solimanql",
        is_model_run=True, seed=EVAL_SEED, model_name=algo_name, gen_id=0)
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"✅ SolimanQL Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")
    print(f"   Q-table states seen: {len(best_agent.q_table)}")


def _run_soliman_dqn_once(num_patients: int = 200,
                           n_agents: int = 3, n_episodes: int = 5000):
    """Online DQN training with bitmask-only state and transition-prob reward (Soliman DQN)."""
    from common.soliman_env import SolimanEnv

    algo_name = "SolimanDQN"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {algo_name} ===\n", num_patients)

    event_log_path = os.path.join("data", "evaluate", f"event_log_{num_patients}_random_base.csv")
    if not os.path.exists(event_log_path):
        _append_note(log_dir, "Baseline event log missing; generating.\n", num_patients)
        run_simulation_isolated(num_patients, None, "random_base",
                                seed=EVAL_SEED, model_name="Random", gen_id=0)

    print("📊 Computing transition probabilities for SolimanDQN env...")
    trans_probs = _compute_transition_probs_from_event_log(event_log_path)
    print(f"   States with observations: {len(trans_probs)}")

    baseline_avg_time = run_simulation_isolated(
        num_patients, None, "random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients)
    _append_note(log_dir, "====\n", num_patients)

    # Epsilon schedule: max(eps_min, exp(-eps_decay * episode)) — from paper
    EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.001

    t0 = time.time()
    best_agent, best_val_time = None, None

    for idx in range(n_agents):
        seed = TRAIN_SEED + idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        env = SolimanEnv(ACTION_SIZE, trans_probs)
        agent = SolimanDQNAgent(action_size=ACTION_SIZE, seed=seed)

        print(f"\n🔄 SolimanDQN candidate {idx + 1}/{n_agents} (seed={seed}, episodes={n_episodes})...")
        for ep in range(n_episodes):
            eps = max(EPS_MIN, np.exp(-EPS_DECAY * ep))
            state = env.reset()
            done = False
            while not done:
                mask = env.get_action_mask()
                action = agent.act(state, mask, eps)
                next_state, reward, done = env.step(action)
                next_mask = env.get_action_mask()
                agent.step(state, action, reward, next_state, done, next_mask)
                state = next_state
            agent.notify_episode_done()

            if (ep + 1) % 1000 == 0:
                print(f"   Episode {ep + 1}/{n_episodes} | eps={eps:.3f}")

        val_time = run_simulation_isolated(
            num_patients, agent, f"solimanDQN_val{idx}",
            is_model_run=False, seed=EVAL_SEED, model_name=algo_name, gen_id=0)
        print(f"   Candidate {idx + 1} val time: {val_time:.2f} mins")
        _append_note(log_dir, f"Candidate {idx + 1}: val_time={val_time:.2f}\n", num_patients)

        if best_val_time is None or val_time < best_val_time:
            best_val_time = val_time
            best_agent = agent

    train_minutes = (time.time() - t0) / 60.0
    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    best_agent.save(model_path)
    _append_note(log_dir, f"Training Time: {train_minutes:.2f} minutes\n", num_patients)

    avg_time = run_simulation_isolated(
        num_patients, best_agent, "solimanDQN",
        is_model_run=True, seed=EVAL_SEED, model_name=algo_name, gen_id=0)
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"✅ SolimanDQN Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")


def run_soliman_ql_once(num_patients: int = 200):
    """Entry point for SolimanQL (tabular Q-learning, transition-prob reward)."""
    _run_soliman_ql_once(num_patients)


def run_soliman_dqn_once(num_patients: int = 200):
    """Entry point for SolimanDQN (small MLP DQN, bitmask-only state)."""
    _run_soliman_dqn_once(num_patients)


def run_hundogan_sarsa_once(num_patients: int = 200):
    """Entry point for HundoganSARSA (tabular SARSA)."""
    _run_hundogan_tabular_once("HundoganSARSA", num_patients)


def run_next_best_action_once(num_patients: int = 200, k_neighbors: int = 5):
    """Entry point for Weinzierl et al. (2020) - Prescriptive BPM: Next Best Actions.

    Offline: multi-task LSTM (mpp) + k-NN suffix store (mcs) trained on random-base event log.
    Online: predict suffix, check KPI threshold, find best valid candidate suffix.
    """
    algo_name = "NextBestAction"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {algo_name} (Weinzierl 2020, k={k_neighbors}) ===\n", num_patients)

    event_log_path = os.path.join("data", "evaluate", f"event_log_{num_patients}_random_base.csv")
    random_queue_log = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    if not os.path.exists(event_log_path) or not os.path.exists(random_queue_log):
        _append_note(log_dir, "Baseline logs missing; generating via simulation.\n", num_patients)
        run_simulation_isolated(
            num_patients=num_patients, agent=None, version_output="random_base",
            seed=EVAL_SEED, model_name="Random", gen_id=0,
        )

    t0 = time.time()
    agent = NextBestActionAgent(STATE_SIZE, ACTION_SIZE, seed=TRAIN_SEED, k_neighbors=k_neighbors)
    agent.train(event_log_path)
    train_minutes = (time.time() - t0) / 60.0

    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    agent.save(model_path)

    baseline_avg_time = run_simulation_isolated(
        num_patients=num_patients, agent=None, version_output="random_base",
        seed=EVAL_SEED, model_name="Random", gen_id=0,
    )
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients)
    _append_note(log_dir, "====\n", num_patients)
    _append_note(log_dir, f"Training Time: {train_minutes:.2f} minutes\n", num_patients)
    _append_note(log_dir, f"KPI Threshold: {agent.threshold:.2f}\n", num_patients)
    _append_note(log_dir, f"k_neighbors: {k_neighbors}\n", num_patients)

    avg_time = run_simulation_isolated(
        num_patients=num_patients, agent=agent, version_output="nextbestaction",
        is_model_run=True, seed=EVAL_SEED, model_name="NextBestAction", gen_id=0,
    )
    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"✅ NextBestAction (k={k_neighbors}) Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")


def train_one_generation(agent, algo_name: str, gen_id: int, train_config, num_patients: int):
    """Train one generation."""
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
        f"Training Time: {train_minutes:.2f} minutes\n",
        num_patients=num_patients,
    )

    return ckpt_paths

SUPPORTED_ALGOS = [
    "DQN", "DDQN", "PerDQN", "Dueling", "Rainbow", "MultiStepDQN",
    "FORLAPS", "LearningToAct", "FCFS", "Greedy",
    "HundoganQL", "HundoganSARSA",
    "SolimanQL", "SolimanDQN",
    "NextBestAction",
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Missing algorithms")
        print("   Example: python main_1_unlimited_queue.py DQN 200")
        print(f"   Supported algorithms: {SUPPORTED_ALGOS}")
        sys.exit(1)

    ALGO_TO_RUN = sys.argv[1].strip()
    NUM_PATIENTS = int(sys.argv[2].strip()) if len(sys.argv) >= 3 else 200

    if ALGO_TO_RUN not in SUPPORTED_ALGOS:
        print(f"❌ Incorrect algorithm: {ALGO_TO_RUN}")
        print(f"   Supported algorithms: {SUPPORTED_ALGOS}")
        sys.exit(1)

    print(f"🚀 ALGO_TO_RUN = {ALGO_TO_RUN}\n")
    set_deterministic_mode()

    # Special cases
    if ALGO_TO_RUN == "FORLAPS":
        run_forlaps_once_forlaps(train_seed=TRAIN_SEED, eval_seed=EVAL_SEED, num_patients=NUM_PATIENTS)
        sys.exit(0)

    if ALGO_TO_RUN == "LearningToAct":
        run_learning_to_act_once(num_patients=NUM_PATIENTS)
        sys.exit(0)
    
    if ALGO_TO_RUN == "FCFS":
        run_fcfs_once(num_patients=NUM_PATIENTS)
        sys.exit(0)
    
    if ALGO_TO_RUN == "Greedy":
        run_greedy_once(num_patients=NUM_PATIENTS)
        sys.exit(0)

    if ALGO_TO_RUN == "HundoganQL":
        run_hundogan_ql_once(num_patients=NUM_PATIENTS)
        sys.exit(0)

    if ALGO_TO_RUN == "HundoganSARSA":
        run_hundogan_sarsa_once(num_patients=NUM_PATIENTS)
        sys.exit(0)

    if ALGO_TO_RUN == "SolimanQL":
        run_soliman_ql_once(num_patients=NUM_PATIENTS)
        sys.exit(0)

    if ALGO_TO_RUN == "SolimanDQN":
        run_soliman_dqn_once(num_patients=NUM_PATIENTS)
        sys.exit(0)

    if ALGO_TO_RUN == "NextBestAction":
        run_next_best_action_once(num_patients=NUM_PATIENTS)
        sys.exit(0)

    # RL training
    agent = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
    log_dir = os.path.join("logs", ALGO_TO_RUN)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {ALGO_TO_RUN} ===\n", num_patients=NUM_PATIENTS)

    # Get baseline
    random_base_queue_log = os.path.join("data", "raw", f"queue_log_{NUM_PATIENTS}_random_base.csv")
    if os.path.exists(random_base_queue_log):
        print(f"✅ Found existing random_base queue log")
        baseline_avg_time = run_simulation_isolated(num_patients=NUM_PATIENTS, agent=None, version_output="random_base_eval", seed=EVAL_SEED, model_name="Random", gen_id=0)
    else:
        print(f"⚠️ Random_base queue log not found. Generating...")
        baseline_avg_time = run_simulation_isolated(num_patients=NUM_PATIENTS, agent=None, version_output="random_base", seed=EVAL_SEED, model_name="Random", gen_id=0)

    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients=NUM_PATIENTS)
    _append_note(log_dir, "====\n", num_patients=NUM_PATIENTS)

    train_config = get_train_config(ALGO_TO_RUN, NUM_PATIENTS)

    for gen_id in range(1, 6):
        if gen_id not in train_config:
            break

        ckpt_paths = train_one_generation(agent, ALGO_TO_RUN, gen_id, train_config, NUM_PATIENTS)

        if not ckpt_paths:
            _append_note(log_dir, f"Gen {gen_id}: No checkpoints produced.\n", num_patients=NUM_PATIENTS)
            continue

        # Evaluate checkpoints
        _append_note(log_dir, f"\n====\nGEN {gen_id} CHECKPOINT EVALUATION\n", num_patients=NUM_PATIENTS)
        
        best_ckpt = None
        best_avg_time = None
        best_ep = None
        
        for ckpt_path in ckpt_paths:
            import re
            match = re.search(r'_(\d+)\.pth$', ckpt_path)
            ep = int(match.group(1)) if match else 0
            
            agent_eval = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
            agent_eval.load(ckpt_path)
            
            avg_time = run_simulation_isolated(
                num_patients=NUM_PATIENTS,
                agent=agent_eval,
                version_output=f"{ALGO_TO_RUN.lower()}_g{gen_id}_ep{ep}",
                is_model_run=False,
                seed=EVAL_SEED,
                model_name=ALGO_TO_RUN,
                gen_id=gen_id,
            )
            
            improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time else 0.0
            _append_note(log_dir, f"Checkpoint ep={ep}: Avg Time={avg_time:.2f} | Improvement={improvement:+.2f}%\n", num_patients=NUM_PATIENTS)
            print(f"  Gen {gen_id} Checkpoint ep={ep}: Avg Time={avg_time:.2f} | Improvement={improvement:+.2f}%")
            
            if best_avg_time is None or avg_time < best_avg_time:
                best_avg_time = avg_time
                best_ckpt = ckpt_path
                best_ep = ep

        if best_ckpt:
            import shutil
            final_name = f"final_{NUM_PATIENTS}_gen_{gen_id}.pth"
            final_path = os.path.join(log_dir, final_name)
            shutil.copyfile(best_ckpt, final_path)
            
            improvement_pct = ((baseline_avg_time - best_avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
            _append_note(log_dir, f"Gen {gen_id} Final: ep={best_ep} | Avg Time: {best_avg_time:.2f} | Improvement: {improvement_pct:+.2f}%\n", num_patients=NUM_PATIENTS)
            print(f"✅ Gen {gen_id} Complete: Best ep={best_ep} | Avg Time: {best_avg_time:.2f} | Improvement: {improvement_pct:+.2f}%")
            
            # Generate data for next generation
            if gen_id < 5 and (gen_id + 1) in train_config:
                agent_for_data = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
                agent_for_data.load(final_path)
                run_simulation_isolated(
                    NUM_PATIENTS,
                    agent_for_data,
                    f"{ALGO_TO_RUN}_gen_{gen_id}",
                    True,
                    EVAL_SEED,
                    ALGO_TO_RUN,
                    gen_id,
                )

    print(f"\n🎉 {ALGO_TO_RUN} Training Cycle Complete!")
