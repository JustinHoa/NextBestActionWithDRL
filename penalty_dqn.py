import os
from collections import deque
import sys
import time
import shutil
import random

import numpy as np
import torch
from tqdm import tqdm

from agents.penalty_dqn_agent import PenaltyDQNAgent
from common.penalty_env import PenaltyEnv
from common.utils import (
    ACTION_SIZE,
    STATE_SIZE,
    DEVICE,
    get_train_config,
    ensure_dir,
    plot_training_status,
)
from simulation.simulation_process import run_simulation


TRAIN_SEED = 123000
EVAL_SEED = 42000


def set_deterministic_mode():
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f" Deterministic mode enabled with TRAIN_SEED={TRAIN_SEED}")


def save_rng_state():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state_dict):
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["numpy"])
    torch.set_rng_state(state_dict["torch"])
    if torch.cuda.is_available() and state_dict["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state_dict["torch_cuda"])


def run_simulation_isolated(num_patients, agent, version_output, is_model_run=False, seed=None, model_name="", gen_id=0):
    training_state = save_rng_state()

    try:
        result = run_simulation(num_patients, agent, version_output, is_model_run, seed, model_name, gen_id)
        return result
    finally:
        restore_rng_state(training_state)


def _append_note(log_dir: str, text: str, num_patients: int = 200):
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, f"training_notes_{num_patients}.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)


def _parse_checkpoint_episode(ckpt_path: str) -> int:
    import re

    match = re.search(r"_(\d+)\.pth$", ckpt_path)
    return int(match.group(1)) if match else 0


def get_agent(seed: int = 0):
    return PenaltyDQNAgent(STATE_SIZE, ACTION_SIZE, seed=seed)


def train_penalty_dqn(agent, num_patients: int):
    algo_name = "PenaltyDQN"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)

    train_config = get_train_config(algo_name, num_patients)
    config = train_config[1]

    smoke_episodes_raw = os.environ.get("SMOKE_EPISODES")
    smoke_episodes = int(smoke_episodes_raw) if smoke_episodes_raw else None
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

        if i_episode % 1000 == 0:
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
        f"Episode: {config['episodes']} episode\n" f"Training Time: {train_minutes:.2f} minutes\n",
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

    agent_eval = get_agent()
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
    return {"total": total, "valid": valid, "invalid": invalid, "valid_pct": valid_pct}


def run_penalty_dqn(num_patients: int):
    algo_name = "PenaltyDQN"

    agent = get_agent(seed=TRAIN_SEED)
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {algo_name} ===\n", num_patients=num_patients)

    random_base_queue_log = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    if os.path.exists(random_base_queue_log):
        print(f"✅ Found existing random_base queue log: {random_base_queue_log}")
        print("   Skipping random_base simulation...")
        baseline_avg_time = run_simulation_isolated(
            num_patients=num_patients,
            agent=None,
            version_output="random_base_eval",
            seed=EVAL_SEED,
            model_name="Random",
            gen_id=0,
        )
    else:
        baseline_avg_time = run_simulation_isolated(
            num_patients=num_patients,
            agent=None,
            version_output="random_base",
            seed=EVAL_SEED,
            model_name="Random",
            gen_id=0,
        )

    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients=num_patients)
    _append_note(log_dir, "====\n", num_patients=num_patients)

    ckpt_paths = train_penalty_dqn(agent, num_patients)

    if not ckpt_paths:
        _append_note(
            log_dir,
            "No checkpoints were produced in this run (episodes < checkpoint interval).\n",
            num_patients=num_patients,
        )
        print("✅ PenaltyDQN Training Complete!")
        print("   No checkpoints produced; skipping checkpoint validation/simulation.")
        return

    _append_note(log_dir, "\n====\nCHECKPOINT VALIDATION (possible_states.csv)\n", num_patients=num_patients)

    best_valid_ckpt = None
    best_valid_avg_time = None
    best_valid_ep = None

    for ckpt_path in ckpt_paths:
        ep = _parse_checkpoint_episode(ckpt_path)
        stats = _validate_penalty_checkpoint(ckpt_path, num_patients)
        _append_note(
            log_dir,
            f"Checkpoint: {os.path.basename(ckpt_path)} | valid={stats['valid']}/{stats['total']} ({stats['valid_pct']:.2f}%) | invalid={stats['invalid']}\n",
            num_patients=num_patients,
        )

        if stats["invalid"] != 0:
            continue

        agent_eval = get_agent()
        agent_eval.load(ckpt_path)

        avg_time = run_simulation_isolated(
            num_patients=num_patients,
            agent=agent_eval,
            version_output=f"penaltydqn_valid_ep{ep}",
            is_model_run=True,
            seed=EVAL_SEED,
            model_name=algo_name,
            gen_id=1,
        )

        _append_note(
            log_dir,
            f"  Simulation: Avg Time={avg_time:.2f} | Improvement={((baseline_avg_time - avg_time) / baseline_avg_time) * 100:+.2f}%\n",
            num_patients=num_patients,
        )

        if best_valid_avg_time is None or avg_time < best_valid_avg_time:
            best_valid_avg_time = avg_time
            best_valid_ckpt = ckpt_path
            best_valid_ep = ep

    if best_valid_ckpt is None:
        _append_note(
            log_dir,
            "No checkpoint reached 100% valid actions; no simulation winner selected.\n",
            num_patients=num_patients,
        )
        print("✅ PenaltyDQN Training Complete!")
        print("   No checkpoint achieved 100% valid actions.")
        return

    final_name = f"final_{num_patients}_gen_1.pth"
    final_path = os.path.join(log_dir, final_name)
    shutil.copyfile(best_valid_ckpt, final_path)

    improvement_pct = ((baseline_avg_time - best_valid_avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(
        log_dir,
        f"Final Model: {final_name} (from ep={best_valid_ep}) | Avg Time: {best_valid_avg_time:.2f} | Improve Percentage: {improvement_pct:+.2f}%\n",
        num_patients=num_patients,
    )

    print("✅ PenaltyDQN Training Complete!")
    print(
        f"   Best VALID checkpoint ep={best_valid_ep} | Avg Time: {best_valid_avg_time:.2f} | Improvement: {improvement_pct:+.2f}%"
    )


if __name__ == "__main__":
    num_patients = int(sys.argv[1].strip()) if len(sys.argv) >= 2 else 200
    set_deterministic_mode()
    run_penalty_dqn(num_patients)
