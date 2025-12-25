import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from common.env import FillBlanksEnv
from common.utils import ACTION_SIZE, DEVICE, STATE_SIZE, ensure_dir
from simulation.simulation_process import run_simulation


def _save_rng_state():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state_dict):
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["numpy"])
    torch.set_rng_state(state_dict["torch"])
    if torch.cuda.is_available() and state_dict["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state_dict["torch_cuda"])


def _run_simulation_isolated(num_patients, agent, version_output, is_model_run=False, seed=None, model_name="", gen_id=0):
    training_state = _save_rng_state()
    try:
        return run_simulation(num_patients, agent, version_output, is_model_run, seed, model_name, gen_id)
    finally:
        _restore_rng_state(training_state)


def _append_note(log_dir: str, text: str, num_patients: int = 200) -> None:
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, f"training_notes_{num_patients}.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)


def _save_pickle(file_path: str, obj) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(file_path: str, default=None):
    if not os.path.exists(file_path):
        return default
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _generate_offline_transitions_random(
    data_path: str,
    num_episodes: int,
    max_steps_per_episode: int = 30,
    seed: int = 123,
):
    rng = np.random.RandomState(seed)
    env = FillBlanksEnv(STATE_SIZE, ACTION_SIZE, data_path=data_path)

    transitions = []
    for _ in tqdm(range(num_episodes), desc="Generating offline dataset"):
        state = env.reset()
        for _step in range(max_steps_per_episode):
            mask = env.get_action_mask()
            valid_actions = np.where(mask == 1.0)[0]
            if len(valid_actions) == 0:
                break
            action = int(rng.choice(valid_actions))
            next_state, reward, done = env.step(action)
            next_mask = env.get_action_mask()
            transitions.append((state, action, float(reward), next_state, float(done), next_mask))
            state = next_state
            if done:
                break
    return transitions


def _augment_transitions_forlaps(
    transitions,
    target_steps: int,
    time_jitter_ratio: float = 0.10,
    deletion_ratio: float = 0.05,
    seed: int = 123,
):
    rng = np.random.RandomState(seed)
    if not transitions:
        return []

    if deletion_ratio > 0:
        keep_mask = rng.rand(len(transitions)) >= deletion_ratio
        kept = [t for t, k in zip(transitions, keep_mask) if k]
    else:
        kept = list(transitions)
    if not kept:
        kept = list(transitions)

    def _jitter_state(x: np.ndarray) -> np.ndarray:
        y = np.array(x, copy=True)
        delta = rng.uniform(-time_jitter_ratio, time_jitter_ratio)
        y[-1] = float(np.clip(y[-1] * (1.0 + delta), 0.0, 1.0))
        return y

    jittered = []
    for (s, a, r, ns, d, nm) in kept:
        jittered.append((_jitter_state(s), a, r, _jitter_state(ns), d, nm))

    if target_steps <= 0:
        return jittered
    if len(jittered) >= target_steps:
        idx = rng.choice(len(jittered), size=target_steps, replace=False)
    else:
        idx = rng.choice(len(jittered), size=target_steps, replace=True)
    return [jittered[i] for i in idx]


def _iter_minibatches(data, batch_size: int, seed: int = 123):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield [data[i] for i in batch_idx]


def _offline_train_dqn_from_transitions(
    agent: DQNAgent,
    transitions,
    epochs: int,
    batch_size: int,
    gamma: float,
    seed: int = 123,
):
    if not transitions:
        raise RuntimeError("Offline dataset is empty; cannot train FORLAPS.")

    losses = []
    for ep in range(1, epochs + 1):
        epoch_losses = []
        for batch in _iter_minibatches(transitions, batch_size=batch_size, seed=seed + ep):
            states = torch.from_numpy(np.vstack([b[0] for b in batch])).float().to(DEVICE)
            actions = torch.from_numpy(np.vstack([b[1] for b in batch])).long().to(DEVICE)
            rewards = torch.from_numpy(np.vstack([b[2] for b in batch])).float().to(DEVICE)
            next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).float().to(DEVICE)
            dones = torch.from_numpy(np.vstack([b[4] for b in batch]).astype(np.uint8)).float().to(DEVICE)
            next_masks = torch.from_numpy(np.vstack([b[5] for b in batch])).float().to(DEVICE)
            weights = torch.ones_like(rewards)
            indices = None
            experiences = (states, actions, rewards, next_states, dones, next_masks, weights, indices)
            loss = agent.learn(experiences, gamma)
            epoch_losses.append(loss)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(mean_loss)
        print(f"FORLAPS offline epoch {ep}/{epochs} | loss={mean_loss:.6f} | steps={len(transitions)}")
    return losses


def run_forlaps_once(
    train_seed: int = 123,
    eval_seed: int = 42,
    num_patients: int = 200,
    num_episodes: int = 6000,
    max_steps_per_episode: int = 30,
    target_steps: int = 150_000,
    epochs: int = 10,
    batch_size: int = 256,
    gamma: float = 0.99,
) -> Tuple[List[float], float, float]:
    """Entry point for offline FORLAPS benchmark.

    Runs: offline dataset -> augmentation -> offline Q-learning -> save -> simulation.
    """
    algo_name = "FORLAPS"
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    _append_note(log_dir, "\n=== Train model: FORLAPS ===\n", num_patients)

    random_queue_log = os.path.join("data", "raw", f"queue_log_{num_patients}_random_base.csv")
    if not os.path.exists(random_queue_log):
        _append_note(log_dir, "Random queue log missing; generating via simulation.\n", num_patients)
        _run_simulation_isolated(num_patients=num_patients, agent=None, version_output="random_base", seed=eval_seed)

    offline_dir = os.path.join("data", "offline")
    ensure_dir(offline_dir)
    offline_path = os.path.join(offline_dir, "forlaps_random_transitions.pkl")

    transitions = _load_pickle(offline_path, default=None)
    if transitions is None:
        transitions = _generate_offline_transitions_random(
            data_path=random_queue_log,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            seed=train_seed,
        )
        _save_pickle(offline_path, transitions)
        _append_note(log_dir, f"Saved offline transitions: {offline_path} | n={len(transitions)}\n", num_patients)
    else:
        _append_note(log_dir, f"Loaded offline transitions: {offline_path} | n={len(transitions)}\n", num_patients)

    augmented = _augment_transitions_forlaps(
        transitions,
        target_steps=target_steps,
        time_jitter_ratio=0.10,
        deletion_ratio=0.05,
        seed=train_seed,
    )
    _append_note(log_dir, f"Augmented transitions: n={len(augmented)}\n", num_patients)

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = 1e-4

    import time as time_module
    t0 = time_module.time()
    
    losses = _offline_train_dqn_from_transitions(
        agent=agent,
        transitions=augmented,
        epochs=epochs,
        batch_size=batch_size,
        gamma=gamma,
        seed=train_seed,
    )
    
    train_seconds = time_module.time() - t0
    train_minutes = train_seconds / 60.0

    model_path = os.path.join(log_dir, f"model_{num_patients}.pth")
    agent.save(model_path)
    _append_note(log_dir, f"Saved FORLAPS model: {model_path}\n", num_patients)

    baseline_avg_time = _run_simulation_isolated(
        num_patients=num_patients,
        agent=None,
        version_output="random_base",
        seed=eval_seed,
    )
    
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", num_patients)
    _append_note(log_dir, "====\n", num_patients)
    _append_note(log_dir, f"Episode: {num_episodes} episode\n", num_patients)
    _append_note(log_dir, f"Training Time: {train_minutes:.2f} minutes\n", num_patients)
    
    avg_time = _run_simulation_isolated(
        num_patients=num_patients,
        agent=agent,
        version_output="forlaps",
        is_model_run=True,
        seed=eval_seed,
        model_name="FORLAPS",
        gen_id=0,
    )

    improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
    _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", num_patients)
    print(f"FORLAPS Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")
    
    # Plot monitoring
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'FORLAPS Training - {num_patients} patients')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, f"monitoring_{num_patients}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Saved monitoring plot: {plot_path}")
    
    return losses, baseline_avg_time, avg_time
