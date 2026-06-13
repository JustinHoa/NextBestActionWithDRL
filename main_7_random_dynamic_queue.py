"""
Main script for Random Dynamic Queue training and evaluation.
Supports: RandomDynamicQueueDQN, RandomDynamicQueueDDQN, RandomDynamicQueueDueling, RandomDynamicQueuePerDQN, RandomDynamicQueueRainbow, RandomDynamicQueueMultiStepDQN
Plus baselines: FCFS, Greedy, FORLAPS
"""
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

from agents.random_dynamic_queue_dqn_agent import RandomDynamicQueueDQNAgent
from agents.random_dynamic_queue_ddqn_agent import RandomDynamicQueueDDQNAgent
from agents.random_dynamic_queue_dueling_agent import RandomDynamicQueueDuelingAgent
from agents.random_dynamic_queue_per_agent import RandomDynamicQueuePerAgent
from agents.random_dynamic_queue_rainbow_agent import RandomDynamicQueueRainbowAgent
from agents.random_dynamic_queue_multistep_agent import RandomDynamicQueueMultiStepAgent
from agents.random_dynamic_queue_fcfs_agent import RandomDynamicQueueFCFSAgent
from agents.random_dynamic_queue_greedy_agent import RandomDynamicQueueGreedyAgent
from agents.dqn_agent import DQNAgent
from common.random_dynamic_queue_env import RandomDynamicQueueEnv
from common.utils import (
    STATE_SIZE_DYNAMIC_QUEUE,
    ACTION_SIZE_DYNAMIC_QUEUE,
    DEVICE,
    get_train_config,
    ensure_dir,
    plot_training_status,
)
from simulation.random_dynamic_queue_simulation import run_random_dynamic_queue_simulation

# === REPRODUCIBILITY SETTINGS ===
TRAIN_SEED = 12300
EVAL_SEED = 4200

def set_deterministic_mode():
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Deterministic mode enabled with TRAIN_SEED={TRAIN_SEED}")

def get_agent(algo_name: str, seed: int = 0):
    if algo_name == "RandomDynamicQueueDQN":
        return RandomDynamicQueueDQNAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueDDQN":
        return RandomDynamicQueueDDQNAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueDueling":
        return RandomDynamicQueueDuelingAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueuePerDQN":
        return RandomDynamicQueuePerAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueRainbow":
        return RandomDynamicQueueRainbowAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueMultiStepDQN":
        return RandomDynamicQueueMultiStepAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueFCFS":
        return RandomDynamicQueueFCFSAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueGreedy":
        return RandomDynamicQueueGreedyAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    if algo_name == "RandomDynamicQueueFORLAPS":
        return DQNAgent(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, seed=seed)
    raise ValueError(f"Unknown algorithm: {algo_name}")

def _append_note(log_dir: str, text: str, num_patients: int = 200):
    ensure_dir(log_dir)
    note_path = os.path.join(log_dir, f"training_notes_{num_patients}.txt")
    with open(note_path, "a", encoding="utf-8") as f:
        f.write(text)

def _parse_checkpoint_episode(ckpt_path: str) -> int:
    import re
    match = re.search(r'_(\d+)\.pth$', ckpt_path)
    return int(match.group(1)) if match else 0

def train_random_dynamic_queue_generic(agent, algo_name: str, num_patients: int, gen_id: int, train_config: dict):
    log_dir = os.path.join("logs", algo_name)
    ensure_dir(log_dir)
    
    config = train_config[gen_id]
    
    if config.get("load_model"):
        model_path = os.path.join(log_dir, config["load_model"])
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"✅ Loaded model: {config['load_model']}")
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING: [{algo_name} Gen {gen_id}]")
    print(f"   Description: {config['description']}")
    print(f"   Episodes: {config['episodes']}")
    print(f"   Data: {config['data_file']}")
    print(f"{'='*60}\n")
    
    env = RandomDynamicQueueEnv(STATE_SIZE_DYNAMIC_QUEUE, ACTION_SIZE_DYNAMIC_QUEUE, data_path=config["data_file"])
    
    scores_window = deque(maxlen=100)
    all_scores, all_losses = [], []
    eps = config["eps_start"]
    ckpt_paths = []
    t0 = time.time()
    
    tqdm_bar = tqdm(range(1, config["episodes"] + 1), desc=f"Training {algo_name} Gen {gen_id}")
    for i_episode in tqdm_bar:
        state = env.reset()
        score = 0
        for _ in range(30):
            mask = env.get_action_mask()
            action = agent.act(state, mask=mask, eps=eps)
            next_state, reward, done = env.step(action)
            next_mask = env.get_action_mask()
            
            loss = agent.step(state, action, reward, next_state, done, next_mask=next_mask)
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
            ckpt_name = f"checkpoint_{num_patients}_gen_{gen_id}_{i_episode}.pth"
            ckpt_path = os.path.join(log_dir, ckpt_name)
            agent.save(ckpt_path)
            ckpt_paths.append(ckpt_path)
            tqdm_bar.write(f"💾 Checkpoint saved: {ckpt_name}")
    
    train_seconds = time.time() - t0
    train_minutes = train_seconds / 60.0
    
    plot_training_status(all_scores, all_losses, algo_name, gen_id, save_dir=log_dir, num_patients=num_patients)
    
    _append_note(log_dir, f"Episode: {config['episodes']} episode\n"
                          f"Training Time: {train_minutes:.2f} minutes\n", 
                 num_patients=num_patients)
    
    return ckpt_paths

SUPPORTED_ALGOS = [
    "RandomDynamicQueueDQN", "RandomDynamicQueueDDQN", "RandomDynamicQueueDueling", 
    "RandomDynamicQueuePerDQN", "RandomDynamicQueueRainbow", "RandomDynamicQueueMultiStepDQN",
    "RandomDynamicQueueFCFS", "RandomDynamicQueueGreedy", "RandomDynamicQueueFORLAPS"
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Missing algorithms")
        print("   Example: python main_7_random_dynamic_queue.py RandomDynamicQueueDQN 200")
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

    # Baseline algorithms
    if ALGO_TO_RUN in ["RandomDynamicQueueFCFS", "RandomDynamicQueueGreedy", "RandomDynamicQueueFORLAPS"]:
        print(f"⚠️ {ALGO_TO_RUN} baseline - simulation only (no training)")
        agent = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
        log_dir = os.path.join("logs", ALGO_TO_RUN)
        ensure_dir(log_dir)
        _append_note(log_dir, f"\n=== Baseline: {ALGO_TO_RUN} ===\n", NUM_PATIENTS)
        
        model_path = os.path.join(log_dir, f"model_{NUM_PATIENTS}.pth")
        agent.save(model_path)
        
        baseline_avg_time = run_random_dynamic_queue_simulation(NUM_PATIENTS, None, "random_base_eval", EVAL_SEED, "Random", 0)
        _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", NUM_PATIENTS)
        
        avg_time = run_random_dynamic_queue_simulation(NUM_PATIENTS, agent, f"random_{ALGO_TO_RUN.lower()}", EVAL_SEED, ALGO_TO_RUN, 0)
        improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
        _append_note(log_dir, f"Avg Time: {avg_time:.2f} | Improve Percentage: {improvement:+.2f}%\n", NUM_PATIENTS)
        print(f"✅ {ALGO_TO_RUN} Avg Time: {avg_time:.2f} mins | Improvement: {improvement:+.2f}%")
        sys.exit(0)

    # RL training
    agent = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
    log_dir = os.path.join("logs", ALGO_TO_RUN)
    ensure_dir(log_dir)
    _append_note(log_dir, f"\n=== Train model: {ALGO_TO_RUN} ===\n", NUM_PATIENTS)
    
    random_base_queue_log = os.path.join("data", "raw", f"queue_log_{NUM_PATIENTS}_Random_gen_0.csv")
    if os.path.exists(random_base_queue_log):
        print(f"✅ Found existing Random Dynamic Queue random_base")
        baseline_avg_time = run_random_dynamic_queue_simulation(NUM_PATIENTS, None, "random_base_eval", EVAL_SEED, "Random", 0)
    else:
        print(f"⚠️ Random Dynamic Queue random_base not found. Generating...")
        baseline_avg_time = run_random_dynamic_queue_simulation(NUM_PATIENTS, None, "random_base", EVAL_SEED, "Random", 0)
    
    _append_note(log_dir, f"Random base Avg Time: {baseline_avg_time:.2f}\n", NUM_PATIENTS)
    
    prev_best_avg_time = None
    train_config = get_train_config(ALGO_TO_RUN, NUM_PATIENTS)
    
    for gen_id in range(1, 4):
        if gen_id not in train_config:
            break
        
        _append_note(log_dir, "====\n", NUM_PATIENTS)
        ckpt_paths = train_random_dynamic_queue_generic(agent, ALGO_TO_RUN, NUM_PATIENTS, gen_id, train_config)
        
        if not ckpt_paths:
            _append_note(log_dir, f"Gen {gen_id}: No checkpoints produced.\n", NUM_PATIENTS)
            continue
        
        _append_note(log_dir, f"\n====\nGEN {gen_id} CHECKPOINT EVALUATION\n", NUM_PATIENTS)
        
        best_ckpt = None
        best_avg_time = None
        best_ep = None
        
        for ckpt_path in ckpt_paths:
            ep = _parse_checkpoint_episode(ckpt_path)
            agent_eval = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
            agent_eval.load(ckpt_path)
            
            avg_time = run_random_dynamic_queue_simulation(NUM_PATIENTS, agent_eval, f"RandomDynamicQueue_g{gen_id}_ep{ep}", EVAL_SEED, ALGO_TO_RUN, gen_id)
            improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
            
            _append_note(log_dir, f"Checkpoint ep={ep}: Avg Time={avg_time:.2f} | Improvement={improvement:+.2f}%\n", NUM_PATIENTS)
            print(f"  Gen {gen_id} Checkpoint ep={ep}: Avg Time={avg_time:.2f} | Improvement={improvement:+.2f}%")
            
            if best_avg_time is None or avg_time < best_avg_time:
                best_avg_time = avg_time
                best_ckpt = ckpt_path
                best_ep = ep
        
        if best_ckpt:
            final_name = f"final_{NUM_PATIENTS}_gen_{gen_id}.pth"
            final_path = os.path.join(log_dir, final_name)
            shutil.copyfile(best_ckpt, final_path)
            
            improvement_pct = ((baseline_avg_time - best_avg_time) / baseline_avg_time) * 100 if baseline_avg_time > 0 else 0.0
            _append_note(log_dir, f"Gen {gen_id} Final: ep={best_ep} | Avg Time: {best_avg_time:.2f} | Improvement: {improvement_pct:+.2f}%\n", NUM_PATIENTS)
            print(f"✅ Gen {gen_id} Complete: Best ep={best_ep} | Avg Time: {best_avg_time:.2f} | Improvement: {improvement_pct:+.2f}%")
            
            if prev_best_avg_time is not None and best_avg_time >= prev_best_avg_time:
                _append_note(log_dir, f"STOP: Gen {gen_id} ({best_avg_time:.2f}) >= Gen {gen_id-1} ({prev_best_avg_time:.2f})\n", NUM_PATIENTS)
                print(f"🛑 Early stopping: Gen {gen_id} did not improve.")
                break
            
            prev_best_avg_time = best_avg_time
            
            if gen_id < 3 and (gen_id + 1) in train_config:
                agent_for_data = get_agent(ALGO_TO_RUN, seed=TRAIN_SEED)
                agent_for_data.load(final_path)
                run_random_dynamic_queue_simulation(NUM_PATIENTS, agent_for_data, f"RandomDynamicQueue_gen_{gen_id}", EVAL_SEED, ALGO_TO_RUN, gen_id)
    
    print(f"\n🎉 {ALGO_TO_RUN} Training Cycle Complete!")
