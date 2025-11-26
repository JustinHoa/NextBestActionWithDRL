import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.agents.dqn_agent import DQNAgent
from src.agents.double_dqn_agent import DoubleDQNAgent
from src.agents.dqn_per_agent import DQNPerAgent
from src.config import (ACTION_SIZE, CHECKPOINT_INTERVAL, EPS_DECAY, EPS_END,
                        EPS_START, MAX_T, STATE_SIZE)
from src.envs.fill_blank_env import FillBlanksEnv

AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dqn_per": DQNPerAgent,
}

AVAILABLE_AGENTS = tuple(AGENT_REGISTRY.keys())


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_loss_plot(loss_history, path: Path, title: str):
    if len(loss_history) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label='DQN Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train(num_episodes: int,
          agent: Optional[object] = None,
          save_interval: int = CHECKPOINT_INTERVAL,
          model_name: str = "dqn_action_mask",
          max_steps: int = MAX_T,
          agent_type: str = "dqn",
          agent_kwargs: Optional[dict] = None):
    env = FillBlanksEnv(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    if agent is None:
        agent_cls = AGENT_REGISTRY.get(agent_type)
        if agent_cls is None:
            raise ValueError(f"Unknown agent type '{agent_type}'. Available: {list(AGENT_REGISTRY)}")
        kwargs = agent_kwargs or {}
        agent = agent_cls(**kwargs)

    model_dir = _ensure_dir(Path("models") / model_name)

    history = []
    loss_history = []
    episode_rewards = []
    eps = EPS_START
    scores_window = deque(maxlen=100)

    tqdm_bar = tqdm(range(1, num_episodes + 1), desc=f"Training {model_name} [{agent_type}]")

    for i_episode in tqdm_bar:
        state = env.reset()
        score = 0.0

        for _ in range(max_steps):
            mask = env.get_action_mask()
            action = agent.act(state, mask, eps)
            next_state, reward, done = env.step(action)

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                loss_history.append(loss)

            history.append({
                'state': state.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
                'episode': i_episode
            })

            state = next_state
            score += reward
            if done:
                break

        episode_rewards.append(score)
        scores_window.append(score)
        eps = max(EPS_END, eps * EPS_DECAY)

        if i_episode % max(1, save_interval) == 0:
            checkpoint_path = model_dir / f"checkpoint_{i_episode}.pth"
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)

            with open(model_dir / f"history_{i_episode}.pkl", "wb") as f:
                pickle.dump(history, f)
            with open(model_dir / f"reward_{i_episode}.pkl", "wb") as f:
                pickle.dump(episode_rewards, f)

            loss_title = f"DQN Loss until Episode {i_episode}"
            _save_loss_plot(loss_history, model_dir / f"loss_checkpoint_{i_episode}.png", loss_title)

        if i_episode % 100 == 0:
            tqdm_bar.set_postfix(avg_score=f"{np.mean(scores_window):.2f}")

    torch.save(agent.qnetwork_local.state_dict(), model_dir / f"final_{num_episodes}.pth")
    with open(model_dir / f"final_history_{num_episodes}.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(model_dir / f"final_rewards_{num_episodes}.pkl", "wb") as f:
        pickle.dump(episode_rewards, f)

    _save_loss_plot(loss_history, model_dir / f"loss_final_{num_episodes}.png",
                    f"DQN Loss Final after {num_episodes} Episodes")

    print("Training finished!")
    return agent, episode_rewards, loss_history


def test(agent: Optional[object] = None,
         model_path: Optional[str] = None,
         max_steps: int = 100,
         agent_type: str = "dqn"):
    print("\n--- Testing Agent ---")
    env = FillBlanksEnv(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    eps = 0.0

    if model_path is not None:
        agent_cls = AGENT_REGISTRY.get(agent_type, DQNAgent)
        agent = agent_cls()
        state_dict = torch.load(model_path, map_location=agent.device)
        agent.qnetwork_local.load_state_dict(state_dict)
        agent.qnetwork_local.eval()
        print(f"Loaded model from '{model_path}'")

    if agent is None:
        raise ValueError("Provide agent or model_path to test.")

    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        mask = env.get_action_mask()
        action = agent.act(state, mask, eps)
        next_state, reward, done = env.step(action)
        print(f"\nStep {steps + 1}: Action {action}")
        print(f"  Blanks: {env.blanks.astype(int)}")
        print(f"  Queues: {env.queues.astype(int)}")
        print(f"  Reward: {reward:.1f}")
        state = next_state
        steps += 1
