import argparse

from src.config import CHECKPOINT_INTERVAL
from src.trainer.dqn_trainer import AVAILABLE_AGENTS, train

MODE_DEFAULT_EPISODES = {
    "test": 1_000,
    "full": 50_000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN experiments with action masking.")
    parser.add_argument("--model-name", required=True, help="Name of the model run for checkpoint/log directories.")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                        help="Training regime. 'test' for quick runs, 'full' for long training.")
    parser.add_argument("--episodes", type=int, help="Number of training episodes. Overrides mode defaults.")
    parser.add_argument("--save-interval", type=int,
                        help="Checkpoint interval in episodes. Defaults to 10k for full, min(episodes,10k) for test.")
    parser.add_argument("--agent", choices=AVAILABLE_AGENTS, default="dqn",
                        help="Choose the agent variant (DQN, Double, Dueling, Multi-step, PER, Rainbow).")
    return parser.parse_args()


def main():
    args = parse_args()
    episodes = args.episodes or MODE_DEFAULT_EPISODES[args.mode]

    if args.save_interval:
        save_interval = args.save_interval
    elif args.mode == "full":
        save_interval = CHECKPOINT_INTERVAL
    else:
        save_interval = min(episodes, CHECKPOINT_INTERVAL)

    print(f"Starting {args.mode} training for model '{args.model_name}' with {episodes} episodes.")
    print(f"Checkpoint every {save_interval} episodes.")

    train(num_episodes=episodes,
          save_interval=save_interval,
          model_name=args.model_name,
          agent_type=args.agent)


if __name__ == "__main__":
    main()
