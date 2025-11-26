from src.agents.dqn_agent import DQNAgent
from src.config import ACTION_SIZE, STATE_SIZE
from src.networks.dueling_q_network import DuelingQNetwork


class DuelingDQNAgent(DQNAgent):
    """DQN agent using a dueling network architecture."""

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE, seed: int = 0):
        super().__init__(state_size=state_size,
                         action_size=action_size,
                         seed=seed,
                         network_cls=DuelingQNetwork)
