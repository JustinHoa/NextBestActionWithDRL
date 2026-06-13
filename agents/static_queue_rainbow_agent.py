from agents.rainbow_agent import RainbowAgent
from networks.noisy_net import RainbowQNetwork
from common.utils import DEVICE


class StaticQueueRainbowAgent(RainbowAgent):
    """
    Rainbow Agent for Static Queue Environment.
    - State size: 67 dimensions
    - Action size: 22 (21 activities + 1 mock)
    - Combines: Double DQN, Dueling, PER, Multi-step, Noisy Nets, Distributional RL
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
