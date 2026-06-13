from agents.rainbow_agent import RainbowAgent
from networks.noisy_net import RainbowQNetwork
from common.utils import DEVICE


class DynamicQueueRainbowAgent(RainbowAgent):
    """
    Rainbow Agent for Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Combines: Double DQN, Dueling, PER, Multi-step, Noisy Nets, Distributional RL
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
