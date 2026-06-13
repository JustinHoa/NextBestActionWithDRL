from agents.ddqn_agent import DDQNAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class DynamicQueueDDQNAgent(DDQNAgent):
    """
    Double DQN Agent for Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Inherits Double DQN learning logic from DDQNAgent
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
