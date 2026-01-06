from agents.dqn_agent import DQNAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class PriorityQueueDQNAgent(DQNAgent):
    """
    DQN Agent for Priority Queue Environment.
    - State size: 86 dimensions
    - Action size: 22 (21 activities + 1 mock)
    - Inherits standard DQN learning logic from DQNAgent
    """
    
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
