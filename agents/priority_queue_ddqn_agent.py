from agents.ddqn_agent import DDQNAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class PriorityQueueDDQNAgent(DDQNAgent):
    """
    Double DQN Agent for Priority Queue Environment.
    - State size: 86 dimensions
    - Action size: 22 (21 activities + 1 mock)
    - Inherits Double DQN learning logic from DDQNAgent
    """
    
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
