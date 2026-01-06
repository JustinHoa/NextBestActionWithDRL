from agents.dueling_agent import DuelingAgent
from networks.duel_net import DuelingQNetwork
from common.utils import DEVICE


class PriorityQueueDuelingAgent(DuelingAgent):
    """
    Dueling DQN Agent for Priority Queue Environment.
    - State size: 86 dimensions
    - Action size: 22 (21 activities + 1 mock)
    - Uses Dueling network architecture with Double DQN learning
    """
    
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
