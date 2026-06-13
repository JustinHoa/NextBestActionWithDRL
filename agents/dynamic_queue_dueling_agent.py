from agents.dueling_agent import DuelingAgent
from networks.duel_net import DuelingQNetwork
from common.utils import DEVICE


class DynamicQueueDuelingAgent(DuelingAgent):
    """
    Dueling DQN Agent for Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Uses Dueling network architecture with Double DQN learning
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
