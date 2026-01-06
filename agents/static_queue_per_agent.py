from agents.per_dqn_agent import PerDqnAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class StaticQueuePerAgent(PerDqnAgent):
    """
    PerDQN Agent for Static Queue Environment.
    - State size: 67 dimensions
    - Action size: 22 (21 activities + 1 mock)
    - Uses Prioritized Experience Replay
    """
    
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
