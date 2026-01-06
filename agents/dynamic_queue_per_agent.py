from agents.per_dqn_agent import PerDqnAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class DynamicQueuePerAgent(PerDqnAgent):
    """
    PerDQN Agent for Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Uses Prioritized Experience Replay
    """
    
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
