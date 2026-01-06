from agents.multi_step_dqn_agent import MultiStepDqnAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class PriorityQueueMultiStepAgent(MultiStepDqnAgent):
    """
    Multi-step DQN Agent for Priority Queue Environment.
    - State size: 86 dimensions
    - Action size: 22 (21 activities + 1 mock)
    - Uses n-step returns for better credit assignment
    """
    
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
