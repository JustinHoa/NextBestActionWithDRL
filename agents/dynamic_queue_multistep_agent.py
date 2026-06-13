from agents.multi_step_dqn_agent import MultiStepDqnAgent
from networks.dqn_net import StandardQNetwork
from common.utils import DEVICE


class DynamicQueueMultiStepAgent(MultiStepDqnAgent):
    """
    Multi-step DQN Agent for Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Uses n-step returns for better credit assignment
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
