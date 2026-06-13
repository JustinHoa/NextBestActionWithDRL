from agents.multi_step_dqn_agent import MultiStepDqnAgent

class GaussDynamicQueueMultiStepAgent(MultiStepDqnAgent):
    """
    Multi-Step DQN Agent for Gauss Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Gaussian capacity: N(2×resource, 0.5×resource), clipped to [1×, 3×]
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
