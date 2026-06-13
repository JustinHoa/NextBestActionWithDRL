from agents.per_dqn_agent import PerDqnAgent

class LinearDynamicQueuePerAgent(PerDqnAgent):
    """
    Prioritized Experience Replay DQN Agent for Linear Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Linear capacity expansion: 1× → 2× → 3× → ...
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
