from agents.dqn_agent import DQNAgent

class RandomDynamicQueueDQNAgent(DQNAgent):
    """
    DQN Agent for Random Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Random capacity: Uniform[1×resource, 3×resource]
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, seed=seed)
