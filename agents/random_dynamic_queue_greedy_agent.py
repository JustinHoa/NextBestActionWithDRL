from agents.greedy_agent import GreedyAgent

class RandomDynamicQueueGreedyAgent(GreedyAgent):
    """
    Greedy Agent for Random Dynamic Queue Environment.
    - State size: 87 dimensions
    - Action size: 21 (no mock action)
    - Random capacity: Uniform[1×resource, 3×resource]
    - Selects action with shortest waiting time
    """
    
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size)
    
    def _extract_waiting_times(self, state):
        """Extract waiting times from Random Dynamic Queue state."""
        if len(state) >= 65:
            # State: [gender, marital, 21 blanks, 21 waiting_times, ...]
            # Indices 23-43: norm_wait (21 activities)
            norm_wait = state[23:44]
            return norm_wait
        return super()._extract_waiting_times(state)
