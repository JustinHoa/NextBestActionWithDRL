import pickle
from typing import Dict, Optional

import numpy as np


class GreedyAgent:
    """
    Greedy Agent based on current waiting time.
    
    Selects the activity with the shortest current waiting time (queue length)
    from the state, rather than using static mean processing times.
    This reflects real-time queue conditions for fairer scheduling.
    """
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
    
    def save(self, filename: str) -> None:
        """Save agent configuration."""
        with open(filename, "wb") as f:
            pickle.dump({"state_size": self.state_size, "action_size": self.action_size}, f)
    
    def load(self, filename: str) -> None:
        """Load agent configuration."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.state_size = data.get("state_size", self.state_size)
            self.action_size = data.get("action_size", self.action_size)
    
    def _extract_waiting_times(self, state: np.ndarray) -> np.ndarray:
        """
        Extract normalized waiting times from state.
        
        State format: [gender, marital, prefix(21), norm_wait(21), norm_time(1)]
        - Indices 0-1: gender, marital
        - Indices 2-22: prefix (21 activities)
        - Indices 23-43: norm_wait (21 activities) <- waiting time info
        - Index 44: norm_time
        
        Returns:
            Array of normalized waiting times for each activity (21 values)
        """
        if len(state) >= 44:
            # Extract normalized wait times (indices 23-43)
            norm_wait = state[23:44]
            return norm_wait
        # Fallback: return zeros if state format is unexpected
        return np.zeros(21)
    
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        """
        Select action using Greedy strategy based on current waiting time.
        
        Chooses the valid action with the shortest current waiting time,
        reflecting real-time queue conditions.
        
        Args:
            state: Current state containing waiting time information
            mask: Action mask (1.0 for valid actions, 0.0 for invalid)
            eps: Epsilon (ignored, Greedy is deterministic)
        
        Returns:
            Action ID with shortest waiting time among valid actions
        """
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        
        # Extract waiting times from state
        waiting_times = self._extract_waiting_times(state)
        
        # Select valid action with minimum waiting time
        best_action = None
        best_wait_time = float('inf')
        
        for action_id in valid_actions:
            action_id_int = int(action_id)
            # Get waiting time for this action
            if action_id_int < len(waiting_times):
                wait_time = waiting_times[action_id_int]
            else:
                wait_time = 0.0  # Fallback
            
            if wait_time < best_wait_time:
                best_wait_time = wait_time
                best_action = action_id
        
        return int(best_action) if best_action is not None else int(valid_actions[0])
