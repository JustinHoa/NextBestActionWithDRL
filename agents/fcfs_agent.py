import json
import os
import pickle
from typing import Dict, Optional

import numpy as np


class FCFSAgent:
    """
    First Come First Serve (FCFS) Agent.
    
    Selects activities in a fixed order based on activity ID (1-21).
    This simulates a simple sequential processing strategy.
    """
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.activity_order = list(range(21))  # Activities 0-20 (mapped from 1-21)
    
    def save(self, filename: str) -> None:
        """Save agent configuration."""
        with open(filename, "wb") as f:
            pickle.dump({"activity_order": self.activity_order}, f)
    
    def load(self, filename: str) -> None:
        """Load agent configuration."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.activity_order = data.get("activity_order", list(range(21)))
    
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        """
        Select action using FCFS strategy.
        
        Args:
            state: Current state (not used in FCFS)
            mask: Action mask (1.0 for valid actions, 0.0 for invalid)
            eps: Epsilon (ignored, FCFS is deterministic)
        
        Returns:
            Action ID following FCFS order
        """
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        
        # Select the first valid action in the predefined order
        for action_id in self.activity_order:
            if action_id < len(mask) and mask[action_id] == 1.0:
                return int(action_id)
        
        # Fallback: return first valid action
        return int(valid_actions[0])
