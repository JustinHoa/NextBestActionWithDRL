import pickle
import numpy as np


class DynamicQueueFCFSAgent:
    """
    First Come First Serve (FCFS) Agent for Dynamic Queue.
    Selects activities in a fixed order based on activity ID (1-21).
    """
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.activity_order = list(range(21))
    
    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump({"activity_order": self.activity_order}, f)
    
    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.activity_order = data.get("activity_order", list(range(21)))
    
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        
        for action_id in self.activity_order:
            if action_id < len(mask) and mask[action_id] == 1.0:
                return int(action_id)
        
        return int(valid_actions[0])
