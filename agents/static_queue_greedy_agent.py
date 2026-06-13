import pickle
import numpy as np


class StaticQueueGreedyAgent:
    """
    Greedy Agent for Static Queue.
    Selects the action with the shortest estimated duration.
    """
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.activity_durations = {
            0: 5, 1: 10, 2: 15, 3: 20, 4: 25, 5: 30, 6: 35,
            7: 40, 8: 45, 9: 50, 10: 55, 11: 60, 12: 65, 13: 70,
            14: 75, 15: 80, 16: 85, 17: 90, 18: 95, 19: 100, 20: 105
        }
    
    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump({"activity_durations": self.activity_durations}, f)
    
    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.activity_durations = data.get("activity_durations", self.activity_durations)
    
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        
        best_action = valid_actions[0]
        best_duration = float('inf')
        
        for action_id in valid_actions:
            duration = self.activity_durations.get(int(action_id), 100)
            if duration < best_duration:
                best_duration = duration
                best_action = action_id
        
        return int(best_action)
