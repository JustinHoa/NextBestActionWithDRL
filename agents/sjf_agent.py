import json
import os
import pickle
from typing import Dict, Optional

import numpy as np


class SJFAgent:
    """
    Shortest Job First (SJF) Agent.
    
    Selects activities based on their mean processing time (shortest first).
    Uses activity_info.json to determine processing times.
    """
    
    def __init__(self, state_size: int, action_size: int, activity_info_path: str = "data/raw/activity_info.json"):
        self.state_size = state_size
        self.action_size = action_size
        self.activity_times = {}
        self._load_activity_times(activity_info_path)
    
    def _load_activity_times(self, activity_info_path: str) -> None:
        """Load activity processing times from activity_info.json."""
        if not os.path.exists(activity_info_path):
            # Try relative path from project root
            activity_info_path = os.path.join(os.path.dirname(__file__), "..", activity_info_path)
        
        if not os.path.exists(activity_info_path):
            print(f"Warning: activity_info.json not found at {activity_info_path}")
            # Default: assume all activities have equal time
            self.activity_times = {i: 5.0 for i in range(21)}
            return
        
        with open(activity_info_path, "r", encoding="utf-8") as f:
            activity_info = json.load(f)
        
        # Map activity names to IDs (1-21) and extract mean_time
        for activity_name, info in activity_info.items():
            activity_id = int(info["id"]) - 1  # Convert to 0-indexed
            mean_time = info.get("mean_time", 5.0)
            
            # Handle special case where mean_time is a dict (e.g., Payment)
            if isinstance(mean_time, dict):
                # Use average of all payment methods
                mean_time = sum(mean_time.values()) / len(mean_time)
            
            self.activity_times[activity_id] = float(mean_time)
    
    def save(self, filename: str) -> None:
        """Save agent configuration."""
        with open(filename, "wb") as f:
            pickle.dump({"activity_times": self.activity_times}, f)
    
    def load(self, filename: str) -> None:
        """Load agent configuration."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.activity_times = data.get("activity_times", {})
    
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        """
        Select action using SJF strategy (shortest processing time first).
        
        Args:
            state: Current state (not used in SJF)
            mask: Action mask (1.0 for valid actions, 0.0 for invalid)
            eps: Epsilon (ignored, SJF is deterministic)
        
        Returns:
            Action ID with shortest processing time among valid actions
        """
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        
        # Select valid action with minimum processing time
        best_action = None
        best_time = float('inf')
        
        for action_id in valid_actions:
            action_time = self.activity_times.get(int(action_id), 5.0)
            if action_time < best_time:
                best_time = action_time
                best_action = action_id
        
        return int(best_action) if best_action is not None else int(valid_actions[0])
