import json
import os
import pickle
from typing import Dict, Optional

import numpy as np


class LinearProgrammingAgent:
    """
    Linear Programming (LP) Agent.
    
    Uses a simple heuristic based on linear combination of:
    1. Processing time (prefer shorter)
    2. Queue length (prefer less congested)
    3. Staff availability (prefer more staff)
    
    This approximates an LP solution without requiring actual optimization solver.
    """
    
    def __init__(self, state_size: int, action_size: int, activity_info_path: str = "data/raw/activity_info.json"):
        self.state_size = state_size
        self.action_size = action_size
        self.activity_info = {}
        self._load_activity_info(activity_info_path)
        
        # Weights for LP heuristic
        self.w_time = 0.4      # Weight for processing time
        self.w_queue = 0.3     # Weight for queue length
        self.w_staff = 0.3     # Weight for staff availability
    
    def _load_activity_info(self, activity_info_path: str) -> None:
        """Load activity information from activity_info.json."""
        if not os.path.exists(activity_info_path):
            # Try relative path from project root
            activity_info_path = os.path.join(os.path.dirname(__file__), "..", activity_info_path)
        
        if not os.path.exists(activity_info_path):
            print(f"Warning: activity_info.json not found at {activity_info_path}")
            # Default values
            self.activity_info = {i: {"mean_time": 5.0, "staff": 2} for i in range(21)}
            return
        
        with open(activity_info_path, "r", encoding="utf-8") as f:
            activity_data = json.load(f)
        
        # Map activity names to IDs and extract info
        for activity_name, info in activity_data.items():
            activity_id = int(info["id"]) - 1  # Convert to 0-indexed
            mean_time = info.get("mean_time", 5.0)
            
            # Handle special case where mean_time is a dict (e.g., Payment)
            if isinstance(mean_time, dict):
                mean_time = sum(mean_time.values()) / len(mean_time)
            
            self.activity_info[activity_id] = {
                "mean_time": float(mean_time),
                "staff": int(info.get("staff", 2))
            }
    
    def save(self, filename: str) -> None:
        """Save agent configuration."""
        with open(filename, "wb") as f:
            pickle.dump({
                "activity_info": self.activity_info,
                "w_time": self.w_time,
                "w_queue": self.w_queue,
                "w_staff": self.w_staff
            }, f)
    
    def load(self, filename: str) -> None:
        """Load agent configuration."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.activity_info = data.get("activity_info", {})
            self.w_time = data.get("w_time", 0.4)
            self.w_queue = data.get("w_queue", 0.3)
            self.w_staff = data.get("w_staff", 0.3)
    
    def _extract_queue_info(self, state: np.ndarray) -> np.ndarray:
        """
        Extract queue lengths from state.
        
        State format: [gender, marital, prefix(21), norm_wait(21), norm_time(1)]
        norm_wait contains normalized queue information.
        """
        if len(state) >= 44:  # 2 + 21 + 21
            # Extract normalized wait times (indices 23-43)
            norm_wait = state[23:44]
            return norm_wait
        return np.zeros(21)
    
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        """
        Select action using LP heuristic.
        
        Score = -w_time * norm_time - w_queue * norm_queue + w_staff * norm_staff
        (Higher score is better)
        
        Args:
            state: Current state
            mask: Action mask (1.0 for valid actions, 0.0 for invalid)
            eps: Epsilon (ignored, LP is deterministic)
        
        Returns:
            Action ID with best LP score among valid actions
        """
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0
        
        # Extract queue information from state
        queue_info = self._extract_queue_info(state)
        
        # Calculate scores for each valid action
        best_action = None
        best_score = float('-inf')
        
        for action_id in valid_actions:
            action_id_int = int(action_id)
            
            # Get activity info
            info = self.activity_info.get(action_id_int, {"mean_time": 5.0, "staff": 2})
            mean_time = info["mean_time"]
            staff = info["staff"]
            
            # Normalize values
            norm_time = mean_time / 10.0  # Assume max time ~10
            norm_queue = queue_info[action_id_int] if action_id_int < len(queue_info) else 0.0
            norm_staff = staff / 5.0  # Assume max staff ~5
            
            # Calculate LP score (higher is better)
            # Prefer: shorter time, less queue, more staff
            score = -self.w_time * norm_time - self.w_queue * norm_queue + self.w_staff * norm_staff
            
            if score > best_score:
                best_score = score
                best_action = action_id
        
        return int(best_action) if best_action is not None else int(valid_actions[0])
