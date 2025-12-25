"""
Utility to reconstruct queue log from event log.

Event log format: CaseID, Activity, Timestamp, Lifecycle (START/COMPLETE)
Queue log format: Time, Activity1, Activity2, ..., ActivityN (queue counts at each timestamp)
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path


def reconstruct_queue_log_from_event_log(event_log_path: str, activity_info_path: str = 'data/raw/activity_info.json'):
    """
    Reconstruct queue log from event log by tracking START/COMPLETE events.
    
    Args:
        event_log_path: Path to event log CSV (CaseID, Activity, Timestamp, Lifecycle)
        activity_info_path: Path to activity_info.json (for activity ordering)
    
    Returns:
        DataFrame with columns: Time, Activity1, Activity2, ..., ActivityN
    """
    # Load event log
    df_event = pd.read_csv(event_log_path)
    required_cols = {'CaseID', 'Activity', 'Timestamp', 'Lifecycle'}
    if not required_cols.issubset(set(df_event.columns)):
        raise ValueError(f"Event log missing required columns. Need {required_cols}, got {set(df_event.columns)}")
    
    # Load activity info for ordering
    activity_info_path = Path(activity_info_path)
    if not activity_info_path.exists():
        activity_info_path = Path('..') / activity_info_path
    activity_info = json.loads(activity_info_path.read_text(encoding='utf-8'))
    activity_names = sorted(activity_info.keys(), key=lambda x: activity_info[x]['id'])
    
    # Sort events by timestamp
    df_event = df_event.sort_values('Timestamp').reset_index(drop=True)
    
    # Track queue counts over time
    queue_state = {act: 0 for act in activity_names}
    queue_log = []
    
    for _, row in df_event.iterrows():
        activity = row['Activity']
        lifecycle = row['Lifecycle']
        timestamp = float(row['Timestamp'])
        
        if activity not in queue_state:
            continue
        
        if lifecycle == 'START':
            # Patient enters queue (waiting for resource)
            queue_state[activity] += 1
        elif lifecycle == 'COMPLETE':
            # Patient leaves queue (finished activity)
            queue_state[activity] = max(0, queue_state[activity] - 1)
        
        # Record snapshot
        snapshot = {'Time': timestamp}
        snapshot.update(queue_state)
        queue_log.append(snapshot)
    
    # Convert to DataFrame
    df_queue = pd.DataFrame(queue_log)
    
    # Ensure column order: Time, then activities in ID order
    cols = ['Time'] + activity_names
    df_queue = df_queue[cols]
    
    return df_queue
