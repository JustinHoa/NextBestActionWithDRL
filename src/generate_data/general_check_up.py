import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import os
import pickle
from tqdm import tqdm  # for progress bar

# ================= Load Data =================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, '../../data/preprocess_data/general_check_up/')
file_path = os.path.join(input_dir, "general_check_up_preprocess.csv")

df = pd.read_csv(file_path)

# Convert state and next_state from string representation to list
df['state'] = df['state'].apply(ast.literal_eval)
df['next_state'] = df['next_state'].apply(ast.literal_eval)

# ================= State Processing =================
def process_state(state):
    """
    Process state by separating prefix and remaining_time.
    Round remaining_time to reduce sparsity for Q-table.
    Assume last 21 elements of state are remaining_time for each node.
    """
    prefix = state[:-21]
    remaining_time = state[-21:]
    remaining_time_rounded = [round(t, 1) for t in remaining_time]  # round to 1 decimal
    return tuple(prefix + remaining_time_rounded)

df['state_tuple'] = df['state'].apply(process_state)
df['next_state_tuple'] = df['next_state'].apply(process_state)

# ================= Q-learning Setup =================
all_actions = df['action'].unique().tolist()
Q = defaultdict(lambda: {a: 0.0 for a in all_actions})

alpha = 0.1
gamma = 0.9
num_epochs = 50

# ================= Offline Q-learning =================
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Training"):
        s = row['state_tuple']
        a = row['action']
        s_next = row['next_state_tuple']
        r = row['reward']

        # Q-learning update rule
        max_q_next = max(Q[s_next].values()) if Q[s_next] else 0.0
        Q[s][a] = Q[s][a] + alpha * (r + gamma * max_q_next - Q[s][a])

# ================= Extract Policy =================
policy = {s: max(actions, key=actions.get) for s, actions in Q.items()}

# ================= Save Q-table =================
# CSV format
q_rows = []
for s, actions in Q.items():
    row = {"state": s}
    row.update(actions)
    q_rows.append(row)
q_df = pd.DataFrame(q_rows)
q_df.to_csv(os.path.join(input_dir, "q_table.csv"), index=False)

# ================= Save Next Best Action =================
nba_rows = []
for s, actions in Q.items():
    best_action = max(actions, key=actions.get)
    nba_rows.append({"state": s, "next_best_action": best_action})

nba_df = pd.DataFrame(nba_rows)
nba_df.to_csv(os.path.join(input_dir, "next_best_action.csv"), index=False)