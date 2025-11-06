import pandas as pd
import numpy as np
import ast
import os
from tqdm import tqdm
import json
from collections import defaultdict

# ================= Load Data =================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, '../../data/preprocess_data/general_check_up/')
file_path = os.path.join(input_dir, "general_check_up_preprocess.csv")

df = pd.read_csv(file_path)

# Convert state and next_state from string to list
df['state'] = df['state'].apply(ast.literal_eval)
df['next_state'] = df['next_state'].apply(ast.literal_eval)

# ================= State Processing =================
def split_state(state):
    """
    Split state into prefix and environment info (remaining_time)
    Assume last 21 elements are remaining_time
    """
    prefix = tuple(state[:-21])
    env = np.array(state[-21:], dtype=float)
    return prefix, env

df[['prefix', 'env']] = df['state'].apply(lambda s: pd.Series(split_state(s)))
df[['next_prefix', 'next_env']] = df['next_state'].apply(lambda s: pd.Series(split_state(s)))

# ================= Q-learning Setup =================
all_actions = df['action'].unique().tolist()

# Mỗi prefix có thể có nhiều cluster môi trường
Q_clusters = defaultdict(list)

alpha = 0.1
gamma = 0.9
num_epochs = 50
similarity_threshold = 0.5  # khoảng cách Euclidean tối đa để gộp cluster

# ================= Helper Functions =================
def find_cluster(prefix, env, clusters, threshold):
    """
    Return index of most similar cluster (same prefix) if within threshold.
    Otherwise, return None.
    """
    if prefix not in clusters or len(clusters[prefix]) == 0:
        return None
    min_dist = float('inf')
    best_idx = None
    for idx, cl in enumerate(clusters[prefix]):
        proto = cl['env_proto']
        dist = np.linalg.norm(env - proto)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
    if min_dist < threshold:
        return best_idx
    return None

def create_cluster(prefix, env, actions):
    """Create a new cluster for this prefix."""
    q = {a: 0.0 for a in actions}
    cluster = {"env_proto": env.copy(), "count": 1, "q": q}
    return cluster

# ================= Offline Q-learning with Similarity =================
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    df_shuffled = df.sample(frac=1, random_state=epoch).reset_index(drop=True)

    for _, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled), desc="Training"):
        prefix = row['prefix']
        a = row['action']
        next_prefix = row['next_prefix']
        env = row['env']
        next_env = row['next_env']
        r = row['reward']

        # --- Find or create cluster for current (prefix, env) ---
        idx = find_cluster(prefix, env, Q_clusters, similarity_threshold)
        if idx is None:
            new_cluster = create_cluster(prefix, env, all_actions)
            Q_clusters[prefix].append(new_cluster)
            idx = len(Q_clusters[prefix]) - 1
        cur_cl = Q_clusters[prefix][idx]

        # --- Find Q_next from next_prefix ---
        next_idx = find_cluster(next_prefix, next_env, Q_clusters, similarity_threshold)
        if next_idx is not None:
            q_next = Q_clusters[next_prefix][next_idx]['q']
            max_q_next = max(q_next.values())
        else:
            max_q_next = 0.0

        # --- Q-learning update ---
        old_q = cur_cl['q'][a]
        new_q = old_q + alpha * (r + gamma * max_q_next - old_q)
        cur_cl['q'][a] = new_q

        # --- Update environment prototype (running average) ---
        cur_cl['count'] += 1
        cur_cl['env_proto'] = cur_cl['env_proto'] + (env - cur_cl['env_proto']) / cur_cl['count']

# ================= Extract Results =================
q_rows = []
best_action_rows = []

for prefix, clusters in Q_clusters.items():
    for i, cl in enumerate(clusters):
        env_repr = cl['env_proto'].round(3).tolist()
        row = {
            "prefix": prefix,
            "cluster_id": i,
            "env_proto": json.dumps(env_repr),
        }
        row.update(cl['q'])
        q_rows.append(row)

        # Best action for this cluster
        best_action = max(cl['q'], key=cl['q'].get)
        best_action_rows.append({
            "prefix": prefix,
            "cluster_id": i,
            "env_proto": json.dumps(env_repr),
            "best_action": best_action,
            "best_q_value": cl['q'][best_action]
        })

# ================= Save to CSV =================
q_df = pd.DataFrame(q_rows)
best_action_df = pd.DataFrame(best_action_rows)

q_path = os.path.join(input_dir, "q_table.csv")
best_path = os.path.join(input_dir, "best_action_table.csv")

q_df.to_csv(q_path, index=False)
best_action_df.to_csv(best_path, index=False)

print(f"✅ Q-table saved to: {q_path}")
print(f"✅ Best actions saved to: {best_path}")
print(q_df.head())