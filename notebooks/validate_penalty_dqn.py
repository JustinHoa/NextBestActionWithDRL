"""
Validation script for PenaltyDQN model.
Tests the trained model on all possible states to check for invalid actions.
"""
import os
import sys
import numpy as np
import pandas as pd
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.penalty_dqn_agent import PenaltyDQNAgent
from common.penalty_env import PenaltyEnv
from common.utils import STATE_SIZE, ACTION_SIZE
from simulation.simulation_process import run_simulation

def load_possible_states(csv_path='../data/raw/possible_states.csv'):
    """Load possible states from CSV file."""
    if not os.path.exists(csv_path):
        csv_path = 'data/raw/possible_states.csv'
    
    df = pd.read_csv(csv_path, header=None)
    print(f"✅ Loaded {len(df)} possible states from {csv_path}")
    return df.values

def create_complete_state(possible_state_row, queue_trace, activity_info):
    """
    Convert possible_state (23 dims) to complete state (45 dims).
    
    possible_state: [gender, marital, 21 prefix]
    complete_state: [gender, marital, 21 prefix, 21 waiting_times, 1 norm_time]
    """
    gender = possible_state_row[0]
    marital = possible_state_row[1]
    prefix = possible_state_row[2:23]
    
    # Sample random queue from trace
    random_idx = np.random.randint(0, len(queue_trace))
    queues = queue_trace[random_idx]
    
    # Calculate waiting times
    mean_time_arr = []
    staff_arr = []
    for key in activity_info.keys():
        staff_arr.append(activity_info[key]["staff"])
        if key == "Payment":
            t = (activity_info[key]["mean_time"]["Cash"] + activity_info[key]["mean_time"]["Credit"]) / 2
            mean_time_arr.append(t)
        else:
            mean_time_arr.append(activity_info[key]["mean_time"])
    
    mean_time_arr = np.array(mean_time_arr)
    staff_arr = np.array(staff_arr)
    
    estimated_wait_time = (queues * mean_time_arr) / staff_arr
    norm_wait_time = estimated_wait_time / 200.0
    
    # Random normalized time
    norm_time = np.random.rand()
    
    # Concatenate to form complete state
    complete_state = np.concatenate(([gender, marital], prefix, norm_wait_time, [norm_time]))
    
    return complete_state

def test_penalty_dqn_on_possible_states(model_path, num_patients=200):
    """
    Test PenaltyDQN model on all possible states.
    
    Returns:
        dict: Statistics about valid/invalid actions
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING PENALTYDQN MODEL")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"📂 Loading model from: {model_path}")
    agent = PenaltyDQNAgent(STATE_SIZE, ACTION_SIZE)
    agent.load(model_path)
    agent.qnetwork_local.eval()
    print("✅ Model loaded successfully\n")
    
    # Load possible states
    possible_states = load_possible_states()
    
    # Load activity info
    activity_info_path = '../data/raw/activity_info.json'
    if not os.path.exists(activity_info_path):
        activity_info_path = 'data/raw/activity_info.json'
    
    with open(activity_info_path, 'r', encoding='utf-8') as f:
        activity_info = json.load(f)
    
    # Load queue trace
    queue_log_path = f'../data/raw/queue_log_{num_patients}_random_base.csv'
    if not os.path.exists(queue_log_path):
        queue_log_path = f'data/raw/queue_log_{num_patients}_random_base.csv'
    
    df_queue = pd.read_csv(queue_log_path)
    queue_trace = df_queue.iloc[:, 1:].values  # Skip Time column
    
    # Create environment for validation
    env = PenaltyEnv(STATE_SIZE, ACTION_SIZE, data_path=queue_log_path)
    
    # Test each possible state
    total_states = len(possible_states)
    valid_actions = 0
    invalid_actions = 0
    invalid_details = []
    
    print(f"🔍 Testing {total_states} possible states...\n")
    
    for i, ps_row in enumerate(possible_states):
        # Create complete state
        complete_state = create_complete_state(ps_row, queue_trace, activity_info)
        
        # Get action from model (no epsilon, greedy)
        action = agent.act(complete_state, mask=None, eps=0.0)
        
        # Manually set env state to match
        env.features = complete_state[:2]
        env.blanks = complete_state[2:23]
        env.queues = (complete_state[23:44] * 200.0) * env.staff_arr / env.mean_time_arr
        env.total_time = complete_state[44] * env.max_trace_len
        env.done = False
        env.goal_mask = env._create_goal_mask()
        
        # Test action by stepping
        _, reward, _ = env.step(action)
        
        # Check if action was valid (reward >= 0) or invalid (reward < 0)
        if reward < 0:
            invalid_actions += 1
            invalid_details.append({
                'state_idx': i,
                'gender': int(ps_row[0]),
                'marital': int(ps_row[1]),
                'prefix': ps_row[2:23].astype(int).tolist(),
                'action': int(action),
                'reward': float(reward)
            })
        else:
            valid_actions += 1
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{total_states} states...")
    
    # Calculate statistics
    valid_pct = (valid_actions / total_states) * 100
    invalid_pct = (invalid_actions / total_states) * 100
    
    # Print results
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Total States Tested:    {total_states}")
    print(f"Valid Actions:          {valid_actions} ({valid_pct:.2f}%)")
    print(f"Invalid Actions:        {invalid_actions} ({invalid_pct:.2f}%)")
    print(f"{'='*80}\n")
    
    # Show some invalid action examples if any
    if invalid_actions > 0:
        print(f"⚠️ INVALID ACTION EXAMPLES (showing first 10):")
        for detail in invalid_details[:10]:
            print(f"   State {detail['state_idx']}: gender={detail['gender']}, marital={detail['marital']}")
            print(f"      Prefix: {detail['prefix']}")
            print(f"      Action: {detail['action']}, Reward: {detail['reward']:.1f}")
        print()
    
    results = {
        'total_states': total_states,
        'valid_actions': valid_actions,
        'invalid_actions': invalid_actions,
        'valid_pct': valid_pct,
        'invalid_pct': invalid_pct,
        'invalid_details': invalid_details
    }
    
    return results

def run_simulation_if_valid(results, model_path, num_patients=200):
    """Run simulation if all predictions are valid."""
    if results['invalid_actions'] == 0:
        print(f"✅ ALL PREDICTIONS ARE VALID! Running simulation...\n")
        
        # Load agent
        agent = PenaltyDQNAgent(STATE_SIZE, ACTION_SIZE)
        agent.load(model_path)
        
        # Run simulation
        from simulation.simulation_process import run_simulation
        
        avg_time = run_simulation(
            num_patients=num_patients,
            agent=agent,
            version_output="penaltydqn_validation",
            is_model_run=False,
            seed=42,
            model_name="PenaltyDQN",
            gen_id=1
        )
        
        print(f"\n{'='*80}")
        print(f"SIMULATION RESULTS")
        print(f"{'='*80}")
        print(f"Average Time: {avg_time:.2f} minutes")
        print(f"{'='*80}\n")
        
        return avg_time
    else:
        print(f"❌ SIMULATION SKIPPED: Model produced {results['invalid_actions']} invalid actions.")
        print(f"   Please retrain the model with more episodes or adjust hyperparameters.\n")
        return None

def validate_all_checkpoints(log_dir, num_patients=200):
    """Validate all checkpoints and save results to training notes."""
    import glob
    import re
    
    # Find all checkpoints
    ckpt_pattern = os.path.join(log_dir, f"checkpoint_{num_patients}_gen_1_*.pth")
    ckpt_paths = sorted(glob.glob(ckpt_pattern), key=lambda x: int(re.search(r'_(\d+)\.pth$', x).group(1)))
    
    if not ckpt_paths:
        print(f"❌ No checkpoints found at {ckpt_pattern}")
        return
    
    print(f"\n{'='*80}")
    print(f"VALIDATING {len(ckpt_paths)} CHECKPOINTS")
    print(f"{'='*80}\n")
    
    # Open training notes for appending
    notes_path = os.path.join(log_dir, f"training_notes_{num_patients}.txt")
    
    with open(notes_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("CHECKPOINT VALIDATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for i, ckpt_path in enumerate(ckpt_paths, 1):
            # Extract episode number
            ep = int(re.search(r'_(\d+)\.pth$', ckpt_path).group(1))
            ckpt_name = os.path.basename(ckpt_path)
            
            print(f"[{i}/{len(ckpt_paths)}] Testing {ckpt_name}...")
            
            # Validate checkpoint
            results = test_penalty_dqn_on_possible_states(ckpt_path, num_patients)
            
            # Write to notes
            f.write(f"Checkpoint: {ckpt_name} (Episode {ep})\n")
            f.write(f"  Valid Actions: {results['valid_actions']}/{results['total_states']} ({results['valid_pct']:.2f}%)\n")
            f.write(f"  Invalid Actions: {results['invalid_actions']}/{results['total_states']} ({results['invalid_pct']:.2f}%)\n")
            
            # Run simulation if 100% valid
            if results['invalid_actions'] == 0:
                print(f"  ✅ 100% Valid! Running simulation...")
                avg_time = run_simulation_if_valid(results, ckpt_path, num_patients)
                if avg_time:
                    f.write(f"  Simulation Avg Time: {avg_time:.2f} minutes\n")
                f.write(f"  Status: PASSED (100% valid actions)\n")
            else:
                print(f"  ❌ Failed: {results['invalid_actions']} invalid actions")
                f.write(f"  Status: FAILED ({results['invalid_actions']} invalid actions)\n")
            
            f.write("\n")
            f.flush()
        
        f.write("="*60 + "\n")
        f.write("VALIDATION COMPLETE\n")
        f.write("="*60 + "\n")
    
    print(f"\n✅ Validation results saved to: {notes_path}\n")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single checkpoint: python validate_penalty_dqn.py <model_path> [num_patients]")
        print("  All checkpoints:   python validate_penalty_dqn.py --all <log_dir> [num_patients]")
        print("\nExamples:")
        print("  python validate_penalty_dqn.py ../logs/PenaltyDQN/final_200_gen_1.pth 200")
        print("  python validate_penalty_dqn.py --all ../logs/PenaltyDQN 200")
        sys.exit(1)
    
    # Check if validating all checkpoints
    if sys.argv[1] == "--all":
        if len(sys.argv) < 3:
            print("❌ Please provide log directory")
            sys.exit(1)
        log_dir = sys.argv[2]
        num_patients = int(sys.argv[3]) if len(sys.argv) >= 4 else 200
        validate_all_checkpoints(log_dir, num_patients)
    else:
        # Single checkpoint validation
        model_path = sys.argv[1]
        num_patients = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
        
        # Validate model
        results = test_penalty_dqn_on_possible_states(model_path, num_patients)
        
        # Run simulation if valid
        run_simulation_if_valid(results, model_path, num_patients)
