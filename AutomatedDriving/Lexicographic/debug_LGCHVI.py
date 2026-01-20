"""
Debug script to inspect Q-hulls and understand why car gets stuck
with pedestrian priority - IMPROVED VERSION
"""
import numpy as np

def inspect_qhulls(q_hulls_file="policies/CHVI_qhulls.npy"):
    """Load and inspect Q-hulls"""
    
    print("="*80)
    print("Q-HULLS INSPECTION FOR LEXICOGRAPHIC EXTRACTION")
    print("="*80)
    
    try:
        q_hulls = np.load(q_hulls_file, allow_pickle=True).item()
        print(f"✓ Loaded Q-hulls from {q_hulls_file}\n")
    except:
        print(f"✗ Could not load {q_hulls_file}")
        print("Run with CHVI_Calculate_hulls = True first!")
        return
    
    # Find all unique states
    all_states = set()
    for key in q_hulls.keys():
        c, p1, p2, action = key
        all_states.add((c, p1, p2))
    
    all_states = sorted(all_states)
    
    print(f"Total unique states: {len(all_states)}")
    
    # Find state ranges
    cars = sorted(set(c for c, p1, p2 in all_states))
    peds = sorted(set(p1 for c, p1, p2 in all_states))
    
    print(f"Car positions: {min(cars)} to {max(cars)} ({len(cars)} positions)")
    print(f"Pedestrian positions: {min(peds)} to {max(peds)} ({len(peds)} positions)")
    
    # Pick some representative states
    print("\n" + "="*80)
    print("EXAMINING REPRESENTATIVE STATES")
    print("="*80)
    
    # Find states where car is at different positions
    # Let's look at car at positions: min, 25%, 50%, 75%, max
    car_positions = [
        cars[0],
        cars[len(cars)//4],
        cars[len(cars)//2],
        cars[3*len(cars)//4],
        cars[-1]
    ]
    
    # For each car position, pick a state with pedestrians at some position
    for car_pos in car_positions[:3]:  # Just first 3 to avoid too much output
        # Find a state with this car position
        matching_states = [s for s in all_states if s[0] == car_pos]
        if not matching_states:
            continue
        
        # Pick first one
        state = matching_states[0]
        c, p1, p2 = state
        
        print(f"\n{'='*80}")
        print(f"STATE: car={c}, ped1={p1}, ped2={p2}")
        print(f"{'='*80}")
        
        # Check all actions
        action_names = ['stay', 'up', 'down', 'left', 'right', 'wait']
        
        # Collect info for all actions
        actions_info = []
        
        for action in range(6):
            key = (c, p1, p2, action)
            if key not in q_hulls:
                continue
            
            hull = np.array(q_hulls[key])
            action_name = action_names[action] if action < 6 else f"action{action}"
            
            # Find max values for each objective
            max_car = np.max(hull[:, 0]) if len(hull) > 0 else -np.inf
            max_ped1 = np.max(hull[:, 1]) if len(hull) > 0 else -np.inf
            max_ped2 = np.max(hull[:, 2]) if len(hull) > 0 else -np.inf
            
            actions_info.append({
                'action': action,
                'name': action_name,
                'hull_size': len(hull),
                'max_car': max_car,
                'max_ped1': max_ped1,
                'max_ped2': max_ped2,
                'hull': hull
            })
        
        # Show summary table
        print(f"\nAction Summary:")
        print(f"{'Action':<12} | {'Hull Size':<10} | {'Max r_car':<12} | {'Max r_ped1':<12} | {'Max r_ped2':<12}")
        print("-"*80)
        for info in actions_info:
            print(f"{info['name']:<12} | {info['hull_size']:<10} | {info['max_car']:<12.3f} | {info['max_ped1']:<12.3f} | {info['max_ped2']:<12.3f}")
        
        # Determine which action would be chosen
        print(f"\n{'='*40}")
        print("LEXICOGRAPHIC SELECTION:")
        print(f"{'='*40}")
        
        # CAR PRIORITY [r_car, r_ped1, r_ped2]
        car_choice = lex_choose_action(actions_info, priority='car')
        print(f"CAR priority:        Action {car_choice['action']} ({car_choice['name']})")
        print(f"  Best Q-vector: [r_car={car_choice['max_car']:.3f}, r_ped1={car_choice['max_ped1']:.3f}, r_ped2={car_choice['max_ped2']:.3f}]")
        
        # PEDESTRIAN PRIORITY [r_ped1, r_ped2, r_car]
        ped_choice = lex_choose_action(actions_info, priority='pedestrian')
        print(f"PEDESTRIAN priority: Action {ped_choice['action']} ({ped_choice['name']})")
        print(f"  Best Q-vector: [r_car={ped_choice['max_car']:.3f}, r_ped1={ped_choice['max_ped1']:.3f}, r_ped2={ped_choice['max_ped2']:.3f}]")
        
        # Analysis
        if car_choice['action'] != ped_choice['action']:
            print(f"\n⚠️  DIFFERENT CHOICES!")
            print(f"  Car priority prefers: {car_choice['name']}")
            print(f"  Ped priority prefers: {ped_choice['name']}")
            
            if ped_choice['action'] == 0:
                print(f"\n❗ PEDESTRIAN PRIORITY CHOOSES 'STAY'")
                print(f"  This might be why the car appears stuck!")
                print(f"\n  Why does it choose 'stay'?")
                
                # Compare stay vs the car priority choice
                stay_info = actions_info[0]
                move_info = car_choice
                
                print(f"\n  Comparison (STAY vs {move_info['name']}):")
                print(f"    r_ped1: {stay_info['max_ped1']:.3f} vs {move_info['max_ped1']:.3f}")
                if stay_info['max_ped1'] > move_info['max_ped1']:
                    print(f"      → STAY is better for ped1 ({stay_info['max_ped1']:.3f} > {move_info['max_ped1']:.3f})")
                    print(f"      → Lexicographic chooses STAY at first priority level!")
                elif abs(stay_info['max_ped1'] - move_info['max_ped1']) < 1e-9:
                    print(f"      → Tied on ped1, checking ped2...")
                    print(f"    r_ped2: {stay_info['max_ped2']:.3f} vs {move_info['max_ped2']:.3f}")
                    if stay_info['max_ped2'] > move_info['max_ped2']:
                        print(f"      → STAY is better for ped2")
                    elif abs(stay_info['max_ped2'] - move_info['max_ped2']) < 1e-9:
                        print(f"      → Tied on ped2, checking car...")
                        print(f"    r_car: {stay_info['max_car']:.3f} vs {move_info['max_car']:.3f}")


def lex_choose_action(actions_info, priority='car'):
    """Choose action using lexicographic ordering"""
    if priority == 'car':
        objective_order = ['max_car', 'max_ped1', 'max_ped2']
    else:
        objective_order = ['max_ped1', 'max_ped2', 'max_car']
    
    best_actions = actions_info[:]
    tol = 1e-9
    
    for obj_key in objective_order:
        if len(best_actions) <= 1:
            break
        
        # Find maximum value for this objective
        max_val = max(a[obj_key] for a in best_actions)
        
        # Keep only actions that achieve this maximum
        best_actions = [a for a in best_actions if abs(a[obj_key] - max_val) < tol]
    
    return best_actions[0]


if __name__ == "__main__":
    inspect_qhulls()


import numpy as np
from ADS_Environment import Environment

env = Environment(weights=None)

print("=" * 60)
print("UNDERSTANDING THE STATE SPACE")
print("=" * 60)

car_agent = env.agents[0]
print(f"Car origin: {car_agent.origin}")
print(f"Car name: {car_agent.name}")

print("\n" + "=" * 60)
print("POSITION MAPPING")
print("=" * 60)

for pos_idx in [0, 10, 20, 30, 40, 43, 50]:
    if pos_idx < env.map_num_cells:
        state = env.translate_state([pos_idx, 0, 0])
        print(f"Position {pos_idx:2d} → grid: {state[0]}")

print("\n" + "=" * 60)
print("CHECKING SUCCESS CONDITION")
print("=" * 60)

# Check different positions
print("Checking success at different positions:")
for pos_idx in [0, 10, 20, 30, 40, 43, 50]:
    if pos_idx < env.map_num_cells:
        state = env.translate_state([pos_idx, 0, 0])
        env.reset(state[0], [0, 0], [0, 0])
        
        # Check the succeeds attribute (not method!)
        success = car_agent.succeeds
        print(f"Position {pos_idx:2d} (grid {state[0]}): succeeds = {success}")

print("\n" + "=" * 60)
print("ACTIONS FROM POSITION 30")
print("=" * 60)

# Set to position 30
state_30 = env.translate_state([30, 31, 17])
print(f"Position 30 → grid: {state_30[0]}")

for action in range(6):
    env.reset(state_30[0], state_30[1], state_30[2])
    
    print(f"\nAction {action}:")
    pos_before = env.agents[0].position.copy()
    print(f"  Before: car at grid {pos_before}")
    
    next_state, reward, done = env.step([action])
    
    pos_after = env.agents[0].position.copy()
    print(f"  After:  car at grid {pos_after}")
    print(f"  Next state index: {next_state[0]}")
    print(f"  Moved: {not np.array_equal(pos_before, pos_after)}")
    print(f"  Done: {done[0]}")
    print(f"  Succeeds: {env.agents[0].succeeds}")

# Let's also check the map
print("\n" + "=" * 60)
print("MAP STRUCTURE")
print("=" * 60)
print(f"Map num cells: {env.map_num_cells}")
print(f"States agent left: {env.states_agent_left}")

# Check where car should end up
print("\n" + "=" * 60)
print("WHERE IS THE GOAL?")
print("=" * 60)
print(f"Car starts at origin: {car_agent.origin}")

# The car's goal might be to return to origin, or go to opposite corner
# Let's check the map to understand the layout
if hasattr(env, 'map') or hasattr(car_agent, 'map'):
    car_map = car_agent.map if hasattr(car_agent, 'map') else env.map
    print(f"Map shape: {np.array(car_map).shape if hasattr(car_map, '__len__') else 'unknown'}")




import numpy as np
from ADS_Environment import Environment

env = Environment(weights=None)
car_agent = env.agents[0]

print("=" * 60)
print("FINDING THE GOAL")
print("=" * 60)

# Check all valid car positions
print("Checking all valid car positions for success condition:")
for pos_idx in env.states_agent_left:
    state = env.translate_state([pos_idx, 0, 0])
    env.reset(state[0], [0, 0], [0, 0])
    
    success = car_agent.succeeds
    if success:
        print(f"✓ GOAL FOUND at position {pos_idx} (grid {state[0]})")
        break
else:
    print("Goal not found in any single position check")
    print("\nLet's check if goal requires a specific action sequence...")
    
    # Try moving from origin
    print(f"\nStarting from origin: position 43, grid {car_agent.origin}")
    
    # Try a path: 43 → 44 → 50 → 51 (moving right)
    for target_pos in [44, 50, 51]:
        state = env.translate_state([target_pos, 0, 0])
        env.reset(state[0], [0, 0], [0, 0])
        success = car_agent.succeeds
        print(f"  Position {target_pos} (grid {state[0]}): succeeds = {success}")

# Check if there's a time limit or other completion criteria
print("\n" + "=" * 60)
print("CHECKING DONE CONDITION IN ENVIRONMENT")
print("=" * 60)

# Start from position 43 (origin) and try to complete an episode
env.reset(env.translate_state([43, 45, 31])[0], 
          env.translate_state([43, 45, 31])[1], 
          env.translate_state([43, 45, 31])[2])

# Try going right repeatedly (might be trying to exit the map?)
print("Trying to move right from origin repeatedly:")
for step in range(10):
    # Action that moves right
    next_state, reward, done = env.step([0])  # Try action 0
    print(f"Step {step+1}: pos={next_state[0]}, done={done[0]}, succeeds={car_agent.succeeds}")
    
    if done[0]:
        print(f"✓ Episode completed at step {step+1}!")
        break

import numpy as np
from ADS_Environment import Environment

env = Environment(weights=None)

# Load the lexA policy
policy = np.load('policies/CHVI_lex_pedestrian_priority-policy.npy')

print("=" * 60)
print("TESTING FULL EPISODE WITH LEX-A POLICY")
print("=" * 60)

# Start from origin
car_pos = 43
p1_pos = 45
p2_pos = 31

state_translated = env.translate_state([car_pos, p1_pos, p2_pos])
env.reset(state_translated[0], state_translated[1], state_translated[2])

print(f"Starting at car={car_pos} (grid {state_translated[0]})")
print(f"Car origin: {env.agents[0].origin}\n")

positions_visited = [car_pos]

for step in range(50):
    action = policy[car_pos, p1_pos, p2_pos]
    
    print(f"Step {step+1}: pos={car_pos}, action={action}")
    
    next_state, reward, done = env.step([action])
    
    car_pos, p1_pos, p2_pos = next_state
    positions_visited.append(car_pos)
    
    print(f"  → moved to pos={car_pos} (grid {env.agents[0].position})")
    print(f"  done={done[0]}, succeeds={env.agents[0].succeeds}")
    
    if done[0]:
        print(f"\n✓ Episode completed in {step+1} steps!")
        print(f"Final position: {car_pos} (grid {env.agents[0].position})")
        break
    
    if car_pos == positions_visited[-2]:  # Stuck
        print(f"\n✗ Car stuck at position {car_pos}")
        break
else:
    print(f"\n✗ Episode didn't complete in 50 steps")
    print(f"Positions visited: {positions_visited}")


import numpy as np
from ADS_Environment import Environment

# Load both policies
policy_lgvi = np.load('policies/LGVI_pedestrian_priority-policy.npy')
policy_lgchvi = np.load('policies/CHVI_lex_pedestrian_priority-policy.npy')

# Load Q-values
Q_lgvi = np.load('policies/LGVI_pedestrian_priority-Q.npy', allow_pickle=True)
Q_hulls = np.load('policies/CHVI_qhulls.npy', allow_pickle=True).item()

state = (30, 31, 17)

print("=" * 60)
print("COMPARING LG_VI vs LG_CH_VI at state (30, 31, 17)")
print("=" * 60)

print("\nLG_VI Q-vectors:")
if isinstance(Q_lgvi, dict):
    q_vecs = Q_lgvi[state] if state in Q_lgvi else Q_lgvi.item()[state]
else:
    q_vecs = Q_lgvi[state[0], state[1], state[2]]

for a in range(6):
    print(f"  Action {a}: {q_vecs[a]}")

print(f"\nLG_VI chosen action: {policy_lgvi[state[0], state[1], state[2]]}")

print("\n" + "-" * 60)

print("\nLG_CH_VI Q-hulls (showing best vector in each hull):")
for a in range(6):
    hull = Q_hulls[(state[0], state[1], state[2], a)]
    # Find vector with max r_ped1
    max_ped1_idx = np.argmax(hull[:, 1])
    best_vec = hull[max_ped1_idx]
    print(f"  Action {a}: {best_vec} (best r_ped1 in hull)")
    
    # Also show max r_car
    max_car_idx = np.argmax(hull[:, 0])
    if max_car_idx != max_ped1_idx:
        print(f"           {hull[max_car_idx]} (best r_car in hull)")

print(f"\nLG_CH_VI chosen action: {policy_lgchvi[state[0], state[1], state[2]]}")


import numpy as np
from ADS_Environment import Environment

env = Environment(weights=None)

state = (30, 31, 17)
state_translated = env.translate_state(list(state))
env.reset(state_translated[0], state_translated[1], state_translated[2])

print("=" * 60)
print("CHECKING PEDESTRIAN POSITIONS")
print("=" * 60)
print(f"Car at position {state[0]} → grid {state_translated[0]}")
print(f"Ped1 at position {state[1]} → grid {state_translated[1]}")
print(f"Ped2 at position {state[2]} → grid {state_translated[2]}")

print("\n" + "=" * 60)
print("SIMULATING EACH ACTION")
print("=" * 60)

for action in range(6):
    env.reset(state_translated[0], state_translated[1], state_translated[2])
    
    next_state, reward, done = env.step([action])
    
    print(f"\nAction {action}:")
    print(f"  Car moves to position {next_state[0]} → grid {env.agents[0].position}")
    print(f"  Ped1 moves to position {next_state[1]} → grid {env.agents[1].position}")
    print(f"  Ped2 moves to position {next_state[2]} → grid {env.agents[2].position}")
    print(f"  Rewards: {reward}")
    print(f"  r_ped1 = {reward[1]:.1f}, r_ped2 = {reward[2]:.1f}")