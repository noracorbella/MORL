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