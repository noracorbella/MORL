import numpy as np
from LG_VI_stoc_lexmax_DEBUG import LG_VI_lexmax
from LG_utils import generate_all_priority_orders
from ADS_Environment import Environment

def main():
    """
    Run LG_VI_lexmax for all lexicographic priority orders.
    Each priority order is trained separately.
    """
    
    print("#" * 80)
    print("LEXICOGRAPHIC MAX VALUE ITERATION")
    print("#" * 80)
    
    # Parameters
    theta = 1.0
    discount_factor = 0.7
    n_objectives = 3
    
    # Get all priority orders
    all_priority_orders = generate_all_priority_orders(n_objectives)
    
    print(f"\nTraining {len(all_priority_orders)} separate policies")
    print(f"Priority orders to train: {all_priority_orders}")
    print(f"\nParameters:")
    print(f"  Theta: {theta}")
    print(f"  Discount factor: {discount_factor}")
    
    # Storage for policies and Q-tables
    policies = {}
    Q_tables = {}
    
    # Train each priority order separately
    for i, priority_order in enumerate(all_priority_orders):
        print("\n" + "=" * 80)
        print(f"TRAINING {i+1}/{len(all_priority_orders)}: Priority order {priority_order}")
        print("=" * 80)
        
        # Create fresh environment for each training
        env = Environment()
        
        # Train
        policy, Q = LG_VI_lexmax(
            env=env,
            theta=theta,
            discount_factor=discount_factor,
            priority=priority_order
        )
        
        # Store results
        priority_tuple = tuple(priority_order)
        policies[priority_tuple] = policy
        Q_tables[priority_tuple] = Q
        
        print(f"\nCompleted training for priority {priority_order}")
    
#     # ========================================================================
#     # RESULTS SUMMARY
#     # ========================================================================
#     print("\n" + "#" * 80)
#     print("RESULTS SUMMARY")
#     print("#" * 80)
    
#     # Policy statistics
#     print(f"\nPolicies trained: {len(policies)}")
    
#     # Example policies
#     car_priority = (0, 1, 2)
#     ped1_priority = (1, 0, 2)
#     ped2_priority = (2, 0, 1)
    
#     print(f"\nSample policies:")
#     print(f"  Car-priority {car_priority}: shape {policies[car_priority].shape}")
#     print(f"  Ped1-priority {ped1_priority}: shape {policies[ped1_priority].shape}")
#     print(f"  Ped2-priority {ped2_priority}: shape {policies[ped2_priority].shape}")
    
#     # ========================================================================
#     # POLICY COMPARISON
#     # ========================================================================
#     print("\n" + "-" * 80)
#     print("POLICY COMPARISON")
#     print("-" * 80)
    
#     # Compare car-priority vs ped1-priority
#     car_policy = policies[car_priority]
#     ped1_policy = policies[ped1_priority]
    
#     different_states = np.sum(car_policy != ped1_policy)
#     total_states = car_policy.size
    
#     print(f"\nCar-priority vs Ped1-priority:")
#     print(f"  States with different actions: {different_states}/{total_states}")
#     print(f"  Percentage different: {100 * different_states / total_states:.2f}%")
    
#     # Example state
#     example_state = [2, 3, 3]
#     print(f"\nExample state {example_state}:")
#     print(f"  Car-priority action: {car_policy[tuple(example_state)]}")
#     print(f"  Ped1-priority action: {ped1_policy[tuple(example_state)]}")
    
#     # Show Q-values for car-priority at example state
#     print(f"\n  Q-values (car-priority):")
#     Q_car = Q_tables[car_priority]
#     for action in range(Q_car.shape[3]):
#         q_vals = Q_car[tuple(example_state)][action]
#         print(f"    Action {action}: {q_vals}")
    
#     # ========================================================================
#     # PAIRWISE POLICY DIFFERENCES
#     # ========================================================================
#     print("\n" + "-" * 80)
#     print("PAIRWISE POLICY DIFFERENCES")
#     print("-" * 80)
    
#     priority_list = list(policies.keys())
#     n_priorities = len(priority_list)
    
#     print(f"\nNumber of different states between each pair of policies:")
#     print(f"(out of {total_states} total states)\n")
    
#     # Print header
#     header = "         "
#     for i, p in enumerate(priority_list):
#         header += f"{i:7}"
#     print(header)
#     print("         " + "-" * (7 * n_priorities))
    
#     # Print matrix
#     for i, p1 in enumerate(priority_list):
#         row = f"{i} {str(p1):6} |"
#         for j, p2 in enumerate(priority_list):
#             if i == j:
#                 row += "   -   "
#             else:
#                 diff = np.sum(policies[p1] != policies[p2])
#                 row += f"{diff:6} "
#         print(row)
    
#     print("\nPriority order indices:")
#     for i, p in enumerate(priority_list):
#         print(f"  {i}: {p}")
    
#     # ========================================================================
#     # SAVE POLICIES
#     # ========================================================================
#     print("\n" + "-" * 80)
#     print("SAVING POLICIES")
#     print("-" * 80)
    
#     for priority_order, policy in policies.items():
#         filename = f"policy_lexmax_{'_'.join(map(str, priority_order))}.npy"
#         np.save(filename, policy)
#         print(f"  Saved: {filename}")
    
#     # Save Q-tables
#     for priority_order, Q in Q_tables.items():
#         filename = f"Q_lexmax_{'_'.join(map(str, priority_order))}.npy"
#         np.save(filename, Q)
#         print(f"  Saved: {filename}")
    
#     print("\n" + "#" * 80)
#     print("TRAINING COMPLETE")
#     print("#" * 80)
    
#     return policies, Q_tables


if __name__ == "__main__":
    policies, Q_tables = main()