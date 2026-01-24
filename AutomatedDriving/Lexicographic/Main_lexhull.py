import numpy as np
from LG_VI_stoc_lexhull import LG_VI_lexhull
from LG_VI_stoc_lexmax import LG_VI  # for comparison
from ADS_Environment import Environment

def main():
    env = Environment()
    
    print("#" * 80)
    print("LEXICOGRAPHIC HULL VALUE ITERATION")
    print("#" * 80)
    
    # Run lex_hull VI - trains ONCE and extracts ALL policies
    policies_all, Q = LG_VI_lexhull(
        env=env,
        theta=1.0,
        discount_factor=0.7
    )
    
    print("\n" + "#" * 80)
    print("RESULTS")
    print("#" * 80)
    
    # Access specific policies
    car_priority = (0, 1, 2)      # Car first, then ped1, then ped2
    ped1_priority = (1, 0, 2)     # Ped1 first, then car, then ped2
    ped2_priority = (2, 0, 1)     # Ped2 first, then car, then ped1
    
    print(f"\nAvailable priority orders: {len(policies_all)}")
    print("\nSample policies extracted:")
    print(f"  Car-priority {car_priority}: shape {policies_all[car_priority].shape}")
    print(f"  Ped1-priority {ped1_priority}: shape {policies_all[ped1_priority].shape}")
    print(f"  Ped2-priority {ped2_priority}: shape {policies_all[ped2_priority].shape}")
    
    # Compare policies
    print("\n" + "-" * 80)
    print("POLICY COMPARISON")
    print("-" * 80)
    
    # Check how many states have different actions between two priority orders
    car_policy = policies_all[car_priority]
    ped1_policy = policies_all[ped1_priority]
    
    different_states = np.sum(car_policy != ped1_policy)
    total_states = car_policy.size
    
    print(f"\nCar-priority vs Ped1-priority:")
    print(f"  States with different actions: {different_states}/{total_states}")
    print(f"  Percentage different: {100 * different_states / total_states:.2f}%")
    
    # Example: check action at a specific state
    example_state = [2, 3, 3]  # car at 2, both peds at crosswalk
    print(f"\nExample state {example_state}:")
    print(f"  Car-priority action: {car_policy[tuple(example_state)]}")
    print(f"  Ped1-priority action: {ped1_policy[tuple(example_state)]}")
    print(f"  Q-values: {Q[tuple(example_state)]}")
    
    # Save policies
    print("\n" + "-" * 80)
    print("SAVING POLICIES")
    print("-" * 80)
    
    for priority_order, policy in policies_all.items():
        filename = f"policy_lex_{'_'.join(map(str, priority_order))}.npy"
        np.save(filename, policy)
        print(f"  Saved: {filename}")
    
    # Save Q-table
    np.save("Q_lexhull.npy", Q)
    print(f"  Saved: Q_lexhull.npy")  
    
    return policies_all, Q


def analyse_policy_differences(policies_all):
    """
    Analyze how different lexicographic orders lead to different policies.
    """
    print("\n" + "=" * 80)
    print("POLICY DIFFERENCE ANALYSIS")
    print("=" * 80)
    
    priority_orders = list(policies_all.keys())
    n_orders = len(priority_orders)
    
    # Create matrix of policy differences - use float to allow np.inf
    diff_matrix = np.zeros((n_orders, n_orders), dtype=float)
    
    for i, order1 in enumerate(priority_orders):
        for j, order2 in enumerate(priority_orders):
            if i < j:
                diff = np.sum(policies_all[order1] != policies_all[order2])
                diff_matrix[i, j] = diff
                diff_matrix[j, i] = diff
    
    # Find most different pairs
    max_diff = np.max(diff_matrix)
    max_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
    
    print(f"\nMost different policy pair:")
    print(f"  {priority_orders[max_idx[0]]} vs {priority_orders[max_idx[1]]}")
    print(f"  Different states: {int(max_diff)}")
    
    # Find most similar pairs (excluding identical)
    diff_matrix_masked = diff_matrix.copy()
    np.fill_diagonal(diff_matrix_masked, np.inf)
    min_diff = np.min(diff_matrix_masked)
    min_idx = np.unravel_index(np.argmin(diff_matrix_masked), diff_matrix_masked.shape)
    
    print(f"\nMost similar policy pair:")
    print(f"  {priority_orders[min_idx[0]]} vs {priority_orders[min_idx[1]]}")
    print(f"  Different states: {int(min_diff)}")
    
    # Print full difference matrix
    print(f"\n" + "-" * 80)
    print("Full pairwise difference matrix:")
    print(f"(Number of states with different actions)")
    print(f"\nPriority orders:")
    for i, order in enumerate(priority_orders):
        print(f"  {i}: {order}")
    
    print(f"\nDifference matrix:")
    # Create nice formatted output
    header = "     " + "".join([f"{i:7}" for i in range(n_orders)])
    print(header)
    for i in range(n_orders):
        row = f"{i}:  "
        for j in range(n_orders):
            if i == j:
                row += "   -   "
            else:
                row += f"{int(diff_matrix[i, j]):6} "
        print(row)
    
    return diff_matrix


if __name__ == "__main__":
    policies, Q = main()
    analyse_policy_differences(policies)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)