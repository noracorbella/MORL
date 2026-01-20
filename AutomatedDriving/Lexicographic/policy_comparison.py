import numpy as np
from tqdm import tqdm

def extract_V_from_Q(policy, Q):
    """
    Extract value function V from Q-table given a policy.
    Args:
        policy: policy array [n_cells, n_cells, n_cells]
        Q: Q-table [n_cells, n_cells, n_cells, n_actions, n_objectives]

    Returns:
        V: value function [n_cells, n_cells, n_cells, n_objectives]
    """
    n_cells = policy.shape[0]
    n_objectives = Q.shape[-1]

    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])

    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                action = policy[c, p1, p2]
                V[c, p1, p2] = Q[c, p1, p2, action]

    return V

def compare_policies(env, policy1, policy2, Q1, Q2, initial_state, discount_factor=0.7):
    """
    Compare two policies by computing their value functions and checking
    if they yield the same value at the initial state.

    Args:
        env: the environment
        policy1: first policy
        policy2: second policy
        Q1: Q-table for first policy
        Q2: Q-table for second policy
        initial_state: initial state [c, p1, p2]
        discount_factor: discount factor

    Returns:
        tuple: (are_identical, V1_initial, V2_initial, V1, V2)
    """
    V1 = extract_V_from_Q(policy1, Q1)
    V2 = extract_V_from_Q(policy2, Q2)

    # values at initial state
    c, p1, p2 = initial_state
    V1_initial = V1[c, p1, p2]
    V2_initial = V2[c, p1, p2]

    print(f"\nV^pi_1({initial_state}) = {V1_initial}")
    print(f"V^pi_2({initial_state}) = {V2_initial}")

    are_identical = np.allclose(V1_initial, V2_initial, atol=1e-6)

    if are_identical:
        print(" Policies are IDENTICAL (same value at initial state)")
    else:
        print(" Policies are DIFFERENT")
        diff = V1_initial - V2_initial
        print(f"  Difference: {diff}")

    return are_identical, V1_initial, V2_initial, V1, V2


def compare_all_states(V1, V2, tol=1e-6):
    """
    Compare two value functions across all states.

    Args:
        V1: first value function
        V2: second value function
        tol: tolerance for comparison

    Returns:
        dict with comparison statistics
    """
    n_cells = V1.shape[0]

    total_states = 0
    identical_states = 0
    different_states = []

    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                total_states += 1

                if np.allclose(V1[c, p1, p2], V2[c, p1, p2], atol=tol):
                    identical_states += 1
                else:
                    different_states.append({
                        'state': [c, p1, p2],
                        'V1': V1[c, p1, p2],
                        'V2': V2[c, p1, p2],
                        'diff': V1[c, p1, p2] - V2[c, p1, p2]
                    })

    return {
        'total_states': total_states,
        'identical_states': identical_states,
        'different_states': different_states,
        'percentage_identical': 100 * identical_states / total_states
    }


if __name__ == "__main__":
    from ADS_Environment import Environment

    env = Environment()

    # priorities to compare
    priority1 = [0,1,2]
    priority2 = [2,1,0]

    priority1_str = f"{priority1[0]}-{priority1[1]}-{priority1[2]}"
    priority2_str = f"{priority2[0]}-{priority2[1]}-{priority2[2]}"

    policy1 = np.load(f"policies/LGVI_v2_{priority1_str}_priority-policy.npy")
    policy2 = np.load(f"policies/LGVI_v2_{priority2_str}_priority-policy.npy")

    Q1 = np.load(f"policies/LGVI_v2_{priority1_str}_priority-Q.npy")
    Q2 = np.load(f"policies/LGVI_v2_{priority2_str}_priority-Q.npy")

    V1 = extract_V_from_Q(policy1, Q1)
    V2 = extract_V_from_Q(policy2, Q2)

    stats = compare_all_states(V1, V2)

    # initial_state = [0,0,0]

    # are_identical, V1_init, V2_init, V1, V2 = compare_policies(env, policy1, policy2, Q1, Q2, initial_state)

    stats = compare_all_states(V1, V2)

    if stats['different_states']:
        print("\nExample different states (first 10):")
        for i, diff_state in enumerate(stats['different_states'][:10]):
            print(f"\n  {i+1}. State {diff_state['state']}: ")
            print(f"     V1 = {diff_state['V1']}")
            print(f"     V2 = {diff_state['V2']}")
            print(f"     Diff = {diff_state['diff']}")
    else:
        print("\n All states have identical values!")
        print("  The two lexicographic orderings produce functionally equivalent policies.")


    different_actions = 0
    total_states = policy1.size

    for c in range(policy1.shape[0]):
        for p1 in range(policy1.shape[1]):
            for p2 in range(policy1.shape[2]):
                if policy1[c, p1, p2] != policy2[c, p1, p2]:
                    different_actions += 1

    print(f"States with different actions: {different_actions} out of {total_states}")
    print(f"Percentage with different actions: {100 * different_actions / total_states:.2f}%")





