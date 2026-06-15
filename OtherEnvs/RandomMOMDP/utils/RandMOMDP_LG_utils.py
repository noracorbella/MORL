from itertools import permutations
import numpy as np


def lex_max(q_vectors, priority=[0, 1], tol=1e-9):
    """
    Lexicographic maximisation implementing Eq. (5) from Vamplew et al. (2021).

    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        priority: list specifying lexicographic order of objectives
        tol: tolerance

    Returns:
        Index of the lexicographically best action
    """

    n_actions = q_vectors.shape[0]

    if isinstance(priority, np.ndarray):
        priority = priority.tolist()

    objective_order = priority

    best_actions = list(range(n_actions)) # initialisation, all actions are valid candidates

    for obj_idx in objective_order:
        if len(best_actions) == 1:
            break

        # get q values of this objective for all possible candidate actions
        obj_values = [q_vectors[action, obj_idx] for action in best_actions]
        # keep candidate actions that maximise this objective
        max_val = np.max(obj_values)


        new_best_actions = []
        for i, action in enumerate(best_actions):
            if abs(obj_values[i] - max_val) < tol:
                new_best_actions.append(action)
        
        best_actions = new_best_actions

        if len(best_actions) == 0:
            print(f"WARNING: No actions left! Returning 0")
            return
    # return the first remaining action candidate
    return best_actions[0]


def generate_all_priority_orders(n_objectives):

    all_permutations = list(permutations(range(n_objectives)))

    priority_orders = [list(perm) for perm in all_permutations]

    return priority_orders


def lex_hull(env, q_vectors, tol=1e-9):
    """
    Calculate lexicographic max for all possible lexicographic orders.

    Args:
        env: environment
        q_vectors: array of shape (n_actions, n_objectives)
        tol: tolerance

    Returns:
        Dictionary mapping priority orders (as tuples) to their optimal action indices
    """
    n_objectives = env.n_rewards

    priority_orders = generate_all_priority_orders(n_objectives)

    lex_optimal_actions = {}

    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        order_tuple = tuple(order)
        lex_optimal_actions[order_tuple] = best_action

    return lex_optimal_actions

def lex_hull_corrected(env, q_vectors, tol=1e-9, priority_orders=None):
    """
    Calculate lexicographic hull: keep only q-vectors that are lexicographically
    optimal for at least one priority order.

    Args:
        env: environment
        q_vectors: array of shape (n_actions, n_objectives)
        n_objectives: number of objectives
        tol: tolerance
    Returns:
        Dictionary mapping priority orders (as tuples) to their optimal action indices
        Set of action indices that are optimal for at least one priority order
    """
    n_objectives = env.n_rewards
    if priority_orders is None:
        priority_orders = generate_all_priority_orders(n_objectives)

    lex_optimal_actions = {}
    optimal_action_set = set()  # Track which action indices are optimal for ANY order

    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        order_tuple = tuple(order)
        lex_optimal_actions[order_tuple] = best_action
        optimal_action_set.add(best_action)

    return lex_optimal_actions, optimal_action_set


def extract_lex_policy_from_qhulls(Q_hulls, priority, env, n_actions):
    """
    Extract the optimal lexicographic policy for a given priority order
    directly from the CHVI Q-hulls using lex_max.

    Args:
        Q_hulls : dict keyed by (state, action) -> hull (Q-hulls from convexhull_VI)
        priority : list (lexicographic priority order)
        env : Environment
        n_actions: int
    Returns:
        policy : np.ndarray (n_states,)
    """
    policy = np.full(env.n_states, -1, dtype=int)

    for state in env.valid_states:

        valid_actions = env.valid_actions(state)
        if len(valid_actions) == 0:
            continue

        # For each available action pick the best hull vector with lex max
        action_best_vectors = []
        for action in valid_actions:
            hull = Q_hulls[(state, action)]
            if not isinstance(hull, np.ndarray):
                hull = np.array(hull)
            best_idx = lex_max(hull, priority=priority)
            action_best_vectors.append(hull[best_idx])

        # Among best vectors, pick the best action
        action_best_vectors = np.array(action_best_vectors)
        best_action = lex_max(action_best_vectors, priority=priority)
        policy[state] = valid_actions[best_action]

    return policy


if __name__ == "__main__":

    q_vectors = np.array([[0.83, -7.30], [0.83, -17.30], [-1.91, 0.00],
                          [-2.32, 0.00], [-0.01, -16.08], [-2.32, 0.00]])

    lex_optimal_actions, optimal_action_set = lex_hull_corrected(q_vectors)
    print("lex_optimal_actions", lex_optimal_actions)
    print("\noptimal_action_set", optimal_action_set)
