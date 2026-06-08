from itertools import permutations
import numpy as np
from tqdm import tqdm


def lex_max(q_vectors, priority=[0, 1, 2], tol=1e-9):
    """
    Lexicographic maximisation implementing Eq. (5) from Vamplew et al. (2021).
        """

    n_actions = q_vectors.shape[0]

    if isinstance(priority, np.ndarray):
        priority = priority.tolist()

    best_actions = list(range(n_actions)) # initialisation, all actions are candidates
    
    for obj_idx in priority:
        if len(best_actions) == 1:
            break

        # get q values of this objective for all possible actions
        obj_values = [q_vectors[a, obj_idx] for a in best_actions]
        # keep actions that maximise this objective
        max_val = np.max(obj_values)


        new_best_actions = []
        for i, action in enumerate(best_actions):
            if abs(obj_values[i] - max_val) < tol:
                new_best_actions.append(action)
        
        best_actions = new_best_actions

        if len(best_actions) == 0:
            print(f"WARNING: No actions left! Returning 0")
            return 0
    # return the first reamaining action I VECTOR ASSOCIAT
    return best_actions[0]


def generate_all_priority_orders(n_objectives):

    all_permutations = list(permutations(range(n_objectives)))

    priority_orders = [list(perm) for perm in all_permutations]

    return priority_orders



def lex_hull(env, q_vectors, tol=1e-9):
    """
    Calculate lexicographic max for all possible lexicographic orders.
    
    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        n_objectives: number of objectives (2 for DST)
        tol: tolerance for floating point comparisons
    
    Returns:
        Dictionary mapping priority orders (as tuples) to their optimal actions
        
    """
    n_objectives = env.n_rewards

    priority_orders = generate_all_priority_orders(n_objectives)

    lex_optimal_actions = {}

    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        order_tuple = tuple(order) # from [0,1] to (0,1)
        lex_optimal_actions[order_tuple] = best_action
    
    return lex_optimal_actions

def lex_hull_corrected(env, q_vectors, tol=1e-9):
    """
    Calculate lexicographic hull: keep only q-vectors that are lexicographically 
    optimal for at least one priority order.
    
    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        n_objectives: number of objectives (default 2 for DST)
        tol: tolerance for floating point comparisons
    
    Returns:
        Dictionary mapping priority orders (as tuples) to their optimal action indices
        Set of action indices that are optimal for at least one priority order
    """
    n_objectives = env.n_rewards
    priority_orders = generate_all_priority_orders(n_objectives)
    
    lex_optimal_actions = {}
    optimal_action_set = set()  # Track which actions are optimal for ANY order
    
    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        order_tuple = tuple(order)
        lex_optimal_actions[order_tuple] = best_action
        optimal_action_set.add(best_action)
    
    return lex_optimal_actions, optimal_action_set

def lex_hull_corrected_fast(priority_orders, q_vectors, tol=1e-9):
    """
    Same as lex_hull_corrected but accepts precomputed priority orders
    instead of generating them internally. Call generate_all_priority_orders
    once before the training loop and reuse here.
    """
    
    lex_optimal_actions = {}
    optimal_action_set = set()  # Track which actions are optimal for ANY order
    
    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        lex_optimal_actions[tuple(order)] = best_action
        optimal_action_set.add(best_action)
    return lex_optimal_actions, optimal_action_set

def extract_lex_policy_from_qhulls(Q_hulls, priority, env, n_actions):
    """Extract lex policy from CHVI Q-hulls. Policy shape [5,3,3,2,2,2]."""
    size   = env.size
    policy = np.zeros([size, 3, 3, 2, 2, 2], dtype=int)
    for state in env.non_terminal_states:
        action_best_vectors = []
        for action in range(n_actions):
            hull     = Q_hulls[(*state, action)]
            if not isinstance(hull, np.ndarray):
                hull = np.array(hull)
            best_idx = lex_max(hull, priority=priority)
            action_best_vectors.append(hull[best_idx])
        action_best_vectors = np.array(action_best_vectors)
        policy[state] = lex_max(action_best_vectors, priority=priority)
    return policy





if __name__ == "__main__":


    q_vectors = np.array([[0.83, -7.30],[0.83, -17.30],[-1.91, 0.00],[-2.32, 0.00],[-0.01, -16.08], [-2.32, 0.00]])

    lex_optimal_actions, optimal_action_set = lex_hull_corrected(q_vectors)
    print("lex_optimal_actions", lex_optimal_actions)
    print("\noptimal_action_set", optimal_action_set)