from itertools import permutations
import numpy as np
from tqdm import tqdm


# LEX_HULL(q_vectors, tol)
# extreure objectius i fer permutacions possibles ic calucalr lexmax associat
def lex_max(q_vectors, priority=[0,1], tol=1e-9):
    """
    Lexicographic maximisation implementing Eq. (5) from Vamplew et al. (2021).
    
    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        priority: list specifying lexicographic order of objectives
        tol
    
    Returns:
        Index of the lexicographically best action
    """

    n_actions = q_vectors.shape[0]

    if isinstance(priority, np.ndarray):
        priority = priority.tolist()

    objective_order = priority

    best_actions = list(range(n_actions)) # initialisation, all actions are candidates
    
    for obj_idx in objective_order:
        if len(best_actions) == 1:
            break

        # get q values of this objective for all possible actions
        obj_values = [q_vectors[action, obj_idx] for action in best_actions]
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

def extract_lex_policy_from_qhulls(Q_hulls, priority, env, n_actions):
    """
    Extract the optimal lexicographic policy for a given priority order
    directly from the CHVI Q-hulls using lex_max.
    Args:
        Q_hulls : dict (Q-hulls from convexhull_VI)
        priority : list (lexicographic priority order)
        env : Environment
        n_actions: int
    Returns:
        policy : np.ndarray ([n_cells, n_cells, n_cells])
    """
    policy = np.zeros([env.n_rows, env.n_cols], dtype=int)

    for (row, col) in env.non_terminal_states:
        state_tuple = (row, col)

        # For each action pick the best hull vector with lex max
        action_best_vectors = []
        for action in range(n_actions):
            hull = Q_hulls[state_tuple + (action,)]
            if not isinstance(hull, np.ndarray):
                hull = np.array(hull)
            best_idx = lex_max(hull, priority=priority)
            action_best_vectors.append(hull[best_idx])

        # Among best vectors, pick the best action
        action_best_vectors = np.array(action_best_vectors)
        best_action = lex_max(action_best_vectors, priority=priority)
        policy[row, col]    = best_action

    return policy





if __name__ == "__main__":


    q_vectors = np.array([[0.83, -7.30],[0.83, -17.30],[-1.91, 0.00],[-2.32, 0.00],[-0.01, -16.08], [-2.32, 0.00]])

    lex_optimal_actions, optimal_action_set = lex_hull_corrected(q_vectors)
    print("lex_optimal_actions", lex_optimal_actions)
    print("\noptimal_action_set", optimal_action_set)