from itertools import permutations
import numpy as np
from tqdm import tqdm


# LEX_HULL(q_vectors, tol)
# extreure objectius i fer permutacions possibles ic calucalr lexmax associat
def lex_max(q_vectors, priority=[0,1,2], tol=1e-9):
    """
    Lexicographic maximisation implementing Eq. (5) from Vamplew et al. (2021).
    
    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        priority: list specifying lexicographic order of objectives
                 e.g., [0,1,2] for car first, [1,2,0] for pedestrian1 first,
                 [2,1,0] for pedestrian2 first
                 Defaults to [0,1,2]
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
            if abs(obj_values[i] - max_val) < 1e-9:
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



def lex_hull(q_vectors, n_objectives=3, tol=1e-9):
    """
    Calculate lexicographic max for all possible lexicographic orders.
    
    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        n_objectives: number of objectives (default 3)
        tol: tolerance for floating point comparisons
    
    Returns:
        Dictionary mapping priority orders (as tuples) to their optimal actions
        
    """

    priority_orders = generate_all_priority_orders(n_objectives)

    lex_optimal_actions = {}

    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        order_tuple = tuple(order) # from [0,1,2] to (0,1,2)
        lex_optimal_actions[order_tuple] = best_action
    
    return lex_optimal_actions