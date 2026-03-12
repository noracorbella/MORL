import numpy as np
import pickle
from CH_operations import get_hull



def get_optimal_vector_from_hull(weights, hull):
    """
    Extract the optimal VECTOR from a hull for given weights.
    
    In CHVI, the value hull contains multiple vectors. For a given weight vector,
    we want to find: v* = argmax_{v in hull} w·v
    
    This returns v* (instead of the scalar w·v*).
    
    :param weights: Weight vector [w_car, w_ped1, w_ped2], normalized or not
    :param hull: Array of vectors representing the hull
    :return: The optimal vector from the hull
    """
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)
    
    if not isinstance(hull, np.ndarray):
        hull = np.array(hull)
    
    if len(hull.shape) == 1:
        return hull
    
    if hull.shape[0] == 1:
        return hull[0]
    
    scalar_values = np.dot(hull, weights)
    
    best_idx = np.argmax(scalar_values)
    optimal_vector = hull[best_idx]
    
    return optimal_vector


def get_initial_state_vector(Q_hulls, initial_state_indices, weights, n_actions):
    """
    Get the optimal VALUE VECTOR for the initial state given specific weights.
    
    This is the CHVI equivalent of loading V_vector[c,p1,p2] in standard VI.
    
    Get value hull for initial state (union of all Q-hulls across actions)
    Extract the specific vector from that hull that's optimal for given weights
    
    :param Q_hulls: The Q_hulls dictionary from convexhull_VI
    :param initial_state_indices: [c, p1, p2] indices for the initial state
    :param weights: Weight vector [w_car, w_ped1, w_ped2]
    :param n_actions: Number of actions
    :return: Optimal vector [v_car, v_ped1, v_ped2] for this state and weights
    """
    c, p1, p2 = initial_state_indices
    
    all_vectors = []
    for action in range(n_actions):
        hull = Q_hulls[(c, p1, p2, action)]
        if isinstance(hull, np.ndarray):
            all_vectors.extend(hull)
        else:
            all_vectors.extend(list(hull))
    
    all_vectors = np.array(all_vectors) # all hull vectors for the 6 actions in initial state
    
    if len(all_vectors) > 1:
        value_hull = get_hull(all_vectors) # get convex hull with all vectors
    else:
        value_hull = all_vectors
    
    optimal_vector = get_optimal_vector_from_hull(weights, value_hull) # pick best vector from hull for these weights
    
    return optimal_vector


def extract_V_vectors_for_weights(Q_hulls, weights, env, n_actions):
    """
    Extract vector-valued V-table for specific weights from CHVI Q-hulls.
    
    Vector decomposition for ALL states.
    
    :param Q_hulls: The Q_hulls dictionary from convexhull_VI
    :param weights: Weight vector [w_car, w_ped1, w_ped2]
    :param env: Environment
    :param n_actions: Number of actions
    :return: V_vector array of shape [n_cells, n_cells, n_cells, 3] for all states
    """
    n_cells = env.map_num_cells
    V_vector = np.zeros([n_cells, n_cells, n_cells, 3])
    
    weights = np.array(weights, dtype=float)
    # weights = weights / np.sum(weights)
    
    
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                all_vectors = []
                for action in range(n_actions):
                    hull = Q_hulls[(c, p1, p2, action)]
                    if isinstance(hull, np.ndarray):
                        all_vectors.extend(hull)
                    else:
                        all_vectors.extend(list(hull))
                
                all_vectors = np.array(all_vectors)
                
                if len(all_vectors) > 1:
                    value_hull = get_hull(all_vectors)
                else:
                    value_hull = all_vectors
                
                optimal_vector = get_optimal_vector_from_hull(weights, value_hull)
                V_vector[c, p1, p2] = optimal_vector
    
    return V_vector


if __name__ == "__main__":
    q_hulls_file = "policies/CH_VI_qhulls.pkl"
    
    with open(q_hulls_file, 'rb') as f:
        Q_hulls = pickle.load(f)
    
    initial_state = [43, 31, 45]
    n_actions = 6
    
    weights_list = [
        [0, 10, 100],
        [0, 100, 10],
        [10, 0, 100],
        [10, 100, 0],
        [100, 0, 10],
        [100, 10, 0]
    ]
    
    print("#"*80)
    print("CONVEX HULL VI - VECTOR EXTRACTION")
    print("#"*80)
    
    for i, weights in enumerate(weights_list, 1):
        optimal_vector = get_initial_state_vector(Q_hulls, initial_state, weights, n_actions) # get vector only for initial state
        
        weights_norm = np.array(weights, dtype=float)
        weights_norm = weights_norm / np.sum(weights_norm)
        scalar_value = np.dot(optimal_vector, weights_norm)
        
        print(f"\n{i}. Weights: {weights}")
        print(f"Optimal vector: [{optimal_vector[0]:.3f}, {optimal_vector[1]:.3f}, {optimal_vector[2]:.3f}]")
        print(f"Scalar value (w·v): {scalar_value:.3f}")
    
    
    from ADS_Environment import Environment
    
    for i, weights in enumerate(weights_list, 1):
        print(f"\n{i}. Processing weights {weights}...")
        
        env = Environment(weights=weights)
        
        V_vector = extract_V_vectors_for_weights(Q_hulls, weights, env, n_actions) #get vectors for all states
        
        v_vector_file = f"policies/CHVI_V_vector_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
        with open(v_vector_file, 'wb') as f:
            pickle.dump(V_vector, f)
        print(f"   Saved to {v_vector_file}")
        
        c, p1, p2 = initial_state
        vector_value = V_vector[c, p1, p2]
        print(f"   Initial state vector: [{vector_value[0]:.3f}, {vector_value[1]:.3f}, {vector_value[2]:.3f}]")
    