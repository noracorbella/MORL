import pickle
import numpy as np
from CH_VI_stochastic import get_initial_state_hull
from CH_operations import max_q_value


VALUE_ITERATION = 1
CONVEX_HULL_VI = 2 


if __name__ == "__main__":

    algorithm_used = CONVEX_HULL_VI

    if algorithm_used == VALUE_ITERATION:

        weights_list = [
            [0, 10, 100],
            [0, 100, 10],
            [10, 0, 100],
            [10, 100, 0],
            [100, 0, 10],
            [100, 10, 0]
        ]

        initial_state = [43, 31, 45] 

        print("Results value iteration:")
        for i, weights in enumerate(weights_list, 1):
            v_file = f"policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
            
            with open(v_file, 'rb') as f:
                V = pickle.load(f)
            
            c, p1, p2 = initial_state
            initial_value = V[c, p1, p2]
            
            print(f"{i}. weights: {weights} -> Value initial state: {initial_value}")
    
    else: # algorithm_used == CONVEX_HULL_VI
        q_hulls_file = "policies/CH_VI_qhulls.pkl"

        with open(q_hulls_file, 'rb') as f:
            Q_hulls = pickle.load(f)

        initial_state = [43, 31, 45] 
        n_actions = 6

        value_hull = get_initial_state_hull(Q_hulls, initial_state, n_actions)

        print("Results Convex Hull Value Iteration:")
        print(f"Initial state value hull has {len(value_hull)} vertices:\n")

        for i, vector in enumerate(value_hull):
            print(f"[{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]")

        print("\n" + "-"*60)
        print("Value for specific weight combinations:")
        print("-"*60)

        weights_list = [
            [0, 10, 100],
            [0, 100, 10],
            [10, 0, 100],
            [10, 100, 0],
            [100, 0, 10],
            [100, 10, 0]
        ]

        for i, weights in enumerate(weights_list, 1):
            weights_normalized = np.array(weights, dtype=float)
            weights_normalized = weights_normalized / np.sum(weights_normalized)
            
            value = max_q_value(weights_normalized, value_hull)
            
            print(f"{i}. weights: {weights} -> Value initial state: {value:.2f}")

