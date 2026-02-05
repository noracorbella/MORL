import pickle
import numpy as np
from CH_operations import max_q_value
from LG_utils import lex_max

LG_VI_LEXMAX = 3
LG_VI_LEXHULL = 4

if __name__ == "__main__":

    algorithm_used = LG_VI_LEXHULL  

    initial_state = [43, 31, 45]  
    n_actions = 6 


    if algorithm_used == LG_VI_LEXMAX:
        priority_orders = [
            [0, 1, 2],  
            [0, 2, 1],  
            [1, 0, 2],  
            [1, 2, 0],  
            [2, 0, 1],  
            [2, 1, 0],  
        ]

        print("Results Lexicographic Value Iteration LEXMAX:")
        
        for i, priority in enumerate(priority_orders, 1):
            priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"
            v_file = f"policies/LG_VI_lexmax_{priority_str}_V_table.pkl"
            
            try:
                with open(v_file, 'rb') as f:
                    V = pickle.load(f)
                
                c, p1, p2 = initial_state
                initial_value_vector = V[c, p1, p2]
                
                print(f"   Value: [{initial_value_vector[0]:.2f}, "
                      f"{initial_value_vector[1]:.2f}, {initial_value_vector[2]:.2f}]")
                
            except FileNotFoundError:
                print(f"{i}. Priority: {priority} -> V table not found! Run training first.")

    elif algorithm_used == LG_VI_LEXHULL:
        v_hulls_file = "policies/LG_VI_lexhull_V_hulls.pkl"
        q_hulls_file = "policies/LG_VI_lexhull_Q_hulls.pkl"

        try:
            with open(v_hulls_file, 'rb') as f:
                V_hulls = pickle.load(f)

            with open(q_hulls_file, 'rb') as f:
                Q_hulls = pickle.load(f)

            initial_state = (43, 31, 45) 
            n_actions = 6

            initial_v_hull = V_hulls[initial_state]

            print(f"\nV hull for initial state has {len(initial_v_hull)} vertices:")

            for i, vector in enumerate(initial_v_hull):
                print(f"[{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]")

            print("\n" + "-"*60)
            print("Value for specific priority orders:")
            print("-"*60 + "\n")

            priority_orders = [
                [0, 1, 2], 
                [0, 2, 1], 
                [1, 0, 2], 
                [1, 2, 0], 
                [2, 0, 1], 
                [2, 1, 0], 
            ]

            for i, priority in enumerate(priority_orders, 1):
                q_vectors = []
                for action in range(n_actions):
                    action_hull = Q_hulls[initial_state + (action,)]
                    best_idx = lex_max(action_hull, priority=priority)
                    q_vectors.append(action_hull[best_idx])
                
                q_vectors = np.array(q_vectors)
                
                best_action = lex_max(q_vectors, priority=priority)
                best_value_vector = q_vectors[best_action]
                

                print(f"{priority}: [{best_value_vector[0]:.2f}, "
                      f"{best_value_vector[1]:.2f}, {best_value_vector[2]:.2f}]")

        except FileNotFoundError as e:
            print(f"Error: Could not find required files. Run training first.")
            print(f"Missing file: {e}")

    else:
        print("Invalid algorithm")