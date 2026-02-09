import pickle
import numpy as np
from CH_VI_stochastic_v2 import get_initial_state_hull
from CH_vector_extraction import get_initial_state_vector
from CH_operations import max_q_value
import os


VALUE_ITERATION = 1
CONVEX_HULL_VI = 2 
COMPARE_BOTH = 3  


if __name__ == "__main__":

    algorithm_used = VALUE_ITERATION  

    weights_list = [
        [0, 10, 100],
        [0, 100, 10],
        [10, 0, 100],
        [10, 100, 0],
        [100, 0, 10],
        [100, 10, 0]
    ]

    initial_state = [43, 31, 45] 
    n_actions = 6

    if algorithm_used == VALUE_ITERATION or algorithm_used == COMPARE_BOTH:
        print("#"*70)
        print("VALUE ITERATION RESULTS")
        print("#"*70)

        print("\nScalar values from V-tables:")
        print("-"*70)
        vi_results = []
        for i, weights in enumerate(weights_list, 1):
            v_file = f"policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl" 
            
            if not os.path.exists(v_file):
                print(f"{i}. weights: {weights} -> V-table not found")
                vi_results.append(None)
                continue

            with open(v_file, 'rb') as f:
                V = pickle.load(f)        # scalar v table
            
            c, p1, p2 = initial_state
            initial_value = V[c, p1, p2]
            
            print(f"{i}. weights: {weights} -> Scalar value: {initial_value:.3f}")
            vi_results.append({'scalar': initial_value})

        print("\n" + "#"*70)
        print("Vector decomposition from V-vector tables:")
        print("#"*70)
        
        has_vector_data = False
        for i, weights in enumerate(weights_list, 1):
            v_vector_file = f"policies/V_vector_{weights[0]}-{weights[1]}-{weights[2]}.pkl"

            if not os.path.exists(v_vector_file):
                continue
            
            has_vector_data = True
            with open(v_vector_file, 'rb') as f:
                V_vector = pickle.load(f)
            
            c, p1, p2 = initial_state
            vector_value = V_vector[c, p1, p2]

            weight_vect = np.array(weights)
            computed_scalar = np.dot(vector_value, weight_vect)

            if vi_results[i-1] is not None:
                vi_results[i-1]['vector'] = vector_value
                vi_results[i-1]['computed_scalar'] = computed_scalar

            v_file = f"policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
            if os.path.exists(v_file):
                with open(v_file, 'rb') as f:
                    V = pickle.load(f)
                stored_scalar = V[c, p1, p2]
                diff = abs(computed_scalar - stored_scalar)
            else:
                stored_scalar = None
                diff = None

            print(f"\n{i}. weights: {weights}")
            print(f"   Vector: [{vector_value[0]:.3f}, {vector_value[1]:.3f}, {vector_value[2]:.3f}]")
            print(f"   w·v = {computed_scalar:.3f}", end="")
            
            if stored_scalar is not None:
                print(f"  (stored: {stored_scalar:.3f}, diff: {diff:.6f})")
            else:
                print()

        if not has_vector_data:
            print("\nNo vector V-tables found.")
            print("Run: python vector_policy_evaluation.py")

        if algorithm_used == VALUE_ITERATION:
            exit(0)
        else:
            print("\n")

    if algorithm_used == CONVEX_HULL_VI or algorithm_used == COMPARE_BOTH:
        print("#"*70)
        print("CONVEX HULL VALUE ITERATION RESULTS")
        print("#"*70)
        
        q_hulls_file = "policies/CH_VI_qhulls.pkl"

        if not os.path.exists(q_hulls_file):
            print(f"\n Q-hulls file not found: {q_hulls_file}")
            print("Run Main.py with CONVEX_HULL_VI mode first.")
            if algorithm_used == CONVEX_HULL_VI:
                exit(1)
            else:
                chvi_results = [None] * len(weights_list)
        else:
            with open(q_hulls_file, 'rb') as f:
                Q_hulls = pickle.load(f)

            value_hull = get_initial_state_hull(Q_hulls, initial_state, n_actions)

            print(f"\nInitial state value hull has {len(value_hull)} vertices:")
            print("-"*70)
            for i, vector in enumerate(value_hull, 1):
                print(f"{i}. [{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]")

            print("\n" + "#"*70)
            print("Scalar values for specific weight combinations:")
            print("#"*70)
            chvi_results = []
            for i, weights in enumerate(weights_list, 1):
                weights_normalized = np.array(weights, dtype=float)
                weights_normalized = weights_normalized / np.sum(weights_normalized)
                
                value = max_q_value(weights_normalized, value_hull)
                
                print(f"{i}. weights: {weights} -> Scalar value: {value:.3f}")
                chvi_results.append({'scalar': value})

            print("\n" + "#"*70)
            print("Vector decomposition for specific weight combinations:")
            print("#"*70)
            for i, weights in enumerate(weights_list, 1):
                optimal_vector = get_initial_state_vector(Q_hulls, initial_state, weights, n_actions)
                
                weights_normalized = np.array(weights, dtype=float)
                weights_normalized = weights_normalized / np.sum(weights_normalized)
                scalar_value = np.dot(optimal_vector, weights_normalized)
                
                chvi_results[i-1]['vector'] = optimal_vector
                chvi_results[i-1]['computed_scalar'] = scalar_value
                
                print(f"\n{i}. weights: {weights}")
                print(f"   Vector: [{optimal_vector[0]:.3f}, {optimal_vector[1]:.3f}, {optimal_vector[2]:.3f}]")
                print(f"   w·v = {scalar_value:.3f}")

    if algorithm_used == COMPARE_BOTH:
        print("\n" + "#"*70)
        print("COMPARISON: VI vs CHVI")
        print("#"*70)

        all_match = True
        for i, weights in enumerate(weights_list):
            vi_res = vi_results[i] if i < len(vi_results) else None
            chvi_res = chvi_results[i] if i < len(chvi_results) else None

            print(f"\n{i+1}. weights: {weights}")
            
            if vi_res and 'vector' in vi_res:
                vi_vec = vi_res['vector']
                vi_scalar = vi_res['computed_scalar']
                print(f"VI vector: [{vi_vec[0]:.3f}, {vi_vec[1]:.3f}, {vi_vec[2]:.3f}] -> {vi_scalar:.3f}")
            else:
                print(f"VI vector: Not available")
                vi_vec = None

            if chvi_res and 'vector' in chvi_res:
                chvi_vec = chvi_res['vector']
                chvi_scalar = chvi_res['computed_scalar']
                print(f"   CHVI vector: [{chvi_vec[0]:.3f}, {chvi_vec[1]:.3f}, {chvi_vec[2]:.3f}] -> {chvi_scalar:.3f}")
            else:
                print(f"   CHVI vector: Not available")
                chvi_vec = None

            # Check match
            if vi_vec is not None and chvi_vec is not None:
                vec_diff = np.linalg.norm(vi_vec - chvi_vec)
                scalar_diff = abs(vi_res['computed_scalar'] - chvi_res['computed_scalar'])
                
                if vec_diff < 1e-3 and scalar_diff < 1e-3:
                    print(f"   -> Match (vec diff: {vec_diff:.6f}, scalar diff: {scalar_diff:.6f})")
                else:
                    print(f"   -> DIFFER (vec diff: {vec_diff:.6f}, scalar diff: {scalar_diff:.6f})")
                    all_match = False

        print("\n" + "#"*70)
        if all_match:
            print("All results match.")
        else:
            print("Some results differ.")
        print("#"*70)