import numpy as np
from ADS_Environment import Environment
from LG_VI_stochastic import LG_VI
from VI_Stochastic import value_iteration
from q_learning_algorithm import q_learning
from LG_CH_VI_stochastic import (
    LG_CH_VI, 
    extract_policy_for_weights,
    extract_lexicographic_policy,
    extract_lexP_policy,
    extract_lexA_policy
)
from auxiliary_functions import tester, test_policy_vectorial, tester_vect

Q_LEARNING = 0
VALUE_ITERATION = 1
CONVEX_HULL_VALUE_ITERATION = 2
LEXICOGRAPHIC_VALUE_ITERATION = 3


if __name__ == "__main__":
    algorithm_used = CONVEX_HULL_VALUE_ITERATION 
    
    # For weighted algorithms
    weights = [1.0, 100.0, 100.0]

    Test_with_vectorial = True  # Use vectorial rewards for testing
    Test_Episodes = 100
    Show_visualisation = True

    # For Q_LEARNING and VALUE_ITERATION
    Training = False  # True: train and save, False: load and test
    ql_policy_name = f"policies/QL_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    vi_policy_name = f"policies/VI_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"

    # For CONVEX_HULL_VALUE_ITERATION
    CHVI_Calculate_hulls = False  # True: compute hulls, False: extract policy
    CHVI_extraction_method = 'lexicographic'  # 'weighted' or 'lexicographic'
    CHVI_priority = 'pedestrian'  # Only for lexicographic extraction: 'car' or 'pedestrian'
    ch_hulls_name = "policies/CH_qhulls.npy"
    chvi_hulls_name = "policies/CHVI_qhulls.npy"
    if CHVI_extraction_method == 'weighted':
        chvi_policy_name = f"policies/CHVI_weighted_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    else:
        chvi_policy_name = f"policies/CHVI_lex_{CHVI_priority}_priority-policy.npy"
    
    # For LEXICOGRAPHIC_VALUE_ITERATION
    LG_Training = False  # True: train, False: load
    LG_priority = 'pedestrian'  # 'car' or 'pedestrian'
    lg_policy_name = f"policies/LGVI_{LG_priority}_priority-policy.npy"
    lg_Q_name = f"policies/LGVI_{LG_priority}_priority-Q.npy"


    # #########################################################################
    # Q-LEARNING
    # #########################################################################    
    
    if algorithm_used == Q_LEARNING:
        print("\n" + "#" * 80)
        print("ALGORITHM: Q-LEARNING")
        print("#" * 80)
        if Training:
            print(f"\nTraining with weights: {weights}")

            env = Environment(weights=weights)
            policy, q = q_learning(env, alpha=0.8, gamma=0.7)
                
            np.save(ql_policy_name, policy)
            print(f"\n Policy saved to {ql_policy_name}\n")

            print("-------------------")
            print("Training Finished!!!")
        
        else: # Testing
            policy = np.load(ql_policy_name)
        
        env = Environment(weights=weights)
        print("\nTesting policy...\n")

        if Test_with_vectorial:
            results = test_policy_vectorial(env, policy, num_episodes=Test_Episodes, verbose=True)
        
        if Show_visualisation:
            tester(env, policy, drawing=True)
            
    # ########################################################################
    # VALUE ITERATION
    # ########################################################################
    elif algorithm_used == VALUE_ITERATION:
        print("\n" + "#" * 80)
        print("ALGORITHM: VALUE ITERATION")
        print("#" * 80)
        
        if Training:
            print(f"\nTraining with weights: {weights}")
            
            env = Environment(weights=weights)
            policy, Q = value_iteration(env, theta=0.01, discount_factor=0.7)
            
            np.save(vi_policy_name, policy)
            print(f"\n Policy saved to {vi_policy_name}\n")
            
            print("-------------------")
            print("Training Finished!!!")
        else:
            policy = np.load(vi_policy_name)
        
        # Testing
        print("\nTesting policy...\n")
        env = Environment(weights=weights)
        
        if Test_with_vectorial:
            results = test_policy_vectorial(env, policy, num_episodes=Test_Episodes, verbose=True)
        
        if Show_visualisation:
            tester(env, policy, drawing=True)
    


    # #########################################################################
    # LEXICOGRAPHIC VALUE ITERATION
    # #########################################################################
    if algorithm_used == LEXICOGRAPHIC_VALUE_ITERATION:
        print("\n" + "#" * 80)
        print("ALGORITHM: LEXICOGRAPHIC VALUE ITERATIO")
        print("#" * 80)
        if LG_Training:
            env = Environment(weights=None)
            policy, Q = LG_VI(env, theta=0.01, discount_factor=0.7, priority=LG_priority)

            np.save(lg_policy_name, policy)
            np.save(lg_Q_name, Q, allow_pickle=True)

            print(f"\nSaved policy to {lg_policy_name}")
            print(f"Saved Q-values to {lg_Q_name}\n")

            print("-------------------")
            print("Training Finished!!!")

        else:
            policy = np.load(lg_policy_name)
        
        print("\nTesting policy...\n")

        env = Environment(weights=None)

        if Test_with_vectorial:
            results = test_policy_vectorial(env, policy, num_episodes=Test_Episodes, verbose=True)
        
        if Show_visualisation:
            tester_vect(env, policy, drawing=True)


    # #########################################################################
    # LEXICOGRAPHIC CONVEX HULL VALUE ITERATION
    # #########################################################################    
    
    if algorithm_used == CONVEX_HULL_VALUE_ITERATION:
        print("\n" + "#" * 80)
        print("ALGORITHM: CONVEX HULL VALUE ITERATION")
        print("#" * 80)

        if CHVI_Calculate_hulls:
            print("\nComputing Q-Hulls...")
            print("-" * 80 + "\n")

            env = Environment(weights=None) 
            q_hulls = LG_CH_VI(env, theta=0.01, discount_factor=0.7)
            
            np.save(chvi_hulls_name, q_hulls, allow_pickle=True)
            print(f"Saved Q-hulls to {chvi_hulls_name}\n")
        
        else:
            # EXTRACT POLICY FROM Q-HULLS 
            print(f"Loading Q-hulls from {chvi_hulls_name}...\n")
            q_hulls = np.load(chvi_hulls_name, allow_pickle=True).item()
            
        env = Environment(weights=None)

        if CHVI_extraction_method == 'weighted':
            print("EXTRACTION METHOD: WEIGHTED")
            print(f"Weights: {weights}")

            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
                
            np.save(chvi_policy_name, policy)
            print(f"Policy saved to {chvi_policy_name}\n")
                
        elif CHVI_extraction_method == 'lexicographic':
            print("EXTRACTION METHOD: LEXICOGRAPHIC")
            print(f"Priority: {CHVI_priority.upper()}")

            policy = extract_lexicographic_policy(q_hulls, CHVI_priority, env, env.n_actions)
            np.save(chvi_policy_name, policy)
            print(f"Policy saved to {chvi_policy_name}\n")

        print("#" * 80)
        print("TESTING")
        print("#" * 80 + "\n")
        
        if Test_with_vectorial:
            results = test_policy_vectorial(env, policy, num_episodes=Test_Episodes, verbose=True)
            
        if Show_visualisation:
            tester_vect(env, policy, drawing=True)


