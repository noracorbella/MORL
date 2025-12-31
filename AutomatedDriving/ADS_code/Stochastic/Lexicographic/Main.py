import numpy as np
from ADS_Environment import Environment
from LG_VI_stochastic import LG_VI
from q_learning_algorithm import q_learning
from CH_VI_stochastic import convexhull_VI, extract_policy_for_weights
from auxiliary_functions import tester, test_policy_vectorial, tester_vect

Q_LEARNING = 0
LEXICOGRAPHIC_VALUE_ITERATION = 1
CONVEX_HULL_VALUE_ITERATION = 2 

if __name__ == "__main__":
    algorithm_used = LEXICOGRAPHIC_VALUE_ITERATION
  

    # For CONVEX_HULL_VI
    Calculate_hulls = False # True: compute Q-hulls, False: extract policy from existing hulls

    # For Q_LEARNING
    Training = False        # True: train and save policy, False: load and test existing policy

    # For LEXICOGRAPHIC_VALUE_ITERATION
    LG_Training = True # True: train lexicographic policy, False: load existing
    LG_priority = 'pedestrian'  # 'car' or 'pedestrian'

    # Testing configuration
    Test_with_vectorial = True # Use vectorial reward to test
    Test_Episodes = 100 # Numbero of episodes to test
    Show_visualisation = True

    weights = [1.0, 100.0, 100.0]

    hulls_name = "policies/CH_qhulls.npy"
    chvi_policy_name = f"policies/CHVI_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    LG_policy_name = f"policies/LGVI_{LG_priority}_priority-policy.npy"
    LG_Q_name = f"policies/LGVI_{LG_priority}_priority-Q.npy"


    # #########################################################################
    # LEXICOGRAPHIC VALUE ITERATION
    # #########################################################################
    if algorithm_used == LEXICOGRAPHIC_VALUE_ITERATION:
        if LG_Training:
            print("Training Lexicographic Value Iteration")
            print(f"{LG_priority.upper()} priority")

            env = Environment(weights=None)

            # Run
            policy, Q = LG_VI(env, theta=0.01, discount_factor=0.7, priority=LG_priority)

            # save
            np.save(LG_policy_name, policy)
            np.save(LG_Q_name, Q, allow_pickle=True)

            print(f"\nSaved policy to {LG_policy_name}")
            print(f"Saved Q-values to {LG_Q_name}\n")

            
        else:
            print("Testing Lexicographic Value Iteration")
            print(f"{LG_priority.upper()} priority")
            print(f"Testing policy {LG_policy_name}\n")

            policy = np.load(LG_policy_name)

 
        
        print("Testing trained policy...")

        env = Environment(weights=None)

        if Test_with_vectorial:
            print(f"Testing policy over {Test_Episodes} episodes...")
            results = test_policy_vectorial(env, policy, num_episodes=Test_Episodes, verbose=True)

        if Show_visualisation:
            tester_vect(env, policy, drawing=True)


    # #########################################################################
    # CONVEX HULL VALUE ITERATION
    # #########################################################################    
    
    if algorithm_used == CONVEX_HULL_VALUE_ITERATION:
        if Calculate_hulls:
            env = Environment(weights=None) 
            env.weights = weights

            q_hulls = convexhull_VI(env, theta=0.01, discount_factor=0.7)
            
            np.save(hulls_name, q_hulls, allow_pickle=True)
            print(f"Saved Q-hulls to {hulls_name}\n")
        
        else:
            print(f"Loading Q-hulls from {hulls_name}...\n")
            q_hulls = np.load(hulls_name, allow_pickle=True).item()

            env = Environment(weights=weights)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_name, policy)
            print(f"Saved policy to {chvi_policy_name}\n")

            print("Testing policy...\n")
            tester(env, policy, drawing=True)

    # #########################################################################
    # Q-LEARNING
    # #########################################################################    
    
    elif algorithm_used == Q_LEARNING:
        if Training:
            print(f"Training {['Q-Learning', 'Value Iteration'][algorithm_used]}\n")
            print(f"Weights: {weights}\n")

            env = Environment(weights=weights)

            policy, q = q_learning(env, alpha=0.8, gamma=0.7)

            q_policy_name = f"policies/QL_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
                
            np.save(q_policy_name, policy)
            print(f"Saved policy to {q_policy_name}\n")

            print("-------------------")
            print("Finnished!!!")

            
            print("\nTesting trained policy...\n")
            tester(env, policy, drawing=True)
        
        else:
            # TESTING ONLY
            q_policy_name = f"policies/QL_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
            policy = np.load(q_policy_name)
            env = Environment(weights=weights)
            tester(env, policy, drawing=True)
            
     