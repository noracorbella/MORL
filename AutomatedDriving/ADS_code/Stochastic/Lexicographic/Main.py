import numpy as np
from ADS_Environment import Environment
from LG_VI_stochastic import LG_VI
from q_learning_algorithm import q_learning
from CH_VI_stochastic import convexhull_VI, extract_policy_for_weights
from auxiliary_functions import tester

Q_LEARNING = 0
LG_VI = 1
CH_VI = 2 

if __name__ == "__main__":
    algorithm_used = LG_VI
  

    # For CONVEX_HULL_VI
    Calculate_hulls = False # True: compute Q-hulls, False: extract policy from existing hulls

    # For Q_LEARNING
    Training = False        # True: train and save policy, False: load and test existing policy

    # For LG_VI
    LG_Training = True # True: train lexicographic policy, False: load existing
    LG_priority = 'car'  # 'car' or 'pedestrian'

    weights = [10.0, 0.0, 10.0]

    hulls_name = "policies/CH_qhulls.npy"
    chvi_policy_name = f"policies/CHVI_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    LG_policy = f"policies/LG_VI_{LG_priority}_priority-policy.npy"
    LG_Q = f"policies/LG_VI_{LG_policy}_priority-Q.npy"


    if algorithm_used == LG_VI:
        if LG_Training:
            print("Training Lexicographic Value Iteration")
            print(f"{LG_priority.upper()} priority")

            env = Environment(weights=None)

            # Run
            policy, Q = LG_VI(env, theta=0.01, discount_factor=0.7, priority=LG_priority)

            # save
            np.save(LG_policy, policy)
            np.save(LG_policy, Q, allow_pickle=True)

            print(f"\nSaved policy to {LG_policy}")
            print(f"Saved Q-values to {LG_Q}\n")

            print("Testing trained policy...")
            test_env = Environment(weights=weights)
            tester(test_env, policy, drawing=True)

        else:
            print("Testing Lexicographic Value Iteration")
            print(f"{LG_priority.upper()} priority")
            print(f"Testing policy {LG_policy}\n")
            policy = np.load(LG_policy)

            env = Environment(weights=weights)
            tester(env, policy, drawing=True)

    
    
    
    if algorithm_used == CONVEX_HULL_VI:
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

    else:
        if Training:
            print(f"Training {['Q-Learning', 'Value Iteration'][algorithm_used]}\n")
            print(f"Weights: {weights}\n")

            env = Environment(weights=weights)
            save = True

            if algorithm_used == Q_LEARNING:
                policy, q = q_learning(env, alpha=0.8, gamma=0.7)
            elif algorithm_used == VALUE_ITERATION:
                policy, q = value_iteration(env, discount_factor=0.7)

                
            if save:
                np.save(train_policy_name, policy)
                print(f"Saved policy to {train_policy_name}\n")

            print("-------------------")
            print("Finnished!!!")

            
            print("\nTesting trained policy...\n")
            tester(env, policy, drawing=True)
        
        else:
            # TESTING ONLY
            
            if train_policy_name != test_policy_name:
                print("WARNING: Notice that the policy that will appear now is not the policy previously trained!!")
            
            print(f"Loading policy from {test_policy_name}...\n")
            policy = np.load(test_policy_name)
            print("Policy loaded successfully\n")

            env = Environment(weights=weights)
            
            print("Testing policy...\n")
            tester(env, policy, drawing=True)