import numpy as np
from ADS_Environment import Environment
from value_iteration_algorithm_original import value_iteration
from q_learning_algorithm import q_learning
from convexhull_VI import convexhull_VI, extract_policy_for_weights, analyze_learned_policies
from auxiliary_functions import tester

Q_LEARNING = 0
VALUE_ITERATION = 1
CONVEX_HULL_VI = 2  # ← New algorithm option

if __name__ == "__main__":
    Training = True  # Set to True to apply algorithm. Set to false to directly observe learned policy.
    weights = [1.0, 100.0, 100.0]  # [car, ped1, ped2] - heavily prioritize pedestrians
    algorithm_used = CONVEX_HULL_VI

    train_policy_name = "policies/convexhull_policy.npy"
    test_policy_name = "policies/convexhull_policy.npy"
    hulls_name = "policies/convexhull_qhulls.npy"  

    if Training:
        env = Environment(weights=weights)
        save = True

        # Training
        if algorithm_used == Q_LEARNING:
            policy, q = q_learning(env, alpha=0.8, gamma=0.7)
            
        elif algorithm_used == VALUE_ITERATION:
            policy, q = value_iteration(env, discount_factor=0.7)
            
        elif algorithm_used == CONVEX_HULL_VI:
            policy, q_hulls = convexhull_VI(env, theta=0.01, discount_factor=0.7)

                        
        if save:
            np.save(train_policy_name, policy)  
            np.save(hulls_name, q_hulls)
            print(f"Saved policy to {train_policy_name}")
            print(f"Saved Q-hulls to {hulls_name}")
            
            # Analyze what policies we learned for different priorities!
            print("\n" + "="*70)
            print("ANALYZING LEARNED POLICIES FOR DIFFERENT ETHICAL FRAMEWORKS")
            print("="*70)
            analyze_learned_policies(q_hulls, env, env.n_actions)
            
            # Extract and test policies for different weight vectors
            print("\n" + "="*70)
            print("EXTRACTING POLICIES FOR DIFFERENT WEIGHT VECTORS")
            print("="*70)
            
            # Example 1: Car-selfish policy
            print("\n1. Car-selfish policy (car ignores pedestrians)")
            selfish_weights = [1.0, 0.0, 0.0]
            selfish_policy = extract_policy_for_weights(
                q_hulls, 
                np.array(selfish_weights) / np.sum(selfish_weights), 
                env, 
                env.n_actions
            )
            np.save("policy_car_selfish.npy", selfish_policy)
            print("   Saved as policy_car_selfish.npy")
            
            # Example 2: Democratic policy
            print("\n2. Democratic policy (equal consideration)")
            democratic_weights = [1.0, 1.0, 1.0]
            democratic_policy = extract_policy_for_weights(
                q_hulls, 
                np.array(democratic_weights) / np.sum(democratic_weights), 
                env, 
                env.n_actions
            )
            np.save("policy_democratic.npy", democratic_policy)
            print("   Saved as policy_democratic.npy")
            
            # # Example 3: Super safe policy
            # print("\n3. Super safe policy (extreme pedestrian priority)")
            # safe_weights = [1.0, 1000.0, 1000.0]
            # safe_policy = extract_policy_for_weights(
            #     q_hulls, 
            #     np.array(safe_weights) / np.sum(safe_weights), 
            #     env, 
            #     env.n_actions
            # )
            # np.save("policy_super_safe.npy", safe_policy)
            # print("   Saved as policy_super_safe.npy")

        if save and algorithm_used != CONVEX_HULL_VI:
            np.save(train_policy_name, policy)
            
        print("-------------------")
        print("Finished!!!")

    # Testing phase
    print("\n" + "="*70)
    print("TESTING PHASE")
    print("="*70)
    
    if train_policy_name != test_policy_name:
        print(
            "WARNING: Notice that the policy that will appear now is not the policy previously trained!!"
        )
    
    # Load and test the policy
    policy = np.load(test_policy_name)
    env = Environment(weights=weights)
    
    print(f"\nTesting policy: {test_policy_name}")
    print(f"Environment weights: {weights}")
    tester(env, policy, drawing=True)
    
    # If you want to test different policies, you can do:
    # print("\n--- Testing selfish policy ---")
    # selfish_policy = np.load("policy_selfish.npy")
    # tester(env, selfish_policy, drawing=True)