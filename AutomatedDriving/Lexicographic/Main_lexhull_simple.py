import numpy as np
from EnvironmentFunctions.ADS_Environment import Environment
from Lexicographic.LG_VI_stoc_lexhull_v3 import LG_VI_lexhull
from EnvironmentFunctions.auxiliary_functions import tester

if __name__ == "__main__":
    
    Training = False  # True: train and save, False: load and test
    
    # For testing, choose which priority order to test
    test_priority = [2, 0, 1]
    
    if Training:
        print("Training Lexicographic Value Iteration (lexhull)")
        print("This will train ONCE and extract policies for ALL 6 priority orders\n")
        
        env = Environment(weights=None)
        
        policies, Q_hulls = LG_VI_lexhull(
            env, 
            theta=1.0, 
            discount_factor=0.7,
            v_hulls_file="policies/LG_VI_lexhull_V_hulls.pkl",
            q_hulls_file="policies/LG_VI_lexhull_Q_hulls.pkl"
        )
        
        # Save all 6 policies
        print("\nSaving all policies...")
        for priority_tuple, policy in policies.items():
            priority_str = f"{priority_tuple[0]}-{priority_tuple[1]}-{priority_tuple[2]}"
            policy_name = f"policies/LG_VI_lexhull_{priority_str}_policy.npy"
            np.save(policy_name, policy)
            print(f"  Saved policy for priority {list(priority_tuple)}")
        
        print("\n-------------------")
        print("Finished!!!")
        
        # Test one of the policies
        print(f"\nTesting policy for priority {test_priority}...\n")
        test_priority_tuple = tuple(test_priority)
        tester(env, policies[test_priority_tuple], drawing=True)
    
    else:
        # TESTING ONLY - load a specific policy
        priority_str = f"{test_priority[0]}-{test_priority[1]}-{test_priority[2]}"
        policy_name = f"policies/LG_VI_lexhull_{priority_str}_policy.npy"
        
        print(f"Loading policy from {policy_name}...\n")
        policy = np.load(policy_name)
        print("Policy loaded successfully\n")
        
        env = Environment(weights=None)
        
        print(f"Testing policy for priority {test_priority}...\n")
        tester(env, policy, drawing=True)