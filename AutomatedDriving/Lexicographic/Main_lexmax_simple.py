import numpy as np
from EnvironmentFunctions.ADS_Environment import Environment
from Lexicographic.LG_VI_stoc_lexmax import LG_VI_lexmax
from EnvironmentFunctions.auxiliary_functions import tester

if __name__ == "__main__":
    
    Training = False  # True: train and save policy, False: load and test existing policy
    
    priority = [2, 0, 1]
    
    priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"
    train_policy_name = f"policies/LG_VI_lexmax_{priority_str}_policy.npy"
    test_policy_name = f"policies/LG_VI_lexmax_{priority_str}_policy.npy"
    v_table_name = f"policies/LG_VI_lexmax_{priority_str}_V_table.pkl"
    
    if Training:
        print(f"Training Lexicographic Value Iteration with lexmax")
        print(f"Priority order: {priority}\n")
        
        env = Environment(weights=None)
        
        policy, Q = LG_VI_lexmax(
            env, 
            theta=1.0, 
            discount_factor=0.7, 
            priority=priority,
            v_table_file=v_table_name
        )
        
        np.save(train_policy_name, policy)
        print(f"Saved policy to {train_policy_name}\n")
        
        print("-------------------")
        print("Finished!!!")
        
        print("\nTesting trained policy...\n")
        tester(env, policy, drawing=True)
    
    else:
        # TESTING ONLY
        if train_policy_name != test_policy_name:
            print("WARNING: Notice that the policy that will appear now is not the policy previously trained!!")
        
        print(f"Loading policy from {test_policy_name}...\n")
        policy = np.load(test_policy_name)
        print("Policy loaded successfully\n")
        
        env = Environment(weights=None)
        
        print("Testing policy...\n")
        tester(env, policy, drawing=True)