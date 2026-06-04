import numpy as np
from ADS_Environment import Environment
from VI_deterministic import value_iteration
from QL_deterministic import q_learning
from CHVI_deterministic import convexhull_VI, extract_policy_for_weights
from auxiliary_functions import tester

Q_LEARNING = 0
VALUE_ITERATION = 1
CONVEX_HULL_VI = 2 

if __name__ == "__main__":
    algorithm_used = VALUE_ITERATION
    weights = [1, 10, 100]
    Test = True             # True: test policy after training/loading 
    # For CONVEX_HULL_VI
    Calculate_hulls = True  # True: compute Q-hulls, False: extract policy from existing hulls

    # For Q_LEARNING and VALUE_ITERATION
    Training = True         # True: train and save policy. False: load existing policy.


    if algorithm_used == Q_LEARNING:
        train_policy_name = f"deterministic_policies/QL_deterministic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        test_policy_name = f"deterministic_policies/QL_deterministic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    if algorithm_used == VALUE_ITERATION:
        train_policy_name = f"deterministic_policies/VI_deterministic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        test_policy_name = f"deterministic_policies/VI_deterministic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        v_table_name = f"deterministic_policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
    if algorithm_used == CONVEX_HULL_VI:
        hulls_name = "deterministic_policies/CHVI_deterministic_qhulls.npy"
        chvi_policy_name = f"deterministic_policies/CHVI_deterministic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"

    if algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = Environment(weights=weights)

            policy, q_hulls = convexhull_VI(env, theta=0.01, discount_factor=0.7)
            np.save(hulls_name, q_hulls, allow_pickle=True)
            np.save(chvi_policy_name, policy)
            print(f"Saved Q-hulls to {hulls_name}")
            print(f"Saved policy to {chvi_policy_name}")
        else:
            print(f"Loading Q hulls from {hulls_name}...\n")
            q_hulls = np.load(hulls_name, allow_pickle=True).item()

            env = Environment(weights=weights)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_name, policy)
            print(f"Saved policy to {chvi_policy_name}\n")

    else:
        if Training:
            print(f"Training {['Q-Learning', 'Value Iteration'][algorithm_used]}\n")
            print(f"Weights: {weights}\n")

            env = Environment(weights=weights)
            env.reset()

            if algorithm_used == Q_LEARNING:
                policy, q = q_learning(env, alpha=0.8, gamma=0.7)
            elif algorithm_used == VALUE_ITERATION:
                policy, q = value_iteration(env, discount_factor=0.7)

            np.save(train_policy_name, policy)
            print(f"Saved policy to {train_policy_name}")
            print("-------------------")
            print("Finnished!!!")

        else:
            if train_policy_name != test_policy_name:
                print(
                    "WARNING: Notice that the policy that will appear now is not the policy previously trained!!"
                )
            print(f"Loading policy from {test_policy_name}...\n")
            policy = np.load(test_policy_name)
            print("Policy loaded successfully\n")

            env = Environment(weights=weights)

    if Test:
        print("\nTesting policy...\n")
        tester(env, policy, drawing=True)
