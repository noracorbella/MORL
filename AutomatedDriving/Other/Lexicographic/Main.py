import numpy as np
import pickle
from ADS_Environment import Environment
from LGVI_stochastic_lexhull import LG_VI_lexhull
from LG_auxiliary_functions import tester
from LGVI_stochastic_lexmax import LG_VI_lexmax
from LG_evaluate_policies import evaluate_policy
from LG_utils import lex_max

LGVI_LEXMAX = 0
LGVI_LEXHULL = 1

if __name__ == "__main__":
    algorithm_used = LGVI_LEXMAX
    priority = [2, 1, 0]

    Training = False # True: train and save policy, False: load and test existing policy
    Test = True     # True: test policy after training/loading (applies to all algorithms)
    
    # Evaluate policy settings
    Evaluate_policy = True
    n_eval_episodes = 10000
    max_steps = 200
    discount_factor = 0.7


    priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"

    if algorithm_used == LGVI_LEXMAX:
        train_policy_name = f"lexicographic_policies/LGVI_lexmax_{priority_str}_policy.npy"
        test_policy_name = f"lexicographic_policies/LGVI_lexmax_{priority_str}_policy.npy"
        v_table_name = f"lexicographic_policies/LGVI_lexmax_{priority_str}_V_table.pkl"

    elif algorithm_used == LGVI_LEXHULL:
        v_hulls_file = "lexicographic_policies/LGVI_lexhull_V_hulls.pkl"
        q_hulls_file = "lexicographic_policies/LGVI_lexhull_Q_hulls.pkl"
        test_policy_name = f"lexicographic_policies/LGVI_lexhull_{priority_str}_policy.npy"

    if Training:
        if algorithm_used == LGVI_LEXMAX:
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

        elif algorithm_used == LGVI_LEXHULL:
            print("Training Lexicographic Value Iteration (lexhull)")
            print("Train ONCE and extract policies for ALL 6 priority orders\n")

            env = Environment(weights=None)

            policies, Q_hulls = LG_VI_lexhull(
                env,
                theta=1.0,
                discount_factor=0.7,
                v_hulls_file=v_hulls_file,
                q_hulls_file=q_hulls_file
            )

            # Save all 6 policies
            print("\nSaving all policies...")
            for priority_tuple, pol in policies.items():
                p_str = f"{priority_tuple[0]}-{priority_tuple[1]}-{priority_tuple[2]}"
                policy_name = f"lexicographic_policies/LGVI_lexhull_{p_str}_policy.npy"
                np.save(policy_name, pol)
                print(f"  Saved policy for priority {list(priority_tuple)}")
            print("\n-------------------")
            print("Finished!!!")

            policy = policies[tuple(priority)]

    else:
        print(f"Loading policy from {test_policy_name}...\n")
        policy = np.load(test_policy_name)
        print("Policy loaded successfully\n")

        env = Environment(weights=None)

    if Test:
        print("\nTesting policy...\n")
        tester(env, policy, drawing=True)

    if Evaluate_policy:
        print(f"Episodes: {n_eval_episodes}")
        print(f"Max steps: {max_steps}")
        print(f"Discount factor: {discount_factor}")

        policy = None
        Q_hulls = None

        if algorithm_used == LGVI_LEXMAX:
            policy_file = f"lexicographic_policies/LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"
            v_table_file = f"lexicographic_policies/LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_V_table.pkl"

            try:
                policy = np.load(policy_file)
                print(f"\nLoaded VI policy from {policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {policy_file}")
                print("Run VI training first or provide correct path.")
                exit(1)
            
            env_tmp = Environment(weights=None)

            try:
                with open(v_table_file, 'rb') as f:
                    V = pickle.load(f)
                env_tmp.reset()
                initial_state = env_tmp.get_state()
                theoretical_value = V[initial_state[0], initial_state[1], initial_state[2]]
            except FileNotFoundError:
                theoretical_value = None
            
            del env_tmp

        elif algorithm_used == LGVI_LEXHULL:
            policy_file = f"lexicographic_policies/LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"
            v_hulls_file = f"lexicographic_policies/LGVI_lexhull_V_hulls.pkl"

            try:
                policy = np.load(policy_file)
                print(f"\nLoaded lexhull policy from {policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {policy_file}")
                print("Run lexhull training first.")
                exit(1)

            env_temp = Environment(weights=None)
            try:
                with open(v_hulls_file, 'rb') as f:
                    V_hulls = pickle.load(f)
                env_temp.reset()
                initial_state = env_temp.get_state()
                state_tuple = (initial_state[0], initial_state[1], initial_state[2])
                v_hull = V_hulls[state_tuple]
                best_idx = lex_max(v_hull, priority=list(priority))
                theoretical_value, value_hull = v_hull[best_idx], v_hull
            except FileNotFoundError:
                theoretical_value, value_hull = None, None
            
            del env_temp

        env = Environment(weights=None)

        results = evaluate_policy(
            env=env,
            policy=policy,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
            discount_factor=discount_factor
        )

        if theoretical_value is not None:
            print("\n THEORETICAL VALUE")
            print(f"V[initial_state] = {theoretical_value[0]:.4f}, {theoretical_value[1]:.4f}, {theoretical_value[2]:.4f}")
            if algorithm_used == LGVI_LEXHULL and value_hull is not None:
                print(f"Hull size at initial state: {len(value_hull)} vertices")

        print(f"\nEVALUATE POLICY RESULTS")
        print(f"Number of episodes: {len(results['episode_returns'])}")
        print(f"Mean episode length: {results['mean_length']:.2f}")

        mean_vec = results['mean_return']
        std_vec = results['std_return']

        print(f"\nMean discounted vector return :")
        print(f"  [r_car, r_ped1, r_ped2] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")
        print(f"\nStd discounted vector return:")
        print(f"  [r_car, r_ped1, r_ped2] = [{std_vec[0]:.4f}, {std_vec[1]:.4f}, {std_vec[2]:.4f}]")




