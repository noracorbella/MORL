import numpy as np
import pickle
from ADS_Environment import Environment
from VI_stochastic import value_iteration
from q_learning_algorithm import q_learning
from CHVI_stochastic import convexhull_VI, extract_policy_for_weights, get_initial_state_hull
from Stocastic_auxiliary_functions import tester
from Stocastic_evaluate_policies import evaluate_policy

Q_LEARNING = 0
VALUE_ITERATION = 1
CONVEX_HULL_VI = 2 

if __name__ == "__main__":
    algorithm_used = Q_LEARNING
    weights = [1, 100, 10000]
    Test = True             # True: test policy after training/loading (applies to all algorithms)

    # For CONVEX_HULL_VI
    Calculate_hulls = True  # True: compute Q-hulls, False: extract policy from existing hulls

    # For Q_LEARNING and VALUE_ITERATION
    Training = False        # True: train and save policy, False: load and test existing policy

    if algorithm_used == Q_LEARNING:
        train_policy_name = f"stochastic_policies/QL_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        test_policy_name = f"stochastic_policies/QL_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    if algorithm_used == VALUE_ITERATION:
        train_policy_name = f"stochastic_policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        test_policy_name = f"stochastic_policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        v_table_name = f"stochastic_policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
    if algorithm_used == CONVEX_HULL_VI:
        hulls_name = "stochastic_policies/CHVI_stochastic_qhulls.npy"
        chvi_policy_name = f"stochastic_policies/CHVI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"

    # Evaluate policy settings
    Evaluate_policy = True
    n_eval_episodes = 10000
    max_steps = 200
    discount_factor = 0.7

    if algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = Environment(weights=None)
            env.weights = weights

            q_hulls_filename = "stochastic_policies/CHVI_stochastic_qhulls.pkl"

            q_hulls = convexhull_VI(env, theta=0.01, discount_factor=0.7, q_hulls_file=q_hulls_filename)

            np.save(hulls_name, q_hulls, allow_pickle=True)
            print(f"Saved Q-hulls to {hulls_name}\n")

            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_name, policy)
            print(f"Saved policy to {chvi_policy_name}\n")

        else:
            print(f"Loading Q-hulls from {hulls_name}...\n")
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
                policy, q = value_iteration(env, discount_factor=0.7, v_table_file=v_table_name)

            np.save(train_policy_name, policy)
            print(f"Saved policy to {train_policy_name}\n")

            print("-------------------")
            print("Finnished!!!")

        else:
            if train_policy_name != test_policy_name:
                print("WARNING: Notice that the policy that will appear now is not the policy previously trained!!")

            print(f"Loading policy from {test_policy_name}...\n")
            policy = np.load(test_policy_name)
            print("Policy loaded successfully\n")

            env = Environment(weights=weights)

    if Test:
        print("\nTesting policy...\n")
        tester(env, policy, drawing=True)

    if Evaluate_policy:
        if algorithm_used == Q_LEARNING:
            print("Evaluate_policy option is not available for Q_LEARNING algorithm.")
            exit(1)
        print(f"Episodes: {n_eval_episodes}")
        print(f"Max steps: {max_steps}")
        print(f"Discount factor: {discount_factor}")

        policy = None
        Q_hulls = None

        if algorithm_used == VALUE_ITERATION:
            policy_file = f"stochastic_policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
            try: 
                policy = np.load(policy_file)
                print(f"\nLoaded VI policy from {policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {policy_file}")
                print("Run VI training first or provide correct path.")
                exit(1)

        elif algorithm_used == CONVEX_HULL_VI:
            hulls_file = "stochastic_policies/CHVI_stochastic_qhulls.pkl"
            chvi_policy_file = f"stochastic_policies/CHVI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
            try:
                policy = np.load(chvi_policy_file)
                print(f"\nLoaded existing CHVI policy from {chvi_policy_file}")
            except FileNotFoundError:
                # Extract policy from Q-hulls
                print(f"\nCHVI policy not found. Extracting from Q-hulls")
                try:
                    with open(hulls_file, 'rb') as f:
                        Q_hulls = pickle.load(f)
                    print(f"Loaded Q-hulls from {hulls_file}")
                    
                    env_temp = Environment(weights=weights)
                    policy = extract_policy_for_weights(Q_hulls, weights, env_temp, env_temp.n_actions)
                    
                    np.save(chvi_policy_file, policy)
                    print(f"Saved extracted policy to {chvi_policy_file}")
                    del env_temp

                except FileNotFoundError:
                    print(f"\nERROR: Q-hulls file not found: {hulls_file}")
                    print("Run CHVI training first.")
                    exit(1)

            if Q_hulls is None:
                try:
                    with open(hulls_file, 'rb') as f:
                        Q_hulls = pickle.load(f)
                except FileNotFoundError:
                    print(f"Warning: Could not load Q-hulls")
                    Q_hulls = None

        if algorithm_used == CONVEX_HULL_VI and Q_hulls is not None:
            env = Environment(weights=weights)
            env.reset()
            initial_state = env.get_state()
            
            value_hull = get_initial_state_hull(Q_hulls, initial_state, env.n_actions)
            
            print(f"\n" + "="*60)
            print(f"VALUE HULL AT INITIAL STATE")
            print("="*60)
            print(f"Initial state indices: {initial_state}")
            print(f"Number of vertices: {len(value_hull)}")
            print(f"\nVertices (Pareto-optimal value vectors):")
            print(f"         [  r_car  ,  r_ped1 ,  r_ped2 ]")
            for i, v in enumerate(value_hull):
                print(f"[{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}]")
            print("="*60)


            del env
        

        env = Environment(weights=weights)

        results = evaluate_policy(
            env=env,
            policy=policy,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
            discount_factor=discount_factor
        )

        print(f"\nNumber of episodes: {len(results['episode_returns'])}")
        print(f"Mean episode length: {results['mean_length']:.2f}")

        print(f"Weights: {weights}")

        mean_vec = results['mean_return']
        std_vec = results['std_return']
        
        print(f"\nMean discounted vector return :")
        print(f"  [r_car, r_ped1, r_ped2] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")


        if weights is not None:
            weights = np.array(weights)
            scalarised_mean = np.dot(mean_vec, weights)
            scalarised_returns = np.dot(results['episode_returns'], weights)
            print(f"  Mean scalarised return: {scalarised_mean:.4f}")
            print(f"  Std scalarised return:  {np.std(scalarised_returns):.4f}")
            



