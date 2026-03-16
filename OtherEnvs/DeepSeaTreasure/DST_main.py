import numpy as np
import os
import sys
import mo_gymnasium as mo_gym


VALUE_ITERATION = 0
CONVEX_HULL_VI = 1


if __name__ == "__main__":
    
    # ##############################################################
    # CONFIGURATION
    # ##############################################################

    algorithm_used = VALUE_ITERATION # VALUE_ITERATION | CONVEX_HULL_VI

    weights = [100, 1] # [w_treasure, w_time]
    Training = False # True: train and save policy, False: load policy
    Calculate_hulls = True # For COVNEX_HULL_VI True: compute, False: extract from saved hulls
    Test = True # True: visualise poolicy after training/loading
    Evaluate_policy = True #True: run evaluation

    discount_factor = 0.7
    n_eval_episodes = 100
    max_steps = 50

    # ##############################################################
    # EXECUTION
    # ##############################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(BASE_DIR, 'environment'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'algorithms'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))


    from DST_Environment import DSTEnvironment
    from DST_VI import value_iteration
    from auxiliary_functions_dst import tester, example_execution
    from CHVI_DST import convexhull_VI, extract_policy_for_weights, get_initial_state_hull

    PDIR = os.path.join(BASE_DIR, 'dst_policies')
    os.makedirs(PDIR, exist_ok=True)
 
    MNS_VI   = os.path.join(PDIR, 'DST_VI_MNS.pkl')
    MNS_CHVI = os.path.join(PDIR, 'CHVI_DST_MNS.pkl')


    if algorithm_used == VALUE_ITERATION:
        train_policy_name = os.path.join(PDIR, f"DST_VI_{weights[0]}-{weights[1]}-policy.npy")
        test_policy_name = os.path.join(PDIR, f"DST_VI_{weights[0]}-{weights[1]}-policy.npy")
        v_table_name = os.path.join(PDIR, f"DST_V_table_{weights[0]}-{weights[1]}.pkl")
    
    elif algorithm_used == CONVEX_HULL_VI:
        hulls_name = os.path.join(PDIR, f"DST_CHVI_qhulls.pkl")
        chvi_policy_name = os.path.join(PDIR, f"DST_CHVI_{weights[0]}-{weights[1]}-policy.npy")



    # -- Run ---------------------------------------------------
    if algorithm_used == VALUE_ITERATION:
        if Training:
            print(f"Training Value Iteration\n")
            print(f"Weights: {weights}\n")

            env = DSTEnvironment(weights=weights)

            start_row, start_col = env.start_state
            print(f"Start state: row={start_row}, col={start_col}\n")

            policy, q = value_iteration(env, theta=1.0, discount_factor=discount_factor, MNS_filename=MNS_VI, v_table_file=v_table_name)

            np.save(train_policy_name, policy)
            print(f"Saved policy to {train_policy_name}\n")

            print("-------------------")
            print("Finnished!!!")

            

        else:
            # Testing only

            if train_policy_name != test_policy_name:
                print("WARNING: Notice that the policy that will appear now is not the policy previously trained!!")

            print(f"Loading policy from {test_policy_name}...\n")
            policy = np.load(test_policy_name)
            print("Policy loaded successfully\n")

            env = DSTEnvironment(weights=weights)
    
    elif algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = DSTEnvironment(weights=weights)
            q_hulls = convexhull_VI(env, theta=0.01, discount_factor=discount_factor)
            policy = extract_policy_for_weights(q_hulls, env, env.n_actions)
            np.save(hulls_name, q_hulls, allow_pickle=True)
            np.save(chvi_policy_name, policy)
            print(f"Saved Q-hulls to {hulls_name}")
            print(f"Saved policy to {chvi_policy_name}")
        else:
            print(f"Loading Q-hulls from {hulls_name}...\n")
            q_hulls = np.load(hulls_name, allow_pickle=True).item()
            env = DSTEnvironment(weights=weights)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_name, policy)
            print(f"Saved policy to {chvi_policy_name}\n")        
    
     # -- Test --------------------------------------------------
    if Test:
        print("\nTesting policy...\n")
        render_env = mo_gym.make("deep-sea-treasure-v0", render_mode="human")
        example_execution(render_env, policy, render=True)

    

    # -- Evaluate ----------------------------------------------
    if Evaluate_policy:
            print(f"Episodes: {n_eval_episodes}, Max steps: {max_steps}, Discount: {discount_factor}")
            policy = None
            Q_hulls = None

            if algorithm_used == STOC_VALUE_ITERATION:
                policy_file = os.path.join(PDIR, f"VI_stochastic_{w[0]}-{w[1]}-{w[2]}-policy.npy")
                try:
                    policy = np.load(policy_file)
                    print(f"\nLoaded VI policy from {policy_file}")
                except FileNotFoundError:
                    print(f"\nERROR: Policy file not foungitd: {policy_file}")
                    print("Run VI training first.")
                    exit(1)

            elif algorithm_used == STOC_CONVEX_HULL_VI:
                hulls_file       = os.path.join(PDIR, "CHVI_stochastic_qhulls.pkl")
                chvi_policy_file = os.path.join(PDIR, f"CHVI_stochastic_{w[0]}-{w[1]}-{w[2]}-policy.npy")
                try:
                    policy = np.load(chvi_policy_file)
                    print(f"\nLoaded existing CHVI policy from {chvi_policy_file}")
                except FileNotFoundError:
                    print(f"\nCHVI policy not found. Extracting from Q-hulls...")
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
                        Q_hulls = None

                if Q_hulls is not None:
                    env = Environment(weights=weights)
                    env.reset()
                    initial_state = env.get_state()
                    value_hull = get_initial_state_hull(Q_hulls, initial_state, env.n_actions)
                    print(f"\n{'='*60}")
                    print("VALUE HULL AT INITIAL STATE")
                    print('='*60)
                    print(f"Initial state: {initial_state}")
                    print(f"Number of vertices: {len(value_hull)}")
                    print(f"\nVertices (Pareto-optimal value vectors):")
                    print(f"         [  r_car  ,  r_ped1 ,  r_ped2 ]")
                    for v in value_hull:
                        print(f"[{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}]")
                    print('='*60)
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
            std_vec  = results['std_return']
            print(f"\nMean discounted vector return:")
            print(f"  [r_car, r_ped1, r_ped2] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")

            if weights is not None:
                weights_arr        = np.array(weights)
                scalarised_mean    = np.dot(mean_vec, weights_arr)
                scalarised_returns = np.dot(results['episode_returns'], weights_arr)
                print(f"  Mean scalarised return: {scalarised_mean:.4f}")
                print(f"  Std scalarised return:  {np.std(scalarised_returns):.4f}")