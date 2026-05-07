import numpy as np
import os
import sys
import pickle
import mo_gymnasium as mo_gym


VALUE_ITERATION = 0
CONVEX_HULL_VI = 1
LGVI_LEXMAX = 2
LGVI_LEXHULL = 3


if __name__ == "__main__":
    
    # ##############################################################
    # CONFIGURATION
    # ##############################################################

    algorithm_used = LGVI_LEXHULL # VALUE_ITERATION | CONVEX_HULL_VI | LGVI_LEXMAX | LGVI_LEXHULL

    tree_depth = 7   # 5 / 6 / 7

    # -- VI and CHVI settings ----------------------------------
    weights         = [0.0001, 0.01, 1, 100, 10000, 1000000]  # [protein, carbs, fats, vitamins, minerals, water]
    Training            = True  # True: train and save policy, False: load policy
    Calculate_hulls     = True  # For COVNEX_HULL_VI True: compute, False: extract from saved hulls
    
    # -- Lexicographic settings --------------------------------
    priority     = [5, 4, 3, 2, 1, 0]   
    lex_Training     = True
    
    # -- Shared settings ---------------------------------------
    Test            = True # True: visualise poolicy after training/loading
    Evaluate_policy = True  # True: run evaluation
    discount_factor = 1
    theta = 1e-4
    n_eval_episodes = 1
    max_steps = 20   # tree depth < 7, episodes terminate in tree_depth steps

    # ##############################################################
    # EXECUTION
    # ##############################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(BASE_DIR, 'environment'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'algorithms'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))


    from FT_Environment import FTEnvironment
    from FT_VI import value_iteration
    from auxiliary_functions_ft import tester, example_execution
    from FT_CHVI import convexhull_VI, extract_policy_for_weights, get_initial_state_hull
    from FT_LGVI_lexmax import LG_VI_lexmax
    # from FT_LGVI_lexhull import LG_VI_lexhull, get_initial_state_hull as get_initial_state_hull_lex
    from FT_LGVI_lexhull_timing_v2 import LG_VI_lexhull, get_initial_state_hull as get_initial_state_hull_lex
    from FT_evaluate_policies import evaluate_policy

    PDIR = os.path.join(BASE_DIR, 'ft_policies')
    os.makedirs(PDIR, exist_ok=True)
 
    MNS_VI      = os.path.join(PDIR, f'FT_VI_d{tree_depth}_MNS.pkl')
    MNS_CHVI    = os.path.join(PDIR, f'FT_CHVI_d{tree_depth}_MNS.pkl')
    MNS_LEXMAX  = os.path.join(PDIR, f'FT_LGVI_lexmax_d{tree_depth}_MNS.pkl')
    MNS_LEXHULL = os.path.join(PDIR, f'FT_LGVI_lexhull_d{tree_depth}_MNS.pkl')

    w            = weights
    priority_str = "-".join(str(p) for p in priority)
    w_str        = "-".join(str(x) for x in w)

    if algorithm_used == VALUE_ITERATION:
        train_policy_file = os.path.join(PDIR, f"FT_VI_d{tree_depth}_{w_str}-policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"FT_V_table_d{tree_depth}_{w_str}.pkl")

    elif algorithm_used == CONVEX_HULL_VI:
        hulls_file       = os.path.join(PDIR, f"FT_CHVI_d{tree_depth}_qhulls.pkl")
        chvi_policy_file = os.path.join(PDIR, f"FT_CHVI_d{tree_depth}_{w_str}-policy.npy")

    elif algorithm_used == LGVI_LEXMAX:
        train_policy_file = os.path.join(PDIR, f"FT_LGVI_lexmax_d{tree_depth}_{priority_str}_policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"FT_LGVI_lexmax_d{tree_depth}_{priority_str}_V_table.pkl")

    elif algorithm_used == LGVI_LEXHULL:
        v_hulls_file     = os.path.join(PDIR, f"FT_LGVI_lexhull_d{tree_depth}_V_hulls.pkl")
        q_hulls_file     = os.path.join(PDIR, f"FT_LGVI_lexhull_d{tree_depth}_Q_hulls.pkl")
        test_policy_file = os.path.join(PDIR, f"FT_LGVI_lexhull_d{tree_depth}_{priority_str}_policy.npy")


    # -- Run ---------------------------------------------------
    if algorithm_used == VALUE_ITERATION:
        if Training:
            print(f"Training Value Iteration\n")
            print(f"Weights: {weights}\n")

            env = FTEnvironment(weights=weights, tree_depth=tree_depth)

            start_row, start_col = env.start_state
            print(f"Start state: row={start_row}, col={start_col}\n")

            policy, q = value_iteration(env, theta=theta, discount_factor=discount_factor, MNS_filename=MNS_VI, v_table_file=v_table_file)

            np.save(train_policy_file, policy)
            print(f"Saved policy to {train_policy_file}\n")

            print("-------------------")
            print("Finnished!!!")

            

        else:
            # Testing only
            if train_policy_file != test_policy_file:
                print("WARNING: policy loaded is not the one just trained!")
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = FTEnvironment(weights=weights, tree_depth=tree_depth)
    
    elif algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = FTEnvironment(weights=weights, tree_depth=tree_depth)
            q_hulls = convexhull_VI(env, theta=theta, discount_factor=discount_factor, MNS_filename=MNS_CHVI, q_hulls_file=hulls_file)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(hulls_file, q_hulls, allow_pickle=True)
            np.save(chvi_policy_file, policy)
            print(f"Saved Q-hulls to {hulls_file}")
            print(f"Saved policy to {chvi_policy_file}")
        else:
            print(f"Loading Q-hulls from {hulls_file}...\n")
            q_hulls = np.load(hulls_file, allow_pickle=True).item()
            env = FTEnvironment(weights=weights, tree_depth=tree_depth)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}\n")        
    
    elif algorithm_used == LGVI_LEXMAX:
        if lex_Training:
            print("Training Lexicographic Value Iteration (lexmax)")
            print(f"Priority order: {priority}\n")
            env = FTEnvironment(weights=None, tree_depth=tree_depth)
            policy, Q = LG_VI_lexmax(
                env,
                theta=theta,
                discount_factor=discount_factor,
                priority=priority,
                MNS_filename=MNS_LEXMAX,
                v_table_file=v_table_file
            )
            np.save(train_policy_file, policy)
            print(f"Saved policy to {train_policy_file}\n")
            print("-------------------\nFinished!!!")
        else:
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded succesfully\n")
            env = FTEnvironment(weights=None, tree_depth=tree_depth)

    elif algorithm_used == LGVI_LEXHULL:
        if lex_Training:
            print("Training Lexicographic Value Iteration (lexhull)")
            env = FTEnvironment(weights=None, tree_depth=tree_depth)
            policies, Q_hulls, _ = LG_VI_lexhull(
                env,
                theta=theta,
                discount_factor=discount_factor,
                MNS_filename=MNS_LEXHULL,
                v_hulls_file=v_hulls_file,
                q_hulls_file=q_hulls_file,
            )
            # Save only the policy for the selected priority (to avoid saving all 720)
            policy = policies[tuple(priority)]
            np.save(test_policy_file, policy)
            print(f"Saved policy for priority {priority} to {test_policy_file}")
            print("\n-------------------\nFinished!!!")
            policy = policies[tuple(priority)]
        else:
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = FTEnvironment(weights=None, tree_depth=tree_depth)

     # -- Test --------------------------------------------------
    if Test:
        print("\nTesting policy...\n")
        render_env = mo_gym.make("fruit-tree-v0", depth=tree_depth, render_mode="human")
        example_execution(render_env, policy, render=True, n_test_episodes=2)

    

    # -- Evaluate ----------------------------------------------
    if Evaluate_policy:
        print("\nEvaluating policy...\n")
        print(f"Episodes: {n_eval_episodes}, Max steps: {max_steps}, Discount: {discount_factor}")

        if algorithm_used == CONVEX_HULL_VI:
            try:
                policy = np.load(chvi_policy_file)
                print(f"\nLoaded existing CHVI policy from {chvi_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {chvi_policy_file}")
                print("Run CHVI training first.")
                exit(1)
 
            Q_hulls = None
            try:
                with open(hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                print(f"Loaded Q-hulls from {hulls_file}")

            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found ({hulls_file}), skipping hull report.")

            if Q_hulls is not None:
                env_tmp    = FTEnvironment(weights=weights, tree_depth=tree_depth)
                value_hull = get_initial_state_hull(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}")
                print("VALUE HULL AT INITIAL STATE")
                print('='*60)
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                print(f"\nVertices (Pareto-optimal value vectors):")
                print(f"  [ protein ,  carbs  ,  fats   , vitamins, minerals,  water  ]")
                for v in value_hull:
                    print(f"  [{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}, {v[3]:8.4f}, {v[4]:8.4f}, {v[5]:8.4f}]")
                print(f"{'='*60}")
                del env_tmp

        elif algorithm_used == VALUE_ITERATION:
            try:
                policy = np.load(train_policy_file)
                print(f"\nLoaded VI policy from {train_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {train_policy_file}")
                print("Run VI training first.")
                exit(1)
        
        elif algorithm_used == LGVI_LEXMAX:
            try:
                policy = np.load(train_policy_file)
                print(f"\nLoaded lexmax policy from {train_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {train_policy_file}")
                print("Run lexmax training first.")
                exit(1)
        
        elif algorithm_used == LGVI_LEXHULL:
            try:
                policy = np.load(test_policy_file)
                print(f"\nLoaded lexhull policy from {test_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {test_policy_file}")
                print("Run lexhull training first.")
                exit(1)
            
            Q_hulls = None
            try:
                with open(q_hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                print(f"Loaded Q-hulls from {q_hulls_file}")
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found ({q_hulls_file}), skipping hull report.")

            if Q_hulls is not None:
                env_tmp    = FTEnvironment(weights=None, tree_depth=tree_depth)
                value_hull = get_initial_state_hull_lex(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}")
                print("VALUE HULL AT INITIAL STATE")
                print(f"{'='*60}")
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                print(f"\nVertices (lex-optimal value vectors):")
                print(f"  [ protein ,  carbs  ,  fats   , vitamins, minerals,  water  ]")
                for v in value_hull:
                    print(f"  [{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}, {v[3]:8.4f}, {v[4]:8.4f}, {v[5]:8.4f}]")
                print(f"{'='*60}")
                del env_tmp




        env = FTEnvironment(weights=weights, tree_depth=tree_depth)
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
            discount_factor=discount_factor
        )

        print(f"\nNumber of episodes: {len(results['episode_returns'])}")
        print(f"Mean episode length: {results['mean_length']:.2f}")
        

        mean_vec = results['mean_return']
        std_vec  = results['std_return']
        labels   = ['protein', 'carbs', 'fats', 'vitamins', 'minerals', 'water']
        print(f"\nMean discounted vector return:")
        print(f"[{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}, {mean_vec[3]:.4f}, {mean_vec[4]:.4f}, {mean_vec[5]:.4f}]")
        for i, label in enumerate(labels):
            print(f"  {label:10s}: {mean_vec[i]:.4f} +- {std_vec[i]:.4f}")

        if weights is not None and algorithm_used != LGVI_LEXHULL:
            weights_arr        = np.array(weights)
            scalarised_mean    = np.dot(mean_vec, weights_arr)
            scalarised_returns = np.dot(results['episode_returns'], weights_arr)
            print(f"\nWeights                : {weights}")
            print(f"Mean scalarised return : {scalarised_mean:.4f}")
            print(f"Std  scalarised return : {np.std(scalarised_returns):.4f}")