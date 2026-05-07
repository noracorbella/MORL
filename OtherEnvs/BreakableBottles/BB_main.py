import numpy as np
import os
import sys
import pickle
import mo_gymnasium as mo_gym


VALUE_ITERATION  = 0
CONVEX_HULL_VI   = 1
LGVI_LEXMAX      = 2
LGVI_LEXHULL     = 3


if __name__ == "__main__":

    # ##############################################################
    # CONFIGURATION 
    # ##############################################################

    algorithm_used = CONVEX_HULL_VI # VALUE_ITERATION | CONVEX_HULL_VI | LGVI_LEXMAX | LGVI_LEXHULL

    # -- Environment settings ----------------------------------
    size                = 5
    prob_drop           = 0.1
    time_penalty        = -1
    bottle_reward       = 25
    unbreakable_bottles = True

    # -- VI and CHVI settings ----------------------------------
    weights         = [1, 100, 10000]   # [w_time, w_delivery, w_potential]
    Training        = True  # True: train and save policy, False: load policy
    Calculate_hulls = True # For COVNEX_HULL_VI True: compute, False: extract from saved hulls
    
    # -- Lexicographic settings --------------------------------
    priority     = [0, 1, 2]     # priority order over [time, delivery, potential]
    lex_Training = True
    Calculate_lexhulls = False

    # -- Shared settings ---------------------------------------
    Test            = True  # True: visualise poolicy after training/loading
    Evaluate_policy = True  # True: run evaluation
    theta            = 0.1
    discount_factor  = 0.99
    n_eval_episodes  = 1000
    max_steps        = 200

    # ##############################################################
    # EXECUTION
    # ##############################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(BASE_DIR, 'environment'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'algorithms'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))

    from BB_Environment import BBEnvironment
    from BB_VI import value_iteration
    from auxiliary_functions_bb import tester, example_execution
    from BB_CHVI import convexhull_VI, extract_policy_for_weights, get_initial_state_hull
    from BB_LGVI_lexmax import LG_VI_lexmax
    from BB_LGVI_lexhull import LG_VI_lexhull, get_initial_state_hull as get_initial_state_hull_lex
    from BB_evaluate_policies import evaluate_policy

    PDIR = os.path.join(BASE_DIR, 'bb_policies')
    os.makedirs(PDIR, exist_ok=True)
 
    MNS_VI      = os.path.join(PDIR, 'BB_VI_MNS.pkl')
    MNS_CHVI    = os.path.join(PDIR, 'BB_CHVI_MNS.pkl')
    MNS_LEXMAX  = os.path.join(PDIR, 'BB_LGVI_lexmax_MNS.pkl')
    MNS_LEXHULL = os.path.join(PDIR, 'BB_LGVI_lexhull_MNS.pkl')

    w            = weights
    priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"
    w_str        = f"{w[0]}-{w[1]}-{w[2]}"

    env_kwargs = dict(
        size=size,
        prob_drop=prob_drop,
        time_penalty=time_penalty,
        bottle_reward=bottle_reward,
        unbreakable_bottles=unbreakable_bottles,
    )

    if algorithm_used == VALUE_ITERATION:
        train_policy_file = os.path.join(PDIR, f"BB_VI_{w_str}-policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"BB_V_table_{w_str}.pkl")
 
    elif algorithm_used == CONVEX_HULL_VI:
        hulls_file       = os.path.join(PDIR, "BB_CHVI_qhulls.pkl")
        chvi_policy_file = os.path.join(PDIR, f"BB_CHVI_{w_str}-policy.npy")
 
    elif algorithm_used == LGVI_LEXMAX:
        train_policy_file = os.path.join(PDIR, f"BB_LGVI_lexmax_{priority_str}_policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"BB_LGVI_lexmax_{priority_str}_V_table.pkl")

    elif algorithm_used == LGVI_LEXHULL:
        v_hulls_file     = os.path.join(PDIR, "BB_LGVI_lexhull_V_hulls.pkl")
        q_hulls_file     = os.path.join(PDIR, "BB_LGVI_lexhull_Q_hulls.pkl")
        test_policy_file = os.path.join(PDIR, f"BB_LGVI_lexhull_{priority_str}_policy.npy")

    # -- Run ---------------------------------------------------

    if algorithm_used == VALUE_ITERATION:
        if Training:
            print(f"Training Value Iteration")
            print(f"Weights: {weights}\n")
            env = BBEnvironment(weights=weights, **env_kwargs)
            policy, Q = value_iteration(
                env,
                theta=theta,
                discount_factor=discount_factor,
                MNS_filename=MNS_VI,
                v_table_file=v_table_file,
            )
            np.save(train_policy_file, policy)
            print(f"Saved policy to {train_policy_file}\n")
            print("-------------------\nFinished!!!")
        else:
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = BBEnvironment(weights=weights, **env_kwargs)

    elif algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = BBEnvironment(weights=weights, **env_kwargs)
            q_hulls = convexhull_VI(
                env,
                theta=theta,
                discount_factor=discount_factor,
                MNS_filename=MNS_CHVI,
                q_hulls_file=hulls_file,
            )
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}")
        else:
            print(f"Loading Q-hulls from {hulls_file}...")
            with open(hulls_file, 'rb') as f:
                q_hulls = pickle.load(f)
            env = BBEnvironment(weights=weights, **env_kwargs)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}\n")

    elif algorithm_used == LGVI_LEXMAX:
        if lex_Training:
            print("Training Lexicographic Value Iteration (lexmax)")
            print(f"Priority order: {priority}\n")
            env = BBEnvironment(weights=None, **env_kwargs)
            policy, Q = LG_VI_lexmax(
                env,
                theta=theta,
                discount_factor=discount_factor,
                priority=priority,
                MNS_filename=MNS_LEXMAX,
                v_table_file=v_table_file,
            )
            np.save(train_policy_file, policy)
            print(f"Saved policy to {train_policy_file}\n")
            print("-------------------\nFinished!!!")
        else:
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = BBEnvironment(weights=None, **env_kwargs)

    elif algorithm_used == LGVI_LEXHULL:
        if Calculate_lexhulls:
            print("Training Lexicographic Hull Value Iteration (lexhull)")
            print("Trains once and extracts policies for all 6 priority orders\n")
            env = BBEnvironment(weights=None, **env_kwargs)
            policies, Q_hulls = LG_VI_lexhull(
                env,
                theta=theta,
                discount_factor=discount_factor,
                MNS_filename=MNS_LEXHULL,
                v_hulls_file=v_hulls_file,
                q_hulls_file=q_hulls_file,
            )
            print("\nSaving all policies...")
            for priority_tuple, pol in policies.items():
                p_str       = f"{priority_tuple[0]}-{priority_tuple[1]}-{priority_tuple[2]}"
                policy_name = os.path.join(PDIR, f"BB_LGVI_lexhull_{p_str}_policy.npy")
                np.save(policy_name, pol)
                print(f"  Saved policy for priority {list(priority_tuple)}")
            print("\n-------------------\nFinished!!!")
            policy = policies[tuple(priority)]
        else:
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = BBEnvironment(weights=None, **env_kwargs)

    # -- Test --------------------------------------------------
    if Test:
        print("\nTesting policy...\n")
        from mo_gymnasium.envs.breakable_bottles.breakable_bottles import BreakableBottles
        render_env = BreakableBottles(
            render_mode="human",
            prob_drop=prob_drop,
            time_penalty=time_penalty,
            bottle_reward=bottle_reward,
            unbreakable_bottles=unbreakable_bottles,
        )
        example_execution(render_env, policy, render=True, n_test_episodes=2)
    
    
    # -- Evaluate ----------------------------------------------
    if Evaluate_policy:
        print(f"\n{'='*60}")
        print("POLICY EVALUATION")
        print(f"{'='*60}")
        print(f"Episodes: {n_eval_episodes}, Max steps: {max_steps}, Discount: {discount_factor}")
 
        if algorithm_used == VALUE_ITERATION:
            try:
                policy = np.load(train_policy_file)
                print(f"\nLoaded VI policy from {train_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {train_policy_file}")
                exit(1)
 
        elif algorithm_used == CONVEX_HULL_VI:
            try:
                policy = np.load(chvi_policy_file)
                print(f"\nLoaded CHVI policy from {chvi_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {chvi_policy_file}")
                exit(1)
 
            Q_hulls = None
            try:
                with open(hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                print(f"Loaded Q-hulls from {hulls_file}")
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found, skipping hull report.")
 
            if Q_hulls is not None:
                env_tmp    = BBEnvironment(weights=weights, **env_kwargs)
                value_hull = get_initial_state_hull(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}")
                print("VALUE HULL AT INITIAL STATE")
                print(f"{'='*60}")
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                print(f"\nVertices (Pareto-optimal value vectors):")
                print(f"  [   time   ,  delivery , potential ]")
                for v in value_hull:
                    print(f"  [{v[0]:10.4f}, {v[1]:10.4f}, {v[2]:10.4f}]")
                print(f"{'='*60}")
                del env_tmp
 
        elif algorithm_used == LGVI_LEXMAX:
            try:
                policy = np.load(train_policy_file)
                print(f"\nLoaded lexmax policy from {train_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {train_policy_file}")
                exit(1)
 
        elif algorithm_used == LGVI_LEXHULL:
            try:
                policy = np.load(test_policy_file)
                print(f"\nLoaded lexhull policy from {test_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {test_policy_file}")
                exit(1)
 
            Q_hulls = None
            try:
                with open(q_hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                print(f"Loaded Q-hulls from {q_hulls_file}")
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found, skipping hull report.")
 
            if Q_hulls is not None:
                env_tmp    = BBEnvironment(weights=None, **env_kwargs)
                value_hull = get_initial_state_hull_lex(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}")
                print("VALUE HULL AT INITIAL STATE")
                print(f"{'='*60}")
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                print(f"\nVertices (lex-optimal value vectors):")
                print(f"  [   time   ,  delivery , potential ]")
                for v in value_hull:
                    print(f"  [{v[0]:10.4f}, {v[1]:10.4f}, {v[2]:10.4f}]")
                print(f"{'='*60}")
                del env_tmp
 
        env = BBEnvironment(weights=weights, **env_kwargs)
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
            discount_factor=discount_factor,
        )
 
        print(f"\nNumber of episodes     : {len(results['episode_returns'])}")
        print(f"Mean episode length    : {results['mean_length']:.2f}")
 
        mean_vec = results['mean_return']
        std_vec  = results['std_return']
        print(f"\nMean discounted vector return:")
        print(f"  [r_time, r_delivery, r_potential] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}] +- [{std_vec[0]:.4f}, {std_vec[1]:.4f}, {std_vec[2]:.4f}]")
        print(f"\nStd discounted vector return:")
        print(f"  [r_time, r_delivery, r_potential] = [{std_vec[0]:.4f}, {std_vec[1]:.4f}, {std_vec[2]:.4f}]")
 
        if weights is not None and algorithm_used != LGVI_LEXHULL:
            weights_arr        = np.array(weights)
            scalarised_mean    = np.dot(mean_vec, weights_arr)
            scalarised_returns = np.dot(results['episode_returns'], weights_arr)
            print(f"\nWeights                : {weights}")
            print(f"Mean scalarised return : {scalarised_mean:.4f}")
            print(f"Std  scalarised return : {np.std(scalarised_returns):.4f}")