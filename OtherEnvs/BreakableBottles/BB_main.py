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

    algorithm_used = VALUE_ITERATION # VALUE_ITERATION | CONVEX_HULL_VI | LGVI_LEXMAX | LGVI_LEXHULL

    # -- Environment settings ----------------------------------
    size                = 5
    prob_drop           = 0.1
    time_penalty        = -1
    bottle_reward       = 25
    unbreakable_bottles = True

    # -- VI and CHVI settings ----------------------------------
    weights         =  [1, 10000, 100]    # [w_time, w_delivery, w_potential]
    chvi_priority   = [2, 1, 0]    # priority order for policy extraction (CONVEX_HULL_VI only)
    Training        = True  # True: train and save policy, False: load policy
    Calculate_hulls = True # For COVNEX_HULL_VI True: compute, False: extract from saved hulls

    # -- Lexicographic settings --------------------------------
    priority     = [0, 1, 2]     # priority order over [time, delivery, potential]
    lex_Training = False
    Calculate_lexhulls = False

    # -- Shared settings ---------------------------------------
    Test            = False  # True: visualise poolicy after training/loading
    Load_V_at_initial_state = True
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
    from BB_CHVI import convexhull_VI, get_initial_state_hull
    from BB_LG_utils import extract_lex_policy_from_qhulls
    from BB_evaluate_policies import evaluate_policy
    from BB_LGVI_lexmax import LG_VI_lexmax
    from BB_LGVI_lexhull import LG_VI_lexhull, get_initial_state_hull as get_initial_state_hull_lex

    PDIR = os.path.join(BASE_DIR, 'bb_policies')
    os.makedirs(PDIR, exist_ok=True)

    MNS_VI      = os.path.join(PDIR, 'uBB_VI_MNS.pkl')
    MNS_CHVI    = os.path.join(PDIR, 'uBB_CHVI_MNS_theta0.1.pkl')
    MNS_LEXMAX  = os.path.join(PDIR, 'uBB_LGVI_lexmax_MNS.pkl')
    MNS_LEXHULL = os.path.join(PDIR, 'uBB_LGVI_lexhull_MNS.pkl')

    w            = weights
    priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"
    chvi_priority_str = f"{chvi_priority[0]}-{chvi_priority[1]}-{chvi_priority[2]}"
    w_str        = f"{w[0]}-{w[1]}-{w[2]}"

    env_kwargs = dict(
        size=size,
        prob_drop=prob_drop,
        time_penalty=time_penalty,
        bottle_reward=bottle_reward,
        unbreakable_bottles=unbreakable_bottles,
    )

    if algorithm_used == VALUE_ITERATION:
        train_policy_file = os.path.join(PDIR, f"uBB_VI_{w_str}-policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"uBB_V_table_{w_str}.pkl")

    elif algorithm_used == CONVEX_HULL_VI:
        v_hulls_file = os.path.join(PDIR, "uBB_CHVI_vhulls_theta0.1.pkl")
        q_hulls_file       = os.path.join(PDIR, "uBB_CHVI_qhulls_theta0.1.pkl")
        chvi_policy_file = os.path.join(PDIR, f"uBB_CHVI_{chvi_priority_str}-policy_theta0.1.npy")

    elif algorithm_used == LGVI_LEXMAX:
        train_policy_file = os.path.join(PDIR, f"uBB_LGVI_lexmax_{priority_str}_policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"uBB_LGVI_lexmax_{priority_str}_V_table.pkl")

    elif algorithm_used == LGVI_LEXHULL:
        v_hulls_file     = os.path.join(PDIR, "uBB_LGVI_lexhull_V_hulls.pkl")
        q_hulls_file     = os.path.join(PDIR, "uBB_LGVI_lexhull_Q_hulls.pkl")
        test_policy_file = os.path.join(PDIR, f"uBB_LGVI_lexhull_{priority_str}_policy.npy")


    # -- Run ---------------------------------------------------
    if algorithm_used == VALUE_ITERATION:
        if Training:
            print(f"Training Value Iteration\n")
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
            if train_policy_file != test_policy_file:
                print("WARNING: policy loaded is not the one just trained!")
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = BBEnvironment(weights=weights, **env_kwargs)

    elif algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = BBEnvironment(weights=None, **env_kwargs)
            q_hulls = convexhull_VI(
                env,
                theta=theta,
                discount_factor=discount_factor,
                MNS_filename=MNS_CHVI,
                q_hulls_file=q_hulls_file,
            )
            policy = extract_lex_policy_from_qhulls(q_hulls, chvi_priority, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}")
        else:
            print(f"Loading Q-hulls from {q_hulls_file}...")
            with open(q_hulls_file, 'rb') as f:
                q_hulls = pickle.load(f)
            env = BBEnvironment(weights=None, **env_kwargs)
            policy = extract_lex_policy_from_qhulls(q_hulls, chvi_priority, env, env.n_actions)
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
                policy_name = os.path.join(PDIR, f"uBB_LGVI_lexhull_{p_str}_policy.npy")
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
                print("Run VI training first.")
                exit(1)

        elif algorithm_used == CONVEX_HULL_VI:
            try:
                policy = np.load(chvi_policy_file)
                print(f"\nLoaded CHVI policy from {chvi_policy_file}")
            except FileNotFoundError:
                print(f"\nERROR: Policy file not found: {chvi_policy_file}")
                print("Run CHVI training first.")
                exit(1)

            Q_hulls = None
            try:
                with open(q_hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                print(f"Loaded Q-hulls from {q_hulls_file}")
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found ({q_hulls_file}), skipping hull report.")

            if Q_hulls is not None:
                env_tmp    = BBEnvironment(weights=None, **env_kwargs)
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

        env = BBEnvironment(weights=None, **env_kwargs)
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
            discount_factor=discount_factor,
        )

        print(f"\nNumber of episodes: {len(results['episode_returns'])}")
        print(f"Mean episode length: {results['mean_length']:.2f}")

        mean_vec = results['mean_return']
        std_vec  = results['std_return']
        print(f"\nMean discounted vector return:")
        print(f"  [r_time, r_delivery, r_potential] = ")
        print(f"[{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}] +- [{std_vec[0]:.4f}, {std_vec[1]:.4f}, {std_vec[2]:.4f}]")

        if weights is not None and algorithm_used != LGVI_LEXHULL:
            weights_arr        = np.array(weights)
            scalarised_mean    = np.dot(mean_vec, weights_arr)
            scalarised_returns = np.dot(results['episode_returns'], weights_arr)
            print(f"\nWeights                : {weights}")
            print(f"Mean scalarised return : {scalarised_mean:.4f}")
            print(f"Std  scalarised return : {np.std(scalarised_returns):.4f}")




    # ##############################################################
    # LOAD V AT INITIAL STATE
    # ##############################################################
    # Loads a V-table / V-hulls file and reports the vector value
    # at the initial state. For hull-based files, selects the vertex
    # corresponding to the given weights (CHVI) or priority (LexHull).
    # ##############################################################

    if Load_V_at_initial_state:
        print(f"\n{'='*60}")
        print("THEORETICAL VALUE AT INITIAL STATE (loaded from file)")
        print(f"{'='*60}")

        load_weights  = weights  # used by VI (scalarisation)
        load_priority = chvi_priority if algorithm_used == CONVEX_HULL_VI else priority

        if algorithm_used == VALUE_ITERATION:
            load_file = os.path.join(PDIR, f"uBB_V_table_{weights[0]}-{weights[1]}-{weights[2]}_vec.pkl")
        elif algorithm_used == CONVEX_HULL_VI:
            load_file = os.path.join(PDIR, "uBB_CHVI_vhulls.pkl")
        elif algorithm_used == LGVI_LEXMAX:
            load_file = os.path.join(PDIR,f"uBB_LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_V_table.pkl")
        elif algorithm_used == LGVI_LEXHULL:
            load_file = os.path.join(PDIR, "uBB_LGVI_lexhull_V_hulls.pkl")

        env_load = BBEnvironment(weights=weights)
        s0       = tuple(env_load.start_state)

        try:
            with open(load_file, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: file not found: {load_file}")
            data = None

        if data is not None:
            fname = os.path.basename(load_file)
            V_s0  = None

            # ---------- VI: vector V table (numpy array) --------------
            if "uBB_V_table" in fname and "_vec" in fname:
                V_s0 = np.asarray(data[s0])
                print(f"File         : {fname}")
                print(f"Algorithm    : VI (vector)")
                print(f"Weights      : {load_weights}")
                print(f"Initial state: {s0}")

            # ---------- LexMax: vector V table ------------------------
            elif "uBB_LGVI_lexmax" in fname and "V_table" in fname:
                V_s0 = np.asarray(data[s0])
                print(f"File         : {fname}")
                print(f"Algorithm    : LexMax")
                print(f"Priority     : {load_priority}")
                print(f"Initial state: {s0}")

            # ---------- CHVI: V-hulls dict ----------------------------
            elif "uBB_CHVI_vhulls" in fname:
                value_hull = np.asarray(data[s0])
                ordered    = value_hull[:, load_priority]
                idx        = int(np.lexsort(ordered.T[::-1])[-1])
                V_s0       = value_hull[idx]
                print(f"File         : {fname}")
                print(f"Algorithm    : CHVI")
                print(f"Priority      : {load_priority}")
                print(f"Initial state: {s0}")
                print(f"Hull size    : {len(value_hull)} vertices")

            # ---------- LexHull: V-hulls dict -------------------------
            elif "uBB_LGVI_lexhull_V_hulls" in fname:
                value_hull = np.asarray(data[s0])
                ordered    = value_hull[:, load_priority]
                idx        = int(np.lexsort(ordered.T[::-1])[-1])
                V_s0       = value_hull[idx]
                print(f"File         : {fname}")
                print(f"Algorithm    : LexHull")
                print(f"Priority     : {load_priority}")
                print(f"Initial state: {s0}")
                print(f"Hull size    : {len(value_hull)} vertices")

            else:
                print(f"ERROR: could not infer algorithm from filename '{fname}'")

            if V_s0 is not None:
                print(f"V(s0)        : [r_time, r_delivery, r_potential] = "
                      f"[{V_s0[0]:.4f}, {V_s0[1]:.4f}, {V_s0[2]:.4f}]")

        print(f"{'='*60}")
        del env_load