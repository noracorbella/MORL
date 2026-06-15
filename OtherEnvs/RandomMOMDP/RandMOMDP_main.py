import numpy as np
import os
import sys
import pickle


VALUE_ITERATION = 0
CONVEX_HULL_VI = 1
LGVI_LEXMAX = 2
LGVI_LEXHULL = 3


if __name__ == "__main__":

    # ##############################################################
    # CONFIGURATION
    # ##############################################################

    algorithm_used = CONVEX_HULL_VI  # VALUE_ITERATION | CONVEX_HULL_VI | LGVI_LEXMAX | LGVI_LEXHULL

    # -- VI and CHVI settings ----------------------------------
    weights         = [1, 100, 10000]   # linear scalarisation weights (len = n_objectives)
    chvi_priority   = [0, 1, 2]         # priority order for policy extraction (CONVEX_HULL_VI only)
    Training        = True   # True: train and save policy, False: load policy
    Calculate_hulls = True   # CHVI: True compute hulls, False load saved hulls

    # -- Lexicographic settings --------------------------------
    priority           = [0, 2, 1]     # priority order over objectives
    lex_Training       = True
    Calculate_lexhulls = True

    # -- Shared settings ---------------------------------------
    Test            = True  # True: simulate a few trajectories after training/loading
    Load_V_at_initial_state = True
    Evaluate_policy = True
    discount_factor = 0.7
    theta = 1e-5
    max_iterations = 10000
    n_eval_episodes = 10000
    max_steps       = 200

    # -- Environment settings ----------------------------------
    env_kwargs = dict(seed=45, states=100, actions=5, actions_per_state=5,
                      successors=1, objectives=3, terminal_states=1)

    # ##############################################################
    # EXECUTION
    # ##############################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(BASE_DIR, 'environment'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'algorithms'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))

    from RandMOMDP_Environment import RandMOMDPEnvironment
    from RandMOMDP_VI import value_iteration
    from auxiliary_functions import example_execution
    from RandMOMDP_CHVI import convexhull_VI, get_initial_state_hull
    from RandMOMDP_LG_utils import extract_lex_policy_from_qhulls
    from RandMOMDP_evaluate_policies import evaluate_policy
    from RandMOMDP_LGVI_lexmax import LG_VI_lexmax
    from RandMOMDP_LGVI_lexhull import LG_VI_lexhull, get_initial_state_hull as get_initial_state_hull_lex

    PDIR = os.path.join(BASE_DIR, 'randmomdp_policies')
    os.makedirs(PDIR, exist_ok=True)

    MNS_VI      = os.path.join(PDIR, 'RandMOMDP_VI_MNS.pkl')
    MNS_CHVI    = os.path.join(PDIR, 'RandMOMDP_CHVI_MNS.pkl')
    MNS_LEXMAX  = os.path.join(PDIR, 'RandMOMDP_LGVI_lexmax_MNS.pkl')
    MNS_LEXHULL = os.path.join(PDIR, 'RandMOMDP_LGVI_lexhull_MNS.pkl')

    w_str = "-".join(str(w) for w in weights)
    p_str = "-".join(str(p) for p in priority)
    cp_str = "-".join(str(p) for p in chvi_priority)

    if algorithm_used == VALUE_ITERATION:
        train_policy_file = os.path.join(PDIR, f"RandMOMDP_VI_{w_str}-policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"RandMOMDP_V_table_{w_str}.pkl")

    elif algorithm_used == CONVEX_HULL_VI:
        hulls_file       = os.path.join(PDIR, "RandMOMDP_CHVI_qhulls.pkl")
        chvi_policy_file = os.path.join(PDIR, f"RandMOMDP_CHVI_{cp_str}-policy.npy")

    elif algorithm_used == LGVI_LEXMAX:
        train_policy_file = os.path.join(PDIR, f"RandMOMDP_LGVI_lexmax_{p_str}_policy.npy")
        test_policy_file  = train_policy_file
        v_table_file      = os.path.join(PDIR, f"RandMOMDP_LGVI_lexmax_{p_str}_V_table.pkl")

    elif algorithm_used == LGVI_LEXHULL:
        v_hulls_file     = os.path.join(PDIR, "RandMOMDP_LGVI_lexhull_V_hulls.pkl")
        q_hulls_file     = os.path.join(PDIR, "RandMOMDP_LGVI_lexhull_Q_hulls.pkl")
        test_policy_file = os.path.join(PDIR, f"RandMOMDP_LGVI_lexhull_{p_str}_policy.npy")

    # -- Run ---------------------------------------------------
    if algorithm_used == VALUE_ITERATION:
        if Training:
            print(f"Training Value Iteration\nWeights: {weights}\n")
            env = RandMOMDPEnvironment(weights=weights, **env_kwargs)
            policy, Q = value_iteration(
                env, theta=theta, discount_factor=discount_factor, max_iterations=max_iterations,
                MNS_filename=MNS_VI, v_table_file=v_table_file,
            )
            np.save(train_policy_file, policy)
            print(f"Saved policy to {train_policy_file}\n-------------------\nFinished!!!")
        else:
            print(f"Loading policy from {test_policy_file}")
            policy = np.load(test_policy_file)
            env = RandMOMDPEnvironment(weights=weights, **env_kwargs)

        print(f"\nPolicy: {policy}")

    elif algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = RandMOMDPEnvironment(weights=None, **env_kwargs)
            q_hulls = convexhull_VI(
                env, theta=theta, discount_factor=discount_factor,
                MNS_filename=MNS_CHVI, q_hulls_file=hulls_file,
            )
            policy = extract_lex_policy_from_qhulls(q_hulls, chvi_priority, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}")
        else:
            print(f"Loading Q-hulls from {hulls_file}")
            with open(hulls_file, 'rb') as f:
                q_hulls = pickle.load(f)
            env = RandMOMDPEnvironment(weights=None, **env_kwargs)
            policy = extract_lex_policy_from_qhulls(q_hulls, chvi_priority, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}\n")

    elif algorithm_used == LGVI_LEXMAX:
        if lex_Training:
            print(f"Training Lexicographic Value Iteration (lexmax)\nPriority order: {priority}\n")
            env = RandMOMDPEnvironment(weights=None, **env_kwargs)
            policy, Q = LG_VI_lexmax(
                env, theta=theta, discount_factor=discount_factor, priority=priority,
                MNS_filename=MNS_LEXMAX, v_table_file=v_table_file,
            )
            np.save(train_policy_file, policy)
            print(f"Saved policy to {train_policy_file}\n-------------------\nFinished!!!")
        else:
            print(f"Loading policy from {test_policy_file}")
            policy = np.load(test_policy_file)
            env = RandMOMDPEnvironment(weights=None, **env_kwargs)

    elif algorithm_used == LGVI_LEXHULL:
        if Calculate_lexhulls:
            print("Training Lexicographic Hull Value Iteration (lexhull)")
            env = RandMOMDPEnvironment(weights=None, **env_kwargs)
            policies, Q_hulls = LG_VI_lexhull(
                env, theta=theta, discount_factor=discount_factor,
                MNS_filename=MNS_LEXHULL, v_hulls_file=v_hulls_file, q_hulls_file=q_hulls_file,
            )
            print("\nSaving all policies")
            for priority_tuple, pol in policies.items():
                pt_str      = "-".join(str(p) for p in priority_tuple)
                policy_name = os.path.join(PDIR, f"RandMOMDP_LGVI_lexhull_{pt_str}_policy.npy")
                np.save(policy_name, pol)
                print(f"  Saved policy for priority {list(priority_tuple)}")
            print("\n-------------------\nFinished!!!")
            policy = policies[tuple(priority)]
        else:
            print(f"Loading policy from {test_policy_file}")
            policy = np.load(test_policy_file)
            env = RandMOMDPEnvironment(weights=None, **env_kwargs)

    # -- Test --------------------------------------------------
    if Test:
        print("\nTesting policy\n")
        example_execution(env, policy, render=False, n_test_episodes=2)

    # -- Evaluate ----------------------------------------------
    if Evaluate_policy:
        print(f"\n{'='*60}\nPOLICY EVALUATION\n{'='*60}")
        print(f"Episodes: {n_eval_episodes}, Max steps: {max_steps}, Discount: {discount_factor}")

        if algorithm_used == VALUE_ITERATION:
            policy = np.load(train_policy_file)
        elif algorithm_used == CONVEX_HULL_VI:
            policy = np.load(chvi_policy_file)
            try:
                with open(hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                env_tmp    = RandMOMDPEnvironment(weights=None, **env_kwargs)
                value_hull = get_initial_state_hull(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}\nVALUE HULL AT INITIAL STATE\n{'='*60}")
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                for v in value_hull:
                    print("  " + np.array2string(np.asarray(v), precision=4))
                del env_tmp
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found ({hulls_file}).")
        elif algorithm_used == LGVI_LEXMAX:
            policy = np.load(train_policy_file)
        elif algorithm_used == LGVI_LEXHULL:
            policy = np.load(test_policy_file)
            try:
                with open(q_hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                env_tmp    = RandMOMDPEnvironment(weights=None, **env_kwargs)
                value_hull = get_initial_state_hull_lex(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}\nVALUE HULL AT INITIAL STATE\n{'='*60}")
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                for v in value_hull:
                    print("  " + np.array2string(np.asarray(v), precision=4))
                del env_tmp
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found ({q_hulls_file}).")

        env = RandMOMDPEnvironment(weights=None, **env_kwargs)
        results = evaluate_policy(
            env=env, policy=policy, n_eval_episodes=n_eval_episodes,
            max_steps=max_steps, discount_factor=discount_factor,
        )

        print(f"\nNumber of episodes: {len(results['episode_returns'])}")
        print(f"Mean episode length: {results['mean_length']:.2f}")

        mean_vec = results['mean_return']
        std_vec  = results['std_return']
        
        print(f"Mean discounted vector return: {np.array2string(mean_vec, precision=4)} +- {np.array2string(std_vec, precision=4)}")
        # print(f"Std discounted vector return: {np.array2string(std_vec, precision=4)}")

        # if weights is not None and algorithm_used != LGVI_LEXHULL:
        #     weights_arr        = np.array(weights)
        #     scalarised_mean    = np.dot(mean_vec, weights_arr)
        #     scalarised_returns = np.dot(results['episode_returns'], weights_arr)
        #     print(f"\nWeights                : {weights}")
        #     print(f"Mean scalarised return : {scalarised_mean:.4f}")
        #     print(f"Std  scalarised return : {np.std(scalarised_returns):.4f}")

    # ##############################################################
    # LOAD V AT INITIAL STATE
    # ##############################################################

    if Load_V_at_initial_state:
        print(f"\n{'='*60}\nTHEORETICAL VALUE AT INITIAL STATE (loaded from file)\n{'='*60}")

        load_priority = chvi_priority if algorithm_used == CONVEX_HULL_VI else priority

        if algorithm_used == VALUE_ITERATION:
            load_file = os.path.join(PDIR, f"RandMOMDP_V_table_{w_str}_vec.pkl")
        elif algorithm_used == CONVEX_HULL_VI:
            load_file = os.path.join(PDIR, "RandMOMDP_CHVI_vhulls.pkl")
        elif algorithm_used == LGVI_LEXMAX:
            load_file = os.path.join(PDIR, f"RandMOMDP_LGVI_lexmax_{p_str}_V_table.pkl")
        elif algorithm_used == LGVI_LEXHULL:
            load_file = os.path.join(PDIR, "RandMOMDP_LGVI_lexhull_V_hulls.pkl")

        env_load = RandMOMDPEnvironment(weights=weights, **env_kwargs)
        s0 = env_load.start_state

        try:
            with open(load_file, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: file not found: {load_file}")
            data = None

        if data is not None:
            fname = os.path.basename(load_file)
            V_s0  = None

            if "RandMOMDP_V_table" in fname and "_vec" in fname:
                V_s0 = np.asarray(data[s0])
                print(f"File: {fname}\nAlgorithm: VI (vector)\nWeights: {weights}\nInitial state: {s0}")

            elif "RandMOMDP_LGVI_lexmax" in fname and "V_table" in fname:
                V_s0 = np.asarray(data[s0])
                print(f"File: {fname}\nAlgorithm: LexMax\nPriority: {load_priority}\nInitial state: {s0}")

            elif "RandMOMDP_CHVI_vhulls" in fname or "RandMOMDP_LGVI_lexhull_V_hulls" in fname:
                value_hull = np.asarray(data[s0])
                ordered    = value_hull[:, load_priority]
                idx        = int(np.lexsort(ordered.T[::-1])[-1])
                V_s0       = value_hull[idx]
                algo = "CHVI" if "CHVI" in fname else "LexHull"
                print(f"File: {fname}\nAlgorithm: {algo}\nPriority: {load_priority}\n"
                      f"Initial state: {s0}\nHull size: {len(value_hull)} vertices")
            else:
                print(f"ERROR: could not infer algorithm from filename '{fname}'")

            if V_s0 is not None:
                print(f"V(s0): {np.array2string(np.asarray(V_s0), precision=4)}")
                

        print(f"{'='*60}")
        del env_load
