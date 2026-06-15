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

    # -- VI and CHVI settings ----------------------------------
    weights             = [1, 100] # [w_treasure, w_time]
    chvi_priority       = [1, 0]   # priority order for policy extraction (CONVEX_HULL_VI only)
    Training            = True  # True: train and save policy, False: load policy
    Calculate_hulls     = True  # For COVNEX_HULL_VI True: compute, False: extract from saved hulls

    # -- Lexicographic settings --------------------------------
    priority         = [1, 0] 
    lex_Training     = True
    Calculate_lexhulls = True
    
    # -- Shared settings ---------------------------------------
    Test            = True # True: visualise poolicy after training/loading
    Load_V_at_initial_state = True
    Evaluate_policy = True  # True: run evaluation
    discount_factor = 0.99
    theta = 0.01
    n_eval_episodes = 1
    max_steps = 200

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
    from DST_CHVI import convexhull_VI, get_initial_state_hull
    from DST_LG_utils import extract_lex_policy_from_qhulls
    from DST_evaluate_policies import evaluate_policy
    from DST_LGVI_lexmax import LG_VI_lexmax
    from DST_LGVI_lexhull import LG_VI_lexhull, get_initial_state_hull as get_initial_state_hull_lex

    PDIR = os.path.join(BASE_DIR, 'dst_policies')
    os.makedirs(PDIR, exist_ok=True)
 
    MNS_VI   = os.path.join(PDIR, 'DST_VI_MNS.pkl')
    MNS_CHVI = os.path.join(PDIR, 'DST_CHVI_MNS.pkl')
    MNS_LEXMAX = os.path.join(PDIR, 'DST_LGVI_lexmax_MNS.pkl')
    MNS_LEXHULL = os.path.join(PDIR, 'DST_LGVI_lexhull_MNS.pkl')


    if algorithm_used == VALUE_ITERATION:
        train_policy_file = os.path.join(PDIR, f"DST_VI_{weights[0]}-{weights[1]}-policy.npy")
        test_policy_file = os.path.join(PDIR, f"DST_VI_{weights[0]}-{weights[1]}-policy.npy")
        v_table_file = os.path.join(PDIR, f"DST_V_table_{weights[0]}-{weights[1]}.pkl")
    
    elif algorithm_used == CONVEX_HULL_VI:
        hulls_file = os.path.join(PDIR, f"DST_CHVI_qhulls.npy")
        chvi_policy_file = os.path.join(PDIR, f"DST_CHVI_{chvi_priority[0]}-{chvi_priority[1]}-policy.npy")

    elif algorithm_used == LGVI_LEXMAX:
        train_policy_file = os.path.join(PDIR, f"DST_LGVI_lexmax_{priority[0]}-{priority[1]}_policy.npy")
        test_policy_file  = os.path.join(PDIR, f"DST_LGVI_lexmax_{priority[0]}-{priority[1]}_policy.npy")
        v_table_file      = os.path.join(PDIR, f"DST_LGVI_lexmax_{priority[0]}-{priority[1]}_V_table.pkl")

    elif algorithm_used == LGVI_LEXHULL:
        v_hulls_file     = os.path.join(PDIR, "DST_LGVI_lexhull_V_hulls.pkl")
        q_hulls_file     = os.path.join(PDIR, "DST_LGVI_lexhull_Q_hulls.pkl")
        test_policy_file  = os.path.join(PDIR, f"DST_LGVI_lexhull_{priority[0]}-{priority[1]}_policy.npy")
        # v_hulls_file      = os.path.join(PDIR, f"DST_LGVI_lexhull_{priority[0]}-{priority[1]}_V_hulls.pkl")
        # q_hulls_file      = os.path.join(PDIR, f"DST_LGVI_lexhull_{priority[0]}-{priority[1]}_q_hulls.pkl")



    # -- Run ---------------------------------------------------
    if algorithm_used == VALUE_ITERATION:
        if Training:
            print(f"Training Value Iteration\n")
            print(f"Weights: {weights}\n")

            env = DSTEnvironment(weights=weights)

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
                print("WARNING: Notice that the policy that will appear now is not the policy previously trained!!")

            print(f"Loading policy from {test_policy_file}...\n")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")

            env = DSTEnvironment(weights=weights)
    
    elif algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = DSTEnvironment(weights=None)
            q_hulls = convexhull_VI(env, theta=theta, discount_factor=discount_factor, MNS_filename=MNS_CHVI, q_hulls_file=hulls_file)
            policy = extract_lex_policy_from_qhulls(q_hulls, chvi_priority, env, env.n_actions)
            np.save(hulls_file, q_hulls, allow_pickle=True)
            np.save(chvi_policy_file, policy)
            print(f"Saved Q-hulls to {hulls_file}")
            print(f"Saved policy to {chvi_policy_file}")
        else:
            print(f"Loading Q-hulls from {hulls_file}...\n")
            q_hulls = np.load(hulls_file, allow_pickle=True).item()
            env = DSTEnvironment(weights=None)
            policy = extract_lex_policy_from_qhulls(q_hulls, chvi_priority, env, env.n_actions)
            np.save(chvi_policy_file, policy)
            print(f"Saved policy to {chvi_policy_file}\n")        
    
    elif algorithm_used == LGVI_LEXMAX:
        if lex_Training:
            print("Training Lexicographic Value Iteration (lexmax)")
            print(f"Priority order: {priority}\n")
            env = DSTEnvironment(weights=None)
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
            env = DSTEnvironment(weights=None)

    elif algorithm_used == LGVI_LEXHULL:
        if Calculate_lexhulls:
            print("Training Lexicographic Value Iteration (lexhull)")
            env = DSTEnvironment(weights=None)
            policies, Q_hulls = LG_VI_lexhull(
                env,
                theta=theta,
                discount_factor=discount_factor,
                MNS_filename=MNS_LEXHULL,
                v_hulls_file=v_hulls_file,
                q_hulls_file=q_hulls_file,
            )
            for priority_tuple, pol in policies.items():
                policy_name  = os.path.join(PDIR, f"DST_LGVI_lexhull_{priority_tuple[0]}-{priority_tuple[1]}_policy.npy")
                np.save(policy_name, pol)
                print(f"Saved policy for priority {list(priority_tuple)}")
            print("\n-------------------\nFinished!!!")
            policy = policies[tuple(priority)]
        else:
            print(f"Loading policy from {test_policy_file}...")
            policy = np.load(test_policy_file)
            print("Policy loaded successfully\n")
            env = DSTEnvironment(weights=None)

     # -- Test --------------------------------------------------
    if Test:
        print("\nTesting policy...\n")
        render_env = mo_gym.make("deep-sea-treasure-v0", render_mode="human")
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
                Q_hulls = np.load(hulls_file, allow_pickle=True).item()
                print(f"Loaded Q-hulls from {hulls_file}")
            except FileNotFoundError:
                print(f"WARNING: Q-hulls file not found ({hulls_file}), skipping hull report.")

            if Q_hulls is not None:
                env_tmp = DSTEnvironment(weights=None)
                value_hull = get_initial_state_hull(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}")
                print("VALUE HULL AT INITIAL STATE")
                print('='*60)
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                print(f"\nVertices (Pareto-optimal value vectors):")
                print(f"  [  treasure ,    time   ]")
                for v in value_hull:
                    print(f"  [{v[0]:10.4f}, {v[1]:10.4f}]")
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
            env_tmp = DSTEnvironment(weights=None)
            try:
                with open(v_table_file, 'rb') as f:
                    V = pickle.load(f)
                row, col = env_tmp.start_state
            except FileNotFoundError:
                pass
            del env_tmp
        
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
                env_tmp    = DSTEnvironment(weights=None)
                value_hull = get_initial_state_hull_lex(Q_hulls, env_tmp, env_tmp.n_actions)
                print(f"\n{'='*60}")
                print("VALUE HULL AT INITIAL STATE")
                print(f"{'='*60}")
                print(f"Initial state: {env_tmp.start_state}")
                print(f"Number of vertices: {len(value_hull)}")
                print(f"\nVertices (lex-optimal value vectors):")
                print(f"  [  treasure ,    time   ]")
                for v in value_hull:
                    print(f"  [{v[0]:10.4f}, {v[1]:10.4f}]")
                print(f"{'='*60}")
                del env_tmp




        env = DSTEnvironment(weights=None)
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
        print(f"\nMean discounted vector return:")
        print(f"  [r_treasure, r_time] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}]")
        print(f"\nStd discounted vector return:")
        print(f"  [r_treasure, r_time] = [{std_vec[0]:.4f}, {std_vec[1]:.4f}]")

        if weights is not None:
            print(f"Weights: {weights}")
            weights_arr        = np.array(weights)
            scalarised_mean    = np.dot(mean_vec, weights_arr)
            scalarised_returns = np.dot(results['episode_returns'], weights_arr)
            print(f"  Mean scalarised return: {scalarised_mean:.4f}")
            print(f"  Std scalarised return:  {np.std(scalarised_returns):.4f}")


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
            load_file = os.path.join(PDIR, f"DST_V_table_{weights[0]}-{weights[1]}_vec.pkl")
        elif algorithm_used == CONVEX_HULL_VI:
            load_file = os.path.join(PDIR, "DST_CHVI_vhulls.pkl")
        elif algorithm_used == LGVI_LEXMAX:
            load_file = os.path.join(PDIR,f"DST_LGVI_lexmax_{priority[0]}-{priority[1]}_V_table.pkl")
        elif algorithm_used == LGVI_LEXHULL:
            load_file = os.path.join(PDIR, "DST_LGVI_lexhull_V_hulls.pkl")

        env_load = DSTEnvironment(weights=None)
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
            if "DST_V_table" in fname and "_vec" in fname:
                V_s0 = np.asarray(data[s0])
                print(f"File         : {fname}")
                print(f"Algorithm    : VI (vector)")
                print(f"Weights      : {load_weights}")
                print(f"Initial state: {s0}")

            # ---------- LexMax: vector V table ------------------------
            elif "DST_LGVI_lexmax" in fname and "V_table" in fname:
                V_s0 = np.asarray(data[s0])
                print(f"File         : {fname}")
                print(f"Algorithm    : LexMax")
                print(f"Priority     : {load_priority}")
                print(f"Initial state: {s0}")

            # ---------- CHVI: V-hulls dict ----------------------------
            elif "DST_CHVI_vhulls" in fname:
                value_hull = np.asarray(data[s0])
                ordered    = value_hull[:, load_priority]
                idx        = int(np.lexsort(ordered.T[::-1])[-1])
                V_s0       = value_hull[idx]
                print(f"File         : {fname}")
                print(f"Algorithm    : CHVI")
                print(f"Priority     : {load_priority}")
                print(f"Initial state: {s0}")
                print(f"Hull size    : {len(value_hull)} vertices")

            # ---------- LexHull: V-hulls dict -------------------------
            elif "DST_LGVI_lexhull_V_hulls" in fname:
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
                print(f"V(s0)        : [r_treasure, r_time] = "
                      f"[{V_s0[0]:.4f}, {V_s0[1]:.4f}]")

        print(f"{'='*60}")
        del env_load