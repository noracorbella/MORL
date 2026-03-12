import numpy as np
import pickle
import sys
import os

# ##############################################################
# DOMAIN CONSTANTS
# ##############################################################
DETERMINISTIC = 0
STOCHASTIC    = 1
LEXICOGRAPHIC = 2

# Deterministic sub-algorithms
DET_Q_LEARNING      = 0
DET_VALUE_ITERATION = 1
DET_CONVEX_HULL_VI  = 2

# Stochastic sub-algorithms
STOC_VALUE_ITERATION = 0
STOC_CONVEX_HULL_VI  = 1

# Lexicographic sub-algorithms
LEX_LGVI_LEXMAX  = 0
LEX_LGVI_LEXHULL = 1

# ##############################################################
# CONFIGURATION  
# ##############################################################
if __name__ == "__main__":

    # -- Top-level choice --------------------------------------
    domain = STOCHASTIC     # DETERMINISTIC | STOCHASTIC | LEXICOGRAPHIC

    # -- Deterministic settings --------------------------------
    det_algorithm       = DET_VALUE_ITERATION   # DET_Q_LEARNING | DET_VALUE_ITERATION | DET_CONVEX_HULL_VI
    det_weights         = [1, 10, 100]
    det_Training        = True   # True: train and save. False: load existing policy.
    det_Calculate_hulls = True   # (CONVEX_HULL_VI only) True: compute. False: extract from saved hulls.
    det_Test            = True   # True: visualise policy after training/loading.

    # -- Stochastic settings -----------------------------------
    stoc_algorithm       = STOC_VALUE_ITERATION  # STOC_VALUE_ITERATION | STOC_CONVEX_HULL_VI
    stoc_weights         = [1, 100, 10000]
    stoc_Training        = False
    stoc_Calculate_hulls = True
    stoc_Test            = True
    stoc_Evaluate_policy = True
    stoc_n_eval_episodes = 10000
    stoc_max_steps       = 200
    stoc_discount_factor = 0.7

    # -- Lexicographic settings --------------------------------
    lex_algorithm       = LEX_LGVI_LEXMAX   # LEX_LGVI_LEXMAX | LEX_LGVI_LEXHULL
    lex_priority        = [2, 1, 0]         # priority order over [r_car, r_ped1, r_ped2]
    lex_Training        = False
    lex_Test            = True
    lex_Evaluate_policy = True
    lex_n_eval_episodes = 10000
    lex_max_steps       = 200
    lex_discount_factor = 0.7

    # ##############################################################
    # EXECUTION
    # ##############################################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # #################### DETERMINISTIC ####################
    if domain == DETERMINISTIC:
        sys.path.insert(0, os.path.join(BASE_DIR, 'ADS_code'))
        from ADS_Environment import Environment
        from auxiliary_functions import tester

        weights         = det_weights
        algorithm_used  = det_algorithm
        Training        = det_Training
        Calculate_hulls = det_Calculate_hulls
        Test            = det_Test
        w               = weights

        PDIR = os.path.join(BASE_DIR, 'ADS_code', 'deterministic_policies')

        if algorithm_used == DET_Q_LEARNING:
            from QL_deterministic import q_learning
            train_policy_name = os.path.join(PDIR, f"QL_deterministic_{w[0]}-{w[1]}-{w[2]}-policy.npy")
            test_policy_name  = train_policy_name

        elif algorithm_used == DET_VALUE_ITERATION:
            from VI_deterministic import value_iteration
            train_policy_name = os.path.join(PDIR, f"VI_deterministic_{w[0]}-{w[1]}-{w[2]}-policy.npy")
            test_policy_name  = train_policy_name
            v_table_name      = os.path.join(PDIR, f"V_table_deterministic_{w[0]}-{w[1]}-{w[2]}.pkl")

        elif algorithm_used == DET_CONVEX_HULL_VI:
            from CHVI_deterministic import convexhull_VI, extract_policy_for_weights
            hulls_name       = os.path.join(PDIR, "CHVI_deterministic_qhulls.npy")
            chvi_policy_name = os.path.join(PDIR, f"CHVI_deterministic_{w[0]}-{w[1]}-{w[2]}-policy.npy")

        # -- Run --
        if algorithm_used == DET_CONVEX_HULL_VI:
            if Calculate_hulls:
                env = Environment(weights=weights)
                policy, q_hulls = convexhull_VI(env, theta=0.01, discount_factor=0.7)
                np.save(hulls_name, q_hulls, allow_pickle=True)
                np.save(chvi_policy_name, policy)
                print(f"Saved Q-hulls to {hulls_name}")
                print(f"Saved policy to {chvi_policy_name}")
            else:
                print(f"Loading Q-hulls from {hulls_name}...\n")
                q_hulls = np.load(hulls_name, allow_pickle=True).item()
                env = Environment(weights=weights)
                policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
                np.save(chvi_policy_name, policy)
                print(f"Saved policy to {chvi_policy_name}\n")
        else:
            if Training:
                algo_name = ['Q-Learning', 'Value Iteration'][algorithm_used]
                print(f"Training {algo_name}\n")
                print(f"Weights: {weights}\n")
                env = Environment(weights=weights)
                env.reset()
                if algorithm_used == DET_Q_LEARNING:
                    policy, q = q_learning(env, alpha=0.8, gamma=0.7)
                elif algorithm_used == DET_VALUE_ITERATION:
                    policy, q = value_iteration(env, discount_factor=0.7)
                np.save(train_policy_name, policy)
                print(f"Saved policy to {train_policy_name}")
                print("-------------------\nFinished!!!")
            else:
                print(f"Loading policy from {test_policy_name}...\n")
                policy = np.load(test_policy_name)
                print("Policy loaded successfully\n")
                env = Environment(weights=weights)

        if Test:
            print("\nTesting policy...\n")
            tester(env, policy, drawing=True)

    # #################### STOCHASTIC ####################
    elif domain == STOCHASTIC:
        sys.path.insert(0, os.path.join(BASE_DIR, 'Stochastic'))
        from ADS_Environment import Environment
        from auxiliary_functions import tester
        from evaluate_policies import evaluate_policy

        weights         = stoc_weights
        algorithm_used  = stoc_algorithm
        Training        = stoc_Training
        Calculate_hulls = stoc_Calculate_hulls
        Test            = stoc_Test
        Evaluate_policy = stoc_Evaluate_policy
        n_eval_episodes = stoc_n_eval_episodes
        max_steps       = stoc_max_steps
        discount_factor = stoc_discount_factor
        w               = weights

        PDIR = os.path.join(BASE_DIR, 'Stochastic', 'stochastic_policies')

        if algorithm_used == STOC_VALUE_ITERATION:
            from VI_stochastic import value_iteration
            train_policy_name = os.path.join(PDIR, f"VI_stochastic_{w[0]}-{w[1]}-{w[2]}-policy.npy")
            test_policy_name  = train_policy_name
            v_table_name      = os.path.join(PDIR, f"V_table_{w[0]}-{w[1]}-{w[2]}.pkl")

        elif algorithm_used == STOC_CONVEX_HULL_VI:
            from CHVI_stochastic import convexhull_VI, extract_policy_for_weights, get_initial_state_hull
            hulls_name       = os.path.join(PDIR, "CHVI_stochastic_qhulls.npy")
            chvi_policy_name = os.path.join(PDIR, f"CHVI_stochastic_{w[0]}-{w[1]}-{w[2]}-policy.npy")

        # -- Run --
        if algorithm_used == STOC_CONVEX_HULL_VI:
            if Calculate_hulls:
                env = Environment(weights=None)
                env.weights = weights
                q_hulls_filename = os.path.join(PDIR, "CHVI_stochastic_qhulls.pkl")
                q_hulls = convexhull_VI(env, theta=0.01, discount_factor=discount_factor,
                                        q_hulls_file=q_hulls_filename)
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

        else:  # STOC_VALUE_ITERATION
            if Training:
                print(f"Training Value Iteration (Stochastic)\n")
                print(f"Weights: {weights}\n")
                env = Environment(weights=weights)
                env.reset()
                policy, q = value_iteration(env, discount_factor=discount_factor,
                                            v_table_file=v_table_name)
                np.save(train_policy_name, policy)
                print(f"Saved policy to {train_policy_name}\n")
                print("-------------------\nFinished!!!")
            else:
                print(f"Loading policy from {test_policy_name}...\n")
                policy = np.load(test_policy_name)
                print("Policy loaded successfully\n")
                env = Environment(weights=weights)

        if Test:
            print("\nTesting policy...\n")
            tester(env, policy, drawing=True)

        if Evaluate_policy:
            print(f"Episodes: {n_eval_episodes}, Max steps: {max_steps}, Discount: {discount_factor}")
            Q_hulls = None

            if algorithm_used == STOC_VALUE_ITERATION:
                policy_file = os.path.join(PDIR, f"VI_stochastic_{w[0]}-{w[1]}-{w[2]}-policy.npy")
                try:
                    policy = np.load(policy_file)
                    print(f"\nLoaded VI policy from {policy_file}")
                except FileNotFoundError:
                    print(f"\nERROR: Policy file not found: {policy_file}")
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

    # #################### LEXICOGRAPHIC ####################
    elif domain == LEXICOGRAPHIC:
        sys.path.insert(0, os.path.join(BASE_DIR, 'Lexicographic'))
        from ADS_Environment import Environment
        from auxiliary_functions import tester
        from evaluate_policies import evaluate_policy
        from LG_utils import lex_max

        algorithm_used  = lex_algorithm
        priority        = lex_priority
        Training        = lex_Training
        Test            = lex_Test
        Evaluate_policy = lex_Evaluate_policy
        n_eval_episodes = lex_n_eval_episodes
        max_steps       = lex_max_steps
        discount_factor = lex_discount_factor

        PDIR         = os.path.join(BASE_DIR, 'Lexicographic', 'lexicographic_policies')
        priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"

        if algorithm_used == LEX_LGVI_LEXMAX:
            from LGVI_stochastic_lexmax import LG_VI_lexmax
            train_policy_name = os.path.join(PDIR, f"LGVI_lexmax_{priority_str}_policy.npy")
            test_policy_name  = train_policy_name
            v_table_name      = os.path.join(PDIR, f"LGVI_lexmax_{priority_str}_V_table.pkl")

        elif algorithm_used == LEX_LGVI_LEXHULL:
            from LGVI_stochastic_lexhull import LG_VI_lexhull
            v_hulls_file     = os.path.join(PDIR, "LGVI_lexhull_V_hulls.pkl")
            q_hulls_file     = os.path.join(PDIR, "LGVI_lexhull_Q_hulls.pkl")
            test_policy_name = os.path.join(PDIR, f"LGVI_lexhull_{priority_str}_policy.npy")

        # -- Run --
        if Training:
            if algorithm_used == LEX_LGVI_LEXMAX:
                print("Training Lexicographic Value Iteration (lexmax)")
                print(f"Priority order: {priority}\n")
                env = Environment(weights=None)
                policy, Q = LG_VI_lexmax(
                    env,
                    theta=1.0,
                    discount_factor=discount_factor,
                    priority=priority,
                    v_table_file=v_table_name
                )
                np.save(train_policy_name, policy)
                print(f"Saved policy to {train_policy_name}\n")
                print("-------------------\nFinished!!!")

            elif algorithm_used == LEX_LGVI_LEXHULL:
                print("Training Lexicographic Value Iteration (lexhull)")
                print("Trains once and extracts policies for all 6 priority orders\n")
                env = Environment(weights=None)
                policies, Q_hulls = LG_VI_lexhull(
                    env,
                    theta=1.0,
                    discount_factor=discount_factor,
                    v_hulls_file=v_hulls_file,
                    q_hulls_file=q_hulls_file
                )
                print("\nSaving all policies...")
                for priority_tuple, pol in policies.items():
                    p_str = f"{priority_tuple[0]}-{priority_tuple[1]}-{priority_tuple[2]}"
                    policy_name = os.path.join(PDIR, f"LGVI_lexhull_{p_str}_policy.npy")
                    np.save(policy_name, pol)
                    print(f"  Saved policy for priority {list(priority_tuple)}")
                print("\n-------------------\nFinished!!!")
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
            print(f"Episodes: {n_eval_episodes}, Max steps: {max_steps}, Discount: {discount_factor}")
            theoretical_value = None
            value_hull        = None

            if algorithm_used == LEX_LGVI_LEXMAX:
                policy_file  = os.path.join(PDIR, f"LGVI_lexmax_{priority_str}_policy.npy")
                v_table_file = os.path.join(PDIR, f"LGVI_lexmax_{priority_str}_V_table.pkl")
                try:
                    policy = np.load(policy_file)
                    print(f"\nLoaded lexmax policy from {policy_file}")
                except FileNotFoundError:
                    print(f"\nERROR: Policy file not found: {policy_file}")
                    print("Run lexmax training first.")
                    exit(1)
                env_tmp = Environment(weights=None)
                try:
                    with open(v_table_file, 'rb') as f:
                        V = pickle.load(f)
                    env_tmp.reset()
                    s = env_tmp.get_state()
                    theoretical_value = V[s[0], s[1], s[2]]
                except FileNotFoundError:
                    pass
                del env_tmp

            elif algorithm_used == LEX_LGVI_LEXHULL:
                policy_file  = os.path.join(PDIR, f"LGVI_lexhull_{priority_str}_policy.npy")
                v_hulls_file = os.path.join(PDIR, "LGVI_lexhull_V_hulls.pkl")
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
                    s      = env_temp.get_state()
                    v_hull = V_hulls[(s[0], s[1], s[2])]
                    best_idx = lex_max(v_hull, priority=list(priority))
                    theoretical_value, value_hull = v_hull[best_idx], v_hull
                except FileNotFoundError:
                    pass
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
                print("\nTHEORETICAL VALUE")
                print(f"V[initial_state] = {theoretical_value[0]:.4f}, "
                      f"{theoretical_value[1]:.4f}, {theoretical_value[2]:.4f}")
                if algorithm_used == LEX_LGVI_LEXHULL and value_hull is not None:
                    print(f"Hull size at initial state: {len(value_hull)} vertices")

            print(f"\nEVALUATE POLICY RESULTS")
            print(f"Number of episodes: {len(results['episode_returns'])}")
            print(f"Mean episode length: {results['mean_length']:.2f}")

            mean_vec = results['mean_return']
            std_vec  = results['std_return']
            print(f"\nMean discounted vector return:")
            print(f"  [r_car, r_ped1, r_ped2] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")
            print(f"\nStd discounted vector return:")
            print(f"  [r_car, r_ped1, r_ped2] = [{std_vec[0]:.4f}, {std_vec[1]:.4f}, {std_vec[2]:.4f}]")
