import numpy as np
from tqdm import tqdm
import pickle
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from FT_LG_utils import lex_hull_corrected, lex_hull_fast, generate_all_priority_orders, lex_max
from FT_utils import get_outcomes
from CH_operations import get_hull, translate_hull, sum_hulls


def LG_VI_lexhull(env, theta=1e-4, discount_factor=1.0,
                  MNS_filename='ft_policies/FT_LGVI_lexhull_MNS.pkl',
                  v_hulls_file=None, q_hulls_file=None):
    """
    Lexicographic Hull Value Iteration for Fruit Tree — with timing instrumentation.

    Adapted from DST_LGVI_lexhull_timing.py. Changes vs DST timing version:
      - Imports FT_LG_utils and ft_utils instead of DST equivalents
      - State is (depth, node) 2-tuple; keys are (*state, action)
      - n_actions = 2, n_objectives = 6 -> 6! = 720 priority orders
      - get_outcomes() replaces step_from() — deterministic, 1 outcome always
      - policy shape [tree_depth+1, 2^tree_depth] instead of [n_rows, n_cols]
      - iterates over env.non_terminal_states as (depth, node) tuples
      - policy[state] tuple indexing instead of policy[row, col]

    Returns:
        policies        : dict mapping priority tuple -> policy array
        Q_hulls         : dict mapping (*state, action) -> hull
        timing          : dict with model_build, remaining_vi, policy_extraction,
                          total, extraction_times
    """

    n_actions    = env.n_actions   # 2
    n_objectives = env.n_rewards   # 6

    V = {}
    for state in env.non_terminal_states:
        V[state] = np.zeros((1, n_objectives))

    Q_hulls = {}
    for state in env.non_terminal_states:
        for a in range(n_actions):
            Q_hulls[(*state, a)] = np.array([np.zeros(n_objectives)])

    os.makedirs(os.path.dirname(MNS_filename) if os.path.dirname(MNS_filename) else '.', exist_ok=True)

    if os.path.exists(MNS_filename):
        print("Initialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("Initialising model_next_state = {}")
        model_next_state = {}

    iteration    = 0
    total_states = len(env.non_terminal_states)

    print(f"Starting Lexicographic Hull Value Iteration with lexhull")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")

    t_start      = time.time()
    t_model_done = None  # set at end of iteration 1

    all_priority_orders = generate_all_priority_orders(n_objectives)
    while True:
        iteration += 1
        print(f"\nIteration {iteration}")

        delta               = 0
        total_hull_vertices = 0
        num_hulls           = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:

            for state in env.non_terminal_states:

                v_old = V[state].copy()

                for action in range(n_actions):

                    if (*state, action) not in model_next_state:
                        outcomes = get_outcomes(env, state, action)
                        model_next_state[(*state, action)] = outcomes
                    else:
                        outcomes = model_next_state[(*state, action)]

                    outcome_hulls = []
                    for next_state, reward_vect, done, prob in outcomes:
                        if done:
                            outcome_hull = prob * np.array([reward_vect])
                        else:
                            next_state_hull = V[next_state]
                            outcome_hull = translate_hull(
                                reward_vect,
                                discount_factor,
                                next_state_hull
                            )
                            if not isinstance(outcome_hull, np.ndarray):
                                outcome_hull = np.array(outcome_hull)
                            outcome_hull = prob * outcome_hull
                        outcome_hulls.append(outcome_hull)

                    # Deterministic: always 1 outcome
                    new_q_hull = outcome_hulls[0]

                    if not isinstance(new_q_hull, np.ndarray):
                        new_q_hull = np.array(new_q_hull)

                    if len(new_q_hull) > 1:
                        new_q_hull = lex_hull_fast(new_q_hull, all_priority_orders)

                    Q_hulls[(*state, action)] = new_q_hull

                all_q_vectors = []
                for action in range(n_actions):
                    q_hull = Q_hulls[(*state, action)]
                    if isinstance(q_hull, np.ndarray):
                        all_q_vectors.extend(q_hull)
                    else:
                        all_q_vectors.extend(list(q_hull))

                all_q_vectors = np.array(all_q_vectors)

                if len(all_q_vectors) > 1:
                    new_V = lex_hull_fast(all_q_vectors, all_priority_orders)
                else:
                    new_V = all_q_vectors


                new_V = new_V[np.lexsort(new_V.T[::-1])]

                V[state] = new_V

                new_hull_size        = len(new_V)
                total_hull_vertices += new_hull_size
                num_hulls           += 1

                if v_old.shape == new_V.shape:
                    max_diff = np.max(np.abs(new_V - v_old))
                else:
                    max_diff = float('inf')

                delta = max(delta, max_diff)

                pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {round(delta, 6)}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state")

        if iteration == 1:
            t_model_done = time.time()

        if delta < theta:
            print(f"\nConverged in {iteration} iterations")
            t_vi_done = time.time()
            break

    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")

    if v_hulls_file is not None:
        os.makedirs(os.path.dirname(v_hulls_file) if os.path.dirname(v_hulls_file) else '.', exist_ok=True)
        with open(v_hulls_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"V hulls saved to {v_hulls_file}")

    if q_hulls_file is not None:
        os.makedirs(os.path.dirname(q_hulls_file) if os.path.dirname(q_hulls_file) else '.', exist_ok=True)
        with open(q_hulls_file, 'wb') as f:
            pickle.dump(Q_hulls, f)
        print(f"Q hulls saved to {q_hulls_file}")

    # Extract policies for all 6! = 720 priority orders
    print("\nExtracting policies for all lexicographic orders...")
    all_priority_orders = generate_all_priority_orders(n_objectives)
    tree_depth = env.tree_depth
    max_nodes  = 2 ** tree_depth
    policies         = {}
    extraction_times = {}

    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy         = np.zeros([tree_depth + 1, max_nodes], dtype=int)
        t_policy_start = time.time()

        for state in env.non_terminal_states:
            q_vectors = []
            for action in range(n_actions):
                action_hull = Q_hulls[(*state, action)]
                best_idx    = lex_max(action_hull, priority=list(priority_order))
                q_vectors.append(action_hull[best_idx])

            q_vectors     = np.array(q_vectors)
            best_action   = lex_max(q_vectors, priority=list(priority_order))
            policy[state] = best_action

        extraction_times[priority_tuple] = time.time() - t_policy_start
        policies[priority_tuple]         = policy

    print(f"Extracted policies for all {len(all_priority_orders)} priority orders")

    t_extract_done = time.time()

    timing = {
        "model_build":       t_model_done   - t_start,
        "remaining_vi":      t_vi_done      - t_model_done,
        "policy_extraction": t_extract_done - t_vi_done,
        "total":             t_extract_done - t_start,
        "extraction_times":  extraction_times,
    }

    return policies, Q_hulls, timing


def get_initial_state_hull(Q_hulls, env, n_actions):
    """Get the lex-optimal hull at the root (start state)."""
    state = env.start_state

    all_vectors = []
    for action in range(n_actions):
        hull = Q_hulls[(*state, action)]
        if isinstance(hull, np.ndarray):
            all_vectors.extend(hull)
        else:
            all_vectors.extend(list(hull))

    all_vectors = np.array(all_vectors)

    if len(all_vectors) > 1:
        all_priority_orders = generate_all_priority_orders(env.n_rewards)
        value_hull = lex_hull_fast(all_vectors, all_priority_orders)
    else:
        value_hull = all_vectors

    return value_hull