import numpy as np
from tqdm import tqdm
import pickle
import os
import sys
from scipy.spatial.distance import directed_hausdorff
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from BB_LG_utils import lex_hull_corrected_fast, generate_all_priority_orders, lex_max
from BB_utils import get_outcomes
from CH_operations import translate_hull, sum_hulls


def LG_VI_lexhull(env, theta=1.0, discount_factor=0.95,
                  MNS_filename='bb_policies/BB_LGVI_lexhull_MNS.pkl',
                  v_hulls_file=None, q_hulls_file=None):
    """
    Lexicographic Hull Value Iteration for Breakable Bottles (stochastic MOMDP).
    """

    n_actions    = env.n_actions  # 3
    n_objectives = env.n_rewards  # 3
    size         = env.size       # 5
 
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

    print(f"Starting Lexicographic Hull Value Iteration")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")

    t_start = time.time()
    t_model_done = None
    
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

                            outcome_hull    = translate_hull(reward_vect, discount_factor, next_state_hull)

                            if not isinstance(outcome_hull, np.ndarray):
                                outcome_hull = np.array(outcome_hull)

                            outcome_hull = prob * outcome_hull

                        outcome_hulls.append(outcome_hull)

                    if len(outcome_hulls) == 1:
                        # Deterministic
                        new_q_hull = outcome_hulls[0]
                    else:
                        # Stochastic — Minkowski sum of scaled hulls
                        combined_hull = outcome_hulls[0]
                        for outcome_hull in outcome_hulls[1:]:
                            combined_hull = sum_hulls(combined_hull, outcome_hull)
                        new_q_hull = combined_hull

                    if not isinstance(new_q_hull, np.ndarray):
                        new_q_hull = np.array(new_q_hull)

                    if len(new_q_hull) > 1:
                        _, optimal_indices = lex_hull_corrected_fast(all_priority_orders, new_q_hull)
                        new_q_hull = new_q_hull[list(optimal_indices)]

                    Q_hulls[(*state, action)] = new_q_hull

                
                # V(s) = lex_hull( union of Q_hulls across actions )
                # all_q_vectors = np.vstack([Q_hulls[(*state, a)] for a in range(n_actions)])
                
                # V(s) = lex hull of union of all Q-hulls across actions
                all_q_vectors = []
                for action in range(n_actions):
                    q_hull = Q_hulls[(*state, action)]
                    if isinstance(q_hull, np.ndarray):
                        all_q_vectors.extend(q_hull)
                    else:
                        all_q_vectors.extend(list(q_hull))

                all_q_vectors = np.array(all_q_vectors)

                if len(all_q_vectors) > 1:
                    _, optimal_indices = lex_hull_corrected_fast(all_priority_orders, all_q_vectors)
                    new_V = all_q_vectors[list(optimal_indices)]
                else:
                    new_V = all_q_vectors
                
                new_V = new_V[np.lexsort(new_V.T[::-1])]

                V[state] = new_V

                total_hull_vertices += len(new_V)
                num_hulls           += 1

                # if v_old.shape == new_V.shape:
                #     max_diff = np.max(np.abs(new_V - v_old))
                # else:
                #     max_diff = float('inf')

                # delta = max(delta, max_diff)
                if v_old.shape == new_V.shape:
                    max_diff = np.max(np.abs(new_V - v_old))
                else:
                    d1 = directed_hausdorff(v_old, new_V)[0]
                    d2 = directed_hausdorff(new_V, v_old)[0]
                    max_diff = max(d1, d2)
                    # max_diff = float('inf')

                delta = max(delta, max_diff)

                pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {round(delta, 3)}, Theta = {theta}")
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

    # Extract policies for all 3! = 6 priority orders
    print("\nExtracting policies for all lexicographic orders...")
    policies = {}
    extraction_times = {}

    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy = np.zeros([size, 3, 3, 2, 2, 2], dtype=int)
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
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")
 
    t_extract_done = time.time()

    timing = {
        "model_build":       t_model_done - t_start,
        "remaining_vi":      t_vi_done    - t_model_done,
        "policy_extraction": t_extract_done - t_vi_done,
        "total":             t_extract_done - t_start,
        "extraction_times":  extraction_times,
    }

    return policies, Q_hulls, timing


def get_initial_state_hull(Q_hulls, env, n_actions, all_priority_orders=None):
    if all_priority_orders is None:
        all_priority_orders = generate_all_priority_orders(env.n_rewards)
    state = env.start_state
    all_vectors = np.vstack([Q_hulls[(*state, a)] for a in range(n_actions)])
    if len(all_vectors) > 1:
        _, optimal_indices = lex_hull_corrected_fast(all_priority_orders, all_vectors)
        return all_vectors[list(optimal_indices)]
    return all_vectors