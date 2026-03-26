import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from DSTc_LG_utils import lex_hull_corrected, generate_all_priority_orders, lex_max
from DSTc_utils import step_from
from CH_operations import get_hull, translate_hull, sum_hulls


def LG_VI_lexhull(env, theta=1.0, discount_factor=0.99, MNS_filename='dstc_policies/DSTc_LGVI_lexhull_MNS.pkl', v_hulls_file=None, q_hulls_file=None):    
    """
    Lexicographic Value Iteration Algorithm using hull-based convergence.
    
    Maintains the lexicographic hull using proper convex hull operations
    in the Bellman update.
    Args:
        env: the environment
        theta: convergence threshold
        discount_factor: discount factor
        MNS_filename: path to pickle file for caching model transitions
        v_hulls_file: optional filename to save the V hulls
        q_hulls_file: optional filename to save the Q hulls
    
    Returns:
        policies: dict of policies for all priority orders
        Q_hulls: Q hulls for each state-action pair
    """
    
    n_actions = env.n_actions    # 4
    n_objectives = env.n_rewards # 2

    # V stores hulls (sets of vectors) for each state
    V = {}
    for (row, col) in env.non_terminal_states:
        V[(row, col)] = np.zeros((1, n_objectives))

    Q_hulls = {}
    for (row, col) in env.non_terminal_states:
        for a in range(n_actions):
            Q_hulls[(row, col, a)] = np.array([np.zeros(n_objectives)])

    os.makedirs(os.path.dirname(MNS_filename) if  os.path.dirname(MNS_filename) else '.', exist_ok=True)

    if os.path.exists(MNS_filename):
        print("Initialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("Initialising model_next_state = {}")
        model_next_state = {}

    iteration = 0
    total_states = len(env.non_terminal_states)

    print(f"Starting Lexicographic Hull Value Iteration with lexhull")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")

    while True:
        iteration += 1
        print(f"\nIteration {iteration}")
        delta = 0
        total_hull_vertices = 0
        num_hulls = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            for (row, col) in env.non_terminal_states:
                state_tuple = (row, col)

                v_old = V[state_tuple].copy()

                for action in range(n_actions):
                    if (row, col, action) not in model_next_state:
                        next_state, reward_vect, done = step_from(env.gym_env, row, col, action)
                        prob = 1.0
                        outcomes = [(next_state, reward_vect, done, prob)]
 
                        model_next_state[(row, col, action)] = outcomes
 
                    else:
                        outcomes = model_next_state[(row, col, action)]

                    outcome_hulls = []
                    for next_state, reward_vect, done, prob in outcomes:
                        if done:
                            # terminal state 
                            outcome_hull = prob * np.array([reward_vect])
                        else:
                            next_row, next_col = next_state
                            next_state_hull = V[(next_row, next_col)]

                            # translate_hull does: reward + gamma * hull
                            outcome_hull = translate_hull(
                                reward_vect,
                                discount_factor,
                                next_state_hull
                            )

                            if not isinstance(outcome_hull, np.ndarray):
                                outcome_hull = np.array(outcome_hull)

                            outcome_hull = prob * outcome_hull

                        outcome_hulls.append(outcome_hull)

                    # Combine outcomes using Minkowski sum 
                    if len(outcome_hulls) == 1:
                        # Deterministic 
                        new_q_hull = outcome_hulls[0]
                    else:
                        # Stochastic Minkowski sum of scaled hulls
                        combined_hull = outcome_hulls[0]
                        for outcome_hull in outcome_hulls[1:]:
                            combined_hull = sum_hulls(combined_hull, outcome_hull)
                        new_q_hull = combined_hull

                    if not isinstance(new_q_hull, np.ndarray):
                        new_q_hull = np.array(new_q_hull)

                    if len(new_q_hull) > 1:
                        _, optimal_indices = lex_hull_corrected(env, new_q_hull)
                        new_q_hull = new_q_hull[list(optimal_indices)]

                    Q_hulls[(row, col, action)] = new_q_hull

                all_q_vectors = []
                for action in range(n_actions):
                    q_hull = Q_hulls[(row, col, action)]
                    if isinstance(q_hull, np.ndarray):
                        all_q_vectors.extend(q_hull)
                    else:
                        all_q_vectors.extend(list(q_hull))

                all_q_vectors = np.array(all_q_vectors)

                if len(all_q_vectors) > 1:
                    _, optimal_indices = lex_hull_corrected(env, all_q_vectors)
                    new_V = all_q_vectors[list(optimal_indices)]
                else:
                    new_V = all_q_vectors

                # Sort for consistent comparison
                new_V = new_V[np.lexsort(new_V.T[::-1])]

                V[state_tuple] = new_V

                # Track 
                new_hull_size = len(new_V)
                total_hull_vertices += new_hull_size
                num_hulls += 1

                # Convergence 
                if v_old.shape == new_V.shape:
                    max_diff = np.max(np.abs(new_V - v_old))
                else:
                    max_diff = float('inf')

                delta = max(delta, max_diff)

                pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {round(delta, 3)}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state")

        if delta < theta:
            print(f"\nConverged in {iteration} iterations")
            break

    # Save model
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

    print("\nExtracting policies for all lexicographic orders...")
    all_priority_orders = generate_all_priority_orders(n_objectives)
    policies = {}

    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy = np.zeros([env.n_rows, env.n_cols], dtype=int)

        for (row, col) in env.non_terminal_states:
            state_tuple = (row, col)

            # For each action, get its best Q-vector for this priority
            q_vectors = []
            for action in range(n_actions):
                action_hull = Q_hulls[state_tuple + (action,)]
                best_idx    = lex_max(action_hull, priority=list(priority_order))
                q_vectors.append(action_hull[best_idx])
 
            q_vectors = np.array(q_vectors)

            # Find best action for this priority
            best_action          = lex_max(q_vectors, priority=list(priority_order))
            policy[row, col]     = best_action
 
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")
 
    return policies, Q_hulls



def get_initial_state_hull(Q_hulls, env, n_actions):
    """
    Get the value hull for the initial state (0, 0) by taking the convex
    hull of all Q-vectors across actions.
    """
    row, col = env.start_state
    
    all_vectors = []
    for action in range(n_actions):
        hull = Q_hulls[(row, col, action)]
        if isinstance(hull, np.ndarray):
            all_vectors.extend(hull)
        else:
            all_vectors.extend(list(hull))
 
    all_vectors = np.array(all_vectors)
    
    # Get the convex hull of all Q-vectors
    if len(all_vectors) > 1:
        _, optimal_indices = lex_hull_corrected(env, all_vectors)
        value_hull = all_vectors[list(optimal_indices)]
    else:
        value_hull = all_vectors
 
    return value_hull
