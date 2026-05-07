import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from FT_utils import get_outcomes
from CH_operations import get_hull, translate_hull, sum_hulls, max_q_value

def convexhull_VI(env, theta=1, discount_factor=0.7, MNS_filename='dst_policies/DST_CHVI_MNS.pkl', q_hulls_file=None):
    """
    Convex Hull Value Iteration for Deep Sea Treasure.
    Adapted from CHVI_stochastic.py.
    """
    n_actions = env.n_actions # 2
    n_rewards = env.n_rewards # 6

    V = {}
    for state in env.non_terminal_states:
        V[state] = np.array([np.zeros(n_rewards)])

    Q_hulls = {}
    for (row, col) in env.non_terminal_states:
        for a in range(n_actions):
            Q_hulls[(*state, a)] = np.array([np.zeros(n_rewards)])
    
    os.makedirs(os.path.dirname(MNS_filename) if os.path.dirname(MNS_filename) else '.', exist_ok=True)
 
    if os.path.exists(MNS_filename):
        print("Initialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("Initialising model_next_state = {}")
        model_next_state = {}


    iteration = 0
    total_states = len(env.non_terminal_states)

    print(f"Starting CHVI with {total_states} states and {n_actions} actions")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0
        total_hull_vertices = 0
        num_hulls = 0

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
                    new_hull = outcome_hulls[0]

                    if not isinstance(new_hull, np.ndarray):
                        new_hull = np.array(new_hull)
                    
                    if len(new_hull) > 1:
                        new_hull = get_hull(new_hull)
                                                
                    # Store new Q-hull
                    Q_hulls[(*state, action)] = new_hull

                all_q_vectors = []
                for action in range(n_actions):
                    q_hull = Q_hulls[(*state, action)]
                    if isinstance(q_hull, np.ndarray):
                        all_q_vectors.extend(q_hull)
                    else:
                        all_q_vectors.extend(list(q_hull))
 
                all_q_vectors = np.array(all_q_vectors)
 
                if len(all_q_vectors) > 1:
                    new_V = get_hull(all_q_vectors)
                else:
                    new_V = all_q_vectors
 
                V[state] = new_V

                total_hull_vertices += len(new_V)
                num_hulls += 1

                if v_old.shape == new_V.shape:
                    max_diff = np.max(np.abs(new_V - v_old))
                else:
                    max_diff = float('inf')
                
                delta = max(delta, max_diff)

                pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {delta}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state-action pair")
            

        # Convergence
        if delta < theta:
            print(f"Converged in {iteration} iterations")
            break
    
    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")
 
    if q_hulls_file is not None:
        os.makedirs(os.path.dirname(q_hulls_file) if os.path.dirname(q_hulls_file) else '.', exist_ok=True)
        with open(q_hulls_file, 'wb') as f:
            pickle.dump(Q_hulls, f)
        print(f"Q_hulls saved to {q_hulls_file}")
 
    return Q_hulls


def extract_policy_for_weights(Q_hulls, weights, env, n_actions):
    """
    Extract the optimal policy for a specific weight vector.
    """
    tree_depth = env.tree_depth
    max_nodes = 2 ** tree_depth
    policy = np.zeros([tree_depth + 1, max_nodes], dtype=int)
 
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)
 
    print(f"\nExtracting policy for weights: {weights}")
 
    for state in env.non_terminal_states:
        best_value  = -np.inf
        best_action = 0
                
        # For each action, find the best Q-value for this weight vector
        for action in range(n_actions):
            hull = Q_hulls[(*state, action)]

            if not isinstance(hull, np.ndarray):
                hull = np.array(hull)

            # Extract max Q-value for this weight vector
            # max_{q in hull} w · q
            q_value = max_q_value(weights, hull)
                
            if q_value > best_value:
                best_value = q_value
                best_action = action
                
        policy[state] = best_action
    print("Policy extraction complete!")
    return policy

def get_initial_state_hull(Q_hulls, env, n_actions):
    """
    Get the value hull for the initial state by taking the convex hull
    of all Q-values across actions.
    """
    state = env.start_state
 
    all_vectors = []
    for action in range(n_actions):
        hull = Q_hulls[(*state, action)]
        if isinstance(hull, np.ndarray):
            all_vectors.extend(hull)
        else:
            all_vectors.extend(list(hull))
 
    all_vectors = np.array(all_vectors)
    
    # Get the convex hull of all Q-vectors
    if len(all_vectors) > 1:
        value_hull = get_hull(all_vectors)
    else:
        value_hull = all_vectors
 
    return value_hull
    
