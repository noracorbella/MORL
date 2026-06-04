import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from BB_LG_utils import lex_max
from BB_utils import get_outcomes


def LG_VI_lexmax(env, theta=1.0, discount_factor=0.95, priority=[0, 1, 2],
                 MNS_filename='bb_policies/BB_LGVI_lexmax_MNS.pkl',
                 v_table_file=None):
    """
    Lexicographic Value Iteration for Breakable Bottles (stochastic MOMDP).
    """

    n_actions    = env.n_actions  # 3
    n_objectives = env.n_rewards  # 3
    size         = env.size       # 5

    # State is 6-tuple (loc, carrying, delivered, d0, d1, d2)
    # V/Q/policy shapes are [5,3,3,2,2,2,3] and [5,3,3,2,2,2,3,3]
    V      = np.zeros([size, 3, 3, 2, 2, 2, n_objectives])
    policy = np.zeros([size, 3, 3, 2, 2, 2], dtype=int)
    Q      = np.zeros([size, 3, 3, 2, 2, 2, n_actions, n_objectives])

    os.makedirs(os.path.dirname(MNS_filename) if os.path.dirname(MNS_filename) else '.', exist_ok=True)

    if os.path.exists(MNS_filename):
        print("\nInitialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("\nInitialising model_next_state = {}")
        model_next_state = {}

    iteration    = 0
    total_states = len(env.non_terminal_states)

    print(f"\nTotal states: {total_states}, Actions: {n_actions}")
    print(f"Priority order: {priority}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:

            for state in env.non_terminal_states:

                v_old     = V[state].copy()
                q_vectors = np.zeros((n_actions, n_objectives))

                for action in range(n_actions):

                    if (*state, action) not in model_next_state:
                        outcomes = get_outcomes(env, state, action)
                        model_next_state[(*state, action)] = outcomes
                    else:
                        outcomes = model_next_state[(*state, action)]

                    q_vector = np.zeros(n_objectives)

                    for next_state, reward_vect, done, prob in outcomes:
                        if done:
                            q_vector += prob * reward_vect
                        else:
                            next_value = V[next_state]
                            q_vector  += prob * (reward_vect + discount_factor * next_value)

                    q_vectors[action] = q_vector

                Q[state] = q_vectors

                best_action = lex_max(q_vectors, priority=priority)
                V[state]    = q_vectors[best_action]

                # delta = max(delta, np.sum(np.abs(v_old - V[state])))
                delta = max(delta, np.max(np.abs(v_old - V[state])))

                pbar.update(1)

        print(f"Delta = {delta}, Theta = {theta}")

        if delta < theta:
            print(f"Converged in {iteration} iterations")
            break

    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"model_next_state saved to {MNS_filename}")

    if v_table_file is not None:
        os.makedirs(os.path.dirname(v_table_file) if os.path.dirname(v_table_file) else '.', exist_ok=True)
        with open(v_table_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"V table saved to {v_table_file}")

    print("\nExtracting policy...")
    for state in env.valid_states:
        policy[state] = lex_max(Q[state], priority=priority)

    return policy, Q