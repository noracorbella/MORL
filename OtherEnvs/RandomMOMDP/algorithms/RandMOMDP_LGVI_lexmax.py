import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from RandMOMDP_LG_utils import lex_max
from RandMOMDP_utils import get_outcomes


def LG_VI_lexmax(env, theta=1.0, discount_factor=0.7, priority=None,
                 MNS_filename='randmomdp_policies/RandMOMDP_LGVI_lexmax_MNS.pkl',
                 v_table_file=None):
    """
    Lexicographic Value Iteration for the random MOMDP (stochastic, integer states).
    """

    n_states     = env.n_states
    n_actions    = env.n_actions
    n_objectives = env.n_rewards

    if priority is None:
        print("Please specify a priority.")

    V      = np.zeros((n_states, n_objectives))
    policy = np.full(n_states, -1, dtype=int)
    Q      = np.zeros((n_states, n_actions, n_objectives))

    os.makedirs(os.path.dirname(MNS_filename) if os.path.dirname(MNS_filename) else '.', exist_ok=True)

    if os.path.exists(MNS_filename):
        print("\nInitialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("\nInitialising model_next_state = {}")
        model_next_state = {}

    iteration    = 0
    total_states = len(env.valid_states)

    print(f"\nTotal states: {total_states}, Actions: {n_actions}")
    print(f"Priority order: {priority}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:

            for state in env.valid_states:

                v_old = V[state].copy()
                valid_actions = env.valid_actions(state)
                q_vectors = np.zeros((n_actions, n_objectives))

                for action in valid_actions:

                    if (state, action) not in model_next_state:
                        outcomes = get_outcomes(env, state, action)
                        model_next_state[(state, action)] = outcomes
                    else:
                        outcomes = model_next_state[(state, action)]

                    q_vector = np.zeros(n_objectives)

                    for next_state, reward_vect, done, prob in outcomes:
                        q_vector += prob * (reward_vect + discount_factor * V[next_state])

                    q_vectors[action] = q_vector

                Q[state] = q_vectors

                if len(valid_actions) > 0:
                    # Only lex-compare among available actions.
                    candidate = q_vectors[valid_actions]
                    best_action = lex_max(candidate, priority=priority)
                    V[state] = candidate[best_action]

                delta = max(delta, np.sum(np.abs(v_old - V[state])))

                pbar.update(1)

        print(f"Delta = {delta}, Theta = {theta}")

        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
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
        valid_actions = env.valid_actions(state)
        if len(valid_actions) > 0:
            candidate = Q[state][valid_actions]
            best_action = lex_max(candidate, priority=priority)
            policy[state] = valid_actions[best_action]

    return policy, Q
