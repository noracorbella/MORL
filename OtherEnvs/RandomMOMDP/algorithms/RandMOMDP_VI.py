import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from RandMOMDP_utils import get_outcomes


def value_iteration(env, theta=1.0, discount_factor=0.7, max_iterations=10000, MNS_filename='randmomdp_policies/RandMOMDP_VI_MNS.pkl', v_table_file=None):
    """
    Value Iteration (Sutton and Barto, Section 4.4) adapted for the random MOMDP (1998).

    States are integers 0..S-1 and actions are masked per state.

    :param env: RandMOMDPEnvironment
    :param theta: convergence parameter
    :param discount_factor: discount factor of the MDP
    :param MNS_filename: path for caching the transition model (pickle)
    :param v_table_file: optional path to save the final V table (pickle)
    :return: policy (n_states,) and Q-table (n_states, n_actions)

    """

    # Initialise value function and policy
    n_states  = env.n_states
    n_actions = env.n_actions
    n_rewards = env.n_rewards

    V      = np.zeros(n_states)              # V table (scalar)
    policy = np.full(n_states, -1, dtype=int)   # For each state, which action to take?
    Q      = np.full((n_states, n_actions), -np.inf)  # For each state-action pair, expected total reward

    V_vec  = np.zeros((n_states, n_rewards))              # V table (vector)
    Q_vec  = np.zeros((n_states, n_actions, n_rewards))   # vector Q table


    weight_vect = np.array(env.weights, dtype=np.float64)
    weight_vect = weight_vect / np.linalg.norm(weight_vect, ord=1)

    os.makedirs(os.path.dirname(MNS_filename) if os.path.dirname(MNS_filename) else '.', exist_ok=True)

    if os.path.exists(MNS_filename):
        print("Initialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("Initialising model_next_state = {}")
        model_next_state = {}

    # iteration    = 0
    total_states = len(env.valid_states)

    print(f"Starting Value Iteration with {total_states} states and {n_actions} actions")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

    for iteration in range(max_iterations):
        print(f"\n Iteration {iteration}")

        delta = 0

        V_prev     = V.copy()
        V_vec_prev = V_vec.copy()

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:

            # Iterate through every possible state
            for state in env.valid_states:

                valid_actions = env.valid_actions(state)
                q_values = np.full(n_actions, -np.inf)
                q_vectors = np.zeros((n_actions, n_rewards))

                for action in valid_actions:

                    if (state, action) not in model_next_state:
                        outcomes = get_outcomes(env, state, action)
                        model_next_state[(state, action)] = outcomes
                    else:
                        outcomes = model_next_state[(state, action)]

                    q_value = 0.0
                    q_vector = np.zeros(n_rewards)

                    for next_state, reward_vect, done, prob in outcomes:
                        reward_vect = np.asarray(reward_vect, dtype=float)
                        reward_scalar = np.dot(reward_vect, weight_vect)

                        q_value  += prob * (reward_scalar + discount_factor * V_prev[next_state])
                        q_vector += prob * (reward_vect   + discount_factor * V_vec_prev[next_state])

                    q_values[action] = q_value
                    q_vectors[action] = q_vector

                # Store Q-values and update V (scalar + vector)
                Q[state]     = q_values
                Q_vec[state] = q_vectors

                if len(valid_actions) > 0:
                    best_action  = int(np.argmax(q_values))
                    V[state]     = q_values[best_action]
                    V_vec[state] = q_vectors[best_action]

                delta = max(delta, np.abs(V_prev[state] - V[state]))

                pbar.update(1)

        print(f"Delta = {delta}, Theta = {theta}")

        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
            print(f"Converged in {iteration} iterations")
            break

    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")

    if v_table_file is not None:
        os.makedirs(os.path.dirname(v_table_file) if os.path.dirname(v_table_file) else '.', exist_ok=True)
        with open(v_table_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"V table (scalar) saved to {v_table_file}")

        v_vec_file = v_table_file.replace('.pkl', '_vec.pkl')
        with open(v_vec_file, 'wb') as f:
            pickle.dump(V_vec, f)
        print(f"V table (vector) saved to {v_vec_file}")

    print("\nExtracting policy...")
    for state in env.valid_states:
        if env.valid_actions(state):
            policy[state] = int(np.argmax(Q[state]))

    return policy, Q
