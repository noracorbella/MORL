import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from DST_utils import step_from

# The Markov property in MDPs means that the future states and rewards are independent of past states and actions,

def value_iteration(env, theta=1.0, discount_factor=0.7, MNS_filename='dst_policies/DST_VI_MNS.pkl', v_table_file=None):
    """
    Value Iteration Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 4.4,
    (1998).

    It has been adapted for the Deep Sea Treasure MOMDP (mo-gymnasium).
    DST is fully deterministic
 
    :param env: DSTEnvironment
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the MDP
    :param MNS_filename: path for caching the transition model (pickle)
    :param v_table_file: optional path to save the final V table (pickle)
    :return: policy and Q-table
    """

    # Initialise value function and policy
    n_rows    = env.n_rows     # 11
    n_cols    = env.n_cols     # 11
    n_actions = env.n_actions  # 4
    n_rewards = 2

    V      = np.zeros([n_rows, n_cols])              # V table
    policy = np.zeros([n_rows, n_cols], dtype=int)   # For each state, which action to take?
    Q      = np.zeros([n_rows, n_cols, n_actions])   # For each state-action pair, expected total reward

    V_vec = np.zeros([n_rows, n_cols, n_rewards])
    Q_vec = np.zeros([n_rows, n_cols, n_actions, n_rewards])  

    weight_vect = np.array(env.weights)
 
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
    print(f"Valid states: {len(env.valid_states)}.")
    print(f"Non terminal states: {len(env.non_terminal_states)}")
    print(f"Starting Value Iteration with {len(env.non_terminal_states)} states and {n_actions} actions")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            
            
            # Iterate through every possible state
            for (row, col) in env.non_terminal_states:

                v_old = V[row, col].copy()

                q_values = np.zeros(n_actions)
                q_vectors = np.zeros((n_actions, n_rewards))

                for action in range(n_actions):

                    if (row, col, action) not in model_next_state:
                        next_state, reward_vect, done = step_from(env.gym_env, row, col, action)

                        prob = 1.0 # deterministic environment

                        outcomes = [(next_state, reward_vect, done, prob)]

                        model_next_state[(row, col, action)] = outcomes
                    
                    else: 
                        outcomes = model_next_state[(row, col, action)]

                    q_value = 0.0
                    q_vector = np.zeros(n_rewards)

                    for next_state, reward_vect, done, prob in outcomes:
                        reward_vect = np.asarray(reward_vect, dtype=float)
                        reward_scalar = np.dot(reward_vect, weight_vect)

                        if done:
                            q_vector += prob * reward_vect
                            q_value += prob * reward_scalar
                        else:
                            next_row, next_col = next_state
                            next_value_vect = V_vec[next_state]
                            next_value = V[next_row, next_col]
                            q_vector += prob * (reward_vect + discount_factor * next_value_vect)
                            q_value += prob * (reward_scalar + discount_factor * next_value)

                        
                    if not isinstance(q_value, (int, float, np.floating)):
                        print(f"ERROR DEBUG:")
                        print(f"  State: row={row}, col={col}, action={action}")
                        print(f"  q_value type: {type(q_value)}")
                        print(f"  q_value: {q_value}")
                        print(f"  outcomes: {outcomes}")
                    
                    q_values[action] = q_value
                    q_vectors[action] = q_vector

                best_action = int(np.argmax(q_values))
                # Store Q-values for this state
                Q[row, col] = q_values
                Q_vec[row, col] = q_vectors

                # Update value function: V(s) = max_a Q(s,a)
                V[row, col] = np.max(q_values)
                V_vec[row, col] = q_vectors[best_action]

                # Update delta - maximum change in value function
                delta = max(delta, np.abs(v_old - V[row, col]))

                pbar.update(1)

        
        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        # Check convergence
        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
            print(f"Converged in {iteration} iterations")
            break

    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")

    if v_table_file is not None:
        print(f"Saving V table to {v_table_file}...")
        os.makedirs(os.path.dirname(v_table_file) if os.path.dirname(v_table_file) else '.', exist_ok=True)
        with open(v_table_file, 'wb') as f:
            pickle.dump(V, f)
    
        v_vec_file = v_table_file.replace('.pkl', '_vec.pkl')
        with open(v_vec_file, 'wb') as f:
            pickle.dump(V_vec, f)
        print(f"V table (vector) saved to {v_vec_file}")

    # Extract policy: for each state, choose action with best Q-value
    print("\nExtracting policy...")
    for row in range(n_rows):
        for col in range(n_cols):
            policy[row, col] = np.argmax(Q[row, col])

    return policy, Q