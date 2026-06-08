import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from FT_utils import get_outcomes


def value_iteration(env, theta=1e-4, discount_factor=0.99, MNS_filename='ft_policies/FT_VI_MNS.pkl', v_table_file=None):    

    n_actions = env.n_actions  # 2
    tree_depth = env.tree_depth
    max_nodes = 2 ** tree_depth # max nodes at any depth level

    V      = np.zeros([tree_depth + 1, max_nodes])              # V table
    policy = np.zeros([tree_depth + 1, max_nodes], dtype=int)   # For each state, which action to take?
    Q      = np.zeros([tree_depth + 1, max_nodes, n_actions])   # For each state-action pair, expected total reward

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
    print(f"Starting Value Iteration with {total_states} non-terminal states and {n_actions} actions")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            
            
            # Iterate through every possible state
            for state in env.non_terminal_states:
                depth, node = state

                v_old    = V[state].copy()
                q_values = np.zeros(n_actions)

                for action in range(n_actions):

                    if (*state, action) not in model_next_state:
                        outcomes = get_outcomes(env, state, action)
                        model_next_state[(*state, action)] = outcomes
                    else:
                        outcomes = model_next_state[(*state, action)]

                    q_value = 0.0

                    for next_state, reward_vect, done, prob in outcomes:
                        reward_scalar = np.dot(reward_vect, weight_vect)

                        if done:
                            q_value += prob * reward_scalar
                        else:
                            next_value = V[next_state]
                            q_value += prob * (reward_scalar + discount_factor * next_value)

                        
                    q_values[action] = q_value

                # Store Q-values for this state
                Q[state]  = q_values

                # Update value function: V(s) = max_a Q(s,a)
                V[state]  = np.max(q_values)

                # Update delta - maximum change in value function
                delta     = max(delta, np.abs(v_old - V[state]))

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
        os.makedirs(os.path.dirname(v_table_file) if os.path.dirname(v_table_file) else '.', exist_ok=True)
        with open(v_table_file, 'wb') as f:
            pickle.dump(V, f)

    # Extract policy: for each state, choose action with best Q-value
    print("\nExtracting policy...")
    for state in env.non_terminal_states:
        policy[state] = np.argmax(Q[state])
 
    return policy, Q