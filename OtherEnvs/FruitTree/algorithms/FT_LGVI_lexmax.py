import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from FT_LG_utils import lex_max
from FT_utils import get_outcomes

def LG_VI_lexmax(env, theta=1e-4, discount_factor=0.99, priority=[0, 1, 2, 3, 4, 5], MNS_filename='ft_policies/FT_LGVI_lexmax_MNS.pkl', v_table_file=None):

    n_actions = env.n_actions    # 2
    n_objectives = env.n_rewards # 6
    tree_depth = env.tree_depth
    max_nodes = 2 ** tree_depth

    V      = np.zeros([tree_depth + 1, max_nodes, n_objectives])              # V table
    policy = np.zeros([tree_depth + 1, max_nodes], dtype=int)   # For each state, which action to take?
    Q      = np.zeros([tree_depth + 1, max_nodes, n_actions, n_objectives])   # For each state-action pair, expected total reward

    os.makedirs(os.path.dirname(MNS_filename) if  os.path.dirname(MNS_filename) else '.', exist_ok=True)

    if os.path.exists(MNS_filename):
        print("\nInitialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("\nInitialising model_next_state = {}")
        model_next_state = {}


    iteration = 0
    total_states = len(env.non_terminal_states)

    print(f"\nTotal states: {total_states}, Actions: {n_actions}")
    print(f"Priority order: {priority}")

    
    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            # Iterate through every possible state
            for state in env.non_terminal_states:
                depth, node = state

                v_old = V[state].copy()
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
                            q_vector += prob * (reward_vect + discount_factor * next_value)
                        
                    q_vectors[action] = q_vector

                # Store Q-values for this state
                Q[state] = q_vectors

                best_action = lex_max(q_vectors, priority=priority)
                V[state] = q_vectors[best_action]

                # Update delta - maximum change in value function
                delta = max(delta, np.sum(np.abs(v_old - V[state])))

                pbar.update(1)


        print(f"Delta = {delta}, Theta = {theta}")

        # Check convergence
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

    # Extract policy: for each state, choose action with best Q-value
    print("\nExtracting policy...")
    for (state) in env.non_terminal_states:
        policy[state] = lex_max(Q[state], priority=priority)
 
    return policy, Q