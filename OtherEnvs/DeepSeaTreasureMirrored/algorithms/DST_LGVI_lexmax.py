import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from DST_LG_utils import lex_max
from DST_utils import step_from

def LG_VI_lexmax(env, theta=1.0, discount_factor=0.99, priority=[0,1], MNS_filename='dst_policies/DST_LGVI_lexmax_MNS.pkl', 
                 v_table_file=None):
    """
    Args:
        env: the environment encoding the MOMDP
        theta: convergence parameter, the smaller it is the more precise the algorithm
        discount_factor: discount factor of the MOMDP, can be set at discretion
        priority: lexicographic priority order (e.g., [0,1])
    
    Returns:
        policy: optimal policy for the given priority order
        
    """

    n_actions = env.n_actions    # 4
    n_objectives = env.n_rewards # 2

    V = np.zeros([env.n_rows, env.n_cols, n_objectives])            # V table: each entry represents how good is it to be in this state
    policy = np.zeros([env.n_rows, env.n_cols], dtype=int)          # For each state, which action should we take?
    Q = np.zeros([env.n_rows, env.n_cols, n_actions, n_objectives]) # For each state-action pair, what's the expected total reward?

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
            for (row, col) in env.non_terminal_states:

                v_old = V[row, col].copy()
                q_vectors = np.zeros((n_actions, n_objectives))

                for action in range(n_actions):

                    if (row, col, action) not in model_next_state:
                        next_state, reward_vect, done = step_from(env.gym_env, row, col, action)
                        prob = 1.0
                        outcomes = [(next_state, reward_vect, done, prob)]
 
                        model_next_state[(row, col, action)] = outcomes
 
                    else:
                        outcomes = model_next_state[(row, col, action)]


                    q_vector = np.zeros(n_objectives)

                    for next_state, reward_vect, done, prob in outcomes:

                        if done:
                            q_vector += prob * reward_vect
                        else:
                            next_row, next_col = next_state
                            next_value = V[next_row, next_col]
                            q_vector += prob * (reward_vect + discount_factor * next_value)
                        
                    q_vectors[action] = q_vector

                # Store Q-values for this state
                Q[row, col] = q_vectors

                best_action = lex_max(q_vectors, priority=priority)
                V[row, col] = q_vectors[best_action]

                # Update delta - maximum change in value function
                delta = max(delta, np.sum(np.abs(v_old - V[row, col])))

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
    for (row, col) in env.non_terminal_states:
        policy[row, col] = lex_max(Q[row, col], priority=priority)
 
    return policy, Q