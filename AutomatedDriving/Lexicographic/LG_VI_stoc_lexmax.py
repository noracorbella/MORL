import numpy as np
from tqdm import tqdm
from Lexicographic.LG_utils import lex_max
import pickle
import os


def LG_VI_lexmax(env, theta=1.0, discount_factor=0.7, priority=[0,1,2], MNS_filename='policies/LG_VI_lexmax_MNS.pkl', 
                 v_table_file=None):
    """
    Lexicographic Value Iteration Algorithm for a single priority order.

    This is the standard lexicographic VI algorithm that computes the optimal policy
    for one specific lexicographic ordering of objectives.

    Args:
        env: the environment encoding the MOMDP
        theta: convergence parameter, the smaller it is the more precise the algorithm
        discount_factor: discount factor of the MOMDP, can be set at discretion
        priority: lexicographic priority order (e.g., [0,1,2] for car first)
    
    Returns:
        policy: optimal policy for the given priority order
        
    """
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = len(priority)

    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])  # V table: each entry represents how good is it to be in this state
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)  # For each state, which action should we take?
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])  # For each state-action pair, what's the expected total reward?

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    if os.path.exists(MNS_filename):
        print("\nInitialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("\nInitialising model_next_state = {}")
        model_next_state = {}


    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2 #n_cells * n_cells * n_cells

    print(f"\nTotal states: {total_states}, Actions: {n_actions}")
    print(f"Priority order: {priority}")

    
    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            # Iterate through every possible state
            for c in env.states_agent_left:
                for p1 in env.states_agent_right:
                    for p2 in env.states_agent_right:
                        state = np.array([c, p1, p2])
                        state_translated = env.translate_state(state)

                        v_old = V[c, p1, p2].copy()

                        q_vectors = np.zeros((n_actions, n_objectives))

                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        for action in range(n_actions):

                            if (c, p1, p2, action) not in model_next_state:
                                outcomes = [] # (next_state, reward, done, probability)

                                if not p1_is_stochastic and not p2_is_stochastic:
                                    # Deterministic case
                                    env.reset(state_translated[0], state_translated[1], state_translated[2])
                                    next_state, reward_vect, done_array = env.step([action])
                                    done = done_array[0]  
                                    prob = 1.0
                                    outcomes.append((next_state, reward_vect, done, prob))

                                elif p1_is_stochastic and not p2_is_stochastic:
                                    for p1_action in pedestrian_stochastic_actions:

                                        env.reset(state_translated[0], state_translated[1], state_translated[2])
                                        next_state, reward_vect, done_array = env.step([action, p1_action, 8000])
                                        done = done_array[0]
                                        prob = 1.0 / len(pedestrian_stochastic_actions)
                                        outcomes.append((next_state, reward_vect, done, prob))

                                elif not p1_is_stochastic and p2_is_stochastic:
                                    for p2_action in pedestrian_stochastic_actions:
                                        env.reset(state_translated[0], state_translated[1], state_translated[2])
                                        next_state, reward_vect, done_array = env.step([action, 8000, p2_action])
                                        done = done_array[0]
                                        prob = 1.0 / len(pedestrian_stochastic_actions)

                                        outcomes.append((next_state, reward_vect, done, prob))
                                else:
                                    # Both pedestrians are stochastic
                                    for p1_action in pedestrian_stochastic_actions:
                                        for p2_action in pedestrian_stochastic_actions:
                                            env.reset(state_translated[0], state_translated[1], state_translated[2])
                                            next_state, reward_vect, done_array = env.step([action, p1_action, p2_action])
                                            done = done_array[0]
                                            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2) # 0.25
                                            outcomes.append((next_state, reward_vect, done, prob))
                                
                                model_next_state[(c, p1, p2, action)] = outcomes
                                
                            else:
                                outcomes = model_next_state[(c, p1, p2, action)]


                            q_vector = np.zeros(n_objectives)

                            for next_state, reward_vect, done, prob in outcomes:

                                if done:
                                    q_vector += prob * reward_vect
                                else:
                                    next_value = V[next_state[0], next_state[1], next_state[2]]
                                    q_vector += prob * (reward_vect + discount_factor * next_value)
                                
                            q_vectors[action] = q_vector

                        # Store Q-values for this state
                        Q[c, p1, p2] = q_vectors

                        best_action = lex_max(q_vectors, priority=priority)
                        V[c, p1, p2] = q_vectors[best_action]

                        # Update delta - maximum change in value function
                        delta = max(delta, np.sum(np.abs(v_old - V[c, p1, p2])))

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
        with open(v_table_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"V table saved to {v_table_file}...")

    # Extract policy: for each state, choose action with best Q-value
    print("\nExtracting policy...")
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                policy[c, p1, p2] = lex_max(Q[c, p1, p2], priority=priority)

    return policy, Q