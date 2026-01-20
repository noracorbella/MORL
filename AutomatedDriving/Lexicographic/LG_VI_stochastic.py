import numpy as np
from tqdm import tqdm


# LEX_HULL(q_vectors, tol)
# extreure objectius i fer permutacions possibles ic calucalr lexmax associat
def lex_max(q_vectors, priority='car',tol=1e-9):
    """
    Lexicographic maximisation implementing Eq. (5) from Vamplew et al. (2021).
    
    Args:
        q_vectors: array of shape (n_actions, n_objectives)
        priority: 'car' or 'pedestrian'
        tol
    
    Returns:
        Index of the lexicographically best action
    """

    n_actions = q_vectors.shape[0]

    if priority == 'car':
        objective_order = [0,1,2]
    else:
        objective_order = [1,2,0]

    best_actions = list(range(n_actions)) # initialisation, all actions are candidates
    
    for obj_idx in objective_order:
        if len(best_actions) == 1:
            # only one action left
            break

        # get q values of this objective for all possible actions
        obj_values = [q_vectors[action, obj_idx] for action in best_actions]
        # keep actions that maximise this objective
        max_val = np.max(obj_values)


        new_best_actions = []
        for i, action in enumerate(best_actions):
            if abs(obj_values[i] - max_val) < 1e-9:
                new_best_actions.append(action)
        
        best_actions = new_best_actions

        if len(best_actions) == 0:
            print(f"WARNING: No actions left! Returning 0")
            return 0
    # return the first reamaining action I VECTOR ASSOCIAT
    return best_actions[0]

def LG_VI(env, theta=1.0, discount_factor=0.7, priority = 'car'):
    """
    Value Iteration Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 4.4,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic environment, and also
    adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the (MO)MPD, can be set at discretion
    :return: policy and Q-table
    """

    # Initialize value function and policy
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])  # V table: each entry represents how good is it to be in this state
    model_next_state = {}  # dict to store multiple possible next states


    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)  # For each state, which action should we take?
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])  # For each state-action pair, what's the expected total reward?

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]


    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2 #n_cells * n_cells * n_cells

    print(f"Starting Value Iteration with {total_states} states and {n_actions} actions")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

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


                        # Check if either pedestrian is in stochastic state
                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        # For this state, try every action
                        for action in range(n_actions):

                            # Take action and observe next state and reward
                            if iteration == 1:
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
                        delta = max(delta, np.max(np.abs(v_old - V[c, p1, p2])))

                        pbar.update(1)

        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        # Check convergence
        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
            print(f"Converged in {iteration} iterations")
            break

    # Extract policy: for each state, choose action with best Q-value
    print("\nExtracting policy...")
    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                policy[c, p1, p2] = lex_max(Q[c, p1, p2], priority=priority)  # ← Ha de ser lex_max!

    return policy, Q
