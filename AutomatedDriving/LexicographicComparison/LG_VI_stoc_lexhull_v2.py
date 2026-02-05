import numpy as np
from tqdm import tqdm
from LG_utils import lex_hull_corrected, generate_all_priority_orders, lex_max

def LG_VI_lexhull(env, theta=1.0, discount_factor=0.7):
    """
    Lexicographic Value Iteration Algorithm using hull-based convergence.
    
    Maintains the lexicographic hull using proper convex hull operations
    in the Bellman update.
    """
    
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    MAX_ITER = 10

    # V stores hulls (sets of vectors) for each state
    V = {}
    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                V[(c, p1, p2)] = np.zeros((1, n_objectives))

    model_next_state = {}
    Q_hulls = {}  # Store hull for each state-action pair

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2

    print(f"Starting Lexicographic Hull Value Iteration (Convex hull operations)")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")

    while True:
        iteration += 1
        print(f"\nIteration {iteration}")
        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            for c in env.states_agent_left:
                for p1 in env.states_agent_right:
                    for p2 in env.states_agent_right:
                        state = np.array([c, p1, p2])
                        state_translated = env.translate_state(state)
                        state_tuple = (c, p1, p2)

                        old_hull = V[state_tuple].copy()

                        # Check stochasticity
                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        # For each action, compute the Q-hull
                        action_hulls = []
                        
                        for action in range(n_actions):
                            if iteration == 1:
                                outcomes = []
                                if not p1_is_stochastic and not p2_is_stochastic:
                                    env.reset(state_translated[0], state_translated[1], state_translated[2])
                                    next_state, reward_vect, done_array = env.step([action])
                                    done = done_array[0]
                                    outcomes.append((next_state, reward_vect, done, 1.0))
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
                                    for p1_action in pedestrian_stochastic_actions:
                                        for p2_action in pedestrian_stochastic_actions:
                                            env.reset(state_translated[0], state_translated[1], state_translated[2])
                                            next_state, reward_vect, done_array = env.step([action, p1_action, p2_action])
                                            done = done_array[0]
                                            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2)
                                            outcomes.append((next_state, reward_vect, done, prob))
                                
                                model_next_state[state_tuple + (action,)] = outcomes
                            else:
                                outcomes = model_next_state[state_tuple + (action,)]

                            # Compute Q-vectors for this action across all stochastic outcomes
                            q_vectors_for_action = []
                            
                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    # Terminal state: Q = reward
                                    q_vectors_for_action.append(reward_vect)
                                else:
                                    # Non-terminal: Q = reward + γ * V_hull
                                    next_hull = V[(next_state[0], next_state[1], next_state[2])]
                                    
                                    # For each vector in the next state's hull, compute Q
                                    for v_vector in next_hull:
                                        q_vec = reward_vect + discount_factor * v_vector
                                        q_vectors_for_action.append(q_vec)
                            
                            # Convert to array and store
                            action_hull = np.array(q_vectors_for_action)
                            action_hulls.append(action_hull)
                            Q_hulls[state_tuple + (action,)] = action_hull

                        # Now compute the lexicographic hull across ALL actions
                        # Concatenate all action hulls
                        all_q_vectors = np.vstack(action_hulls)
                        
                        # Apply lexicographic hull to get only non-dominated vectors
                        _, optimal_actions_in_combined = lex_hull_corrected(all_q_vectors, n_objectives=n_objectives)
                        new_hull = all_q_vectors[list(optimal_actions_in_combined)]

                        # Sort hull for consistent comparison
                        new_hull = new_hull[np.lexsort(new_hull.T[::-1])]
                        
                        V[state_tuple] = new_hull

                        # Convergence check
                        if old_hull.shape == new_hull.shape:
                            max_diff = np.max(np.abs(new_hull - old_hull))
                        else:
                            max_diff = float('inf')
                        
                        delta = max(delta, max_diff)

                        pbar.update(1)

        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        if delta < theta:
            print(f"\nConverged in {iteration} iterations")
            break
        elif iteration >= MAX_ITER:
            print(f"\nReached maximum iterations ({MAX_ITER})")
            print(f"Final delta = {round(delta, 3)}")
            break


    # Extract policies
    print("\nExtracting policies for all lexicographic orders...")
    all_priority_orders = generate_all_priority_orders(n_objectives)
    policies = {}
    
    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
        
        for c in env.states_agent_left:
            for p1 in env.states_agent_right:
                for p2 in env.states_agent_right:
                    state_tuple = (c, p1, p2)
                    
                    # For each action, get its representative Q-vector
                    # (We need to pick ONE vector from each action's hull)
                    q_vectors = []
                    for action in range(n_actions):
                        action_hull = Q_hulls[state_tuple + (action,)]
                        # Use lexicographic max for THIS priority to select representative
                        best_idx = lex_max(action_hull, priority=list(priority_order))
                        q_vectors.append(action_hull[best_idx])
                    
                    q_vectors = np.array(q_vectors)
                    
                    # Now find best action for this priority
                    best_action = lex_max(q_vectors, priority=list(priority_order))
                    policy[c, p1, p2] = best_action
        
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")

    return policies, Q_hulls