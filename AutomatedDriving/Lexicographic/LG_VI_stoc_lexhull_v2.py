import numpy as np
from tqdm import tqdm
from LG_utils import lex_hull_corrected, generate_all_priority_orders, lex_max
import os
import pickle

def LG_VI_lexhull(env, theta=1.0, discount_factor=0.7, MNS_filename='policies/LG_VI_lexhull_MNS.pkl', v_hulls_file=None, q_hulls_file=None):    
    """
    Lexicographic Value Iteration Algorithm using hull-based convergence.
    
    Maintains the lexicographic hull using proper convex hull operations
    in the Bellman update.
    Args:
        env: the environment
        theta: convergence threshold
        discount_factor: discount factor
        MNS_filename: path to pickle file for caching model transitions
        v_hulls_file: optional filename to save the V hulls
        q_hulls_file: optional filename to save the Q hulls
    
    Returns:
        policies: dict of policies for all priority orders
        Q_hulls: Q hulls for each state-action pair
    """
    
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    # V stores hulls (sets of vectors) for each state
    V = {}
    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                V[(c, p1, p2)] = np.zeros((1, n_objectives))

    Q_hulls = {}  # Store hull for each state-action pair

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    if os.path.exists(MNS_filename):
        print("Initialising model_next_state = pickle.load(f)")
        with open(MNS_filename, 'rb') as f:
            model_next_state = pickle.load(f)
    else:
        print("Initialising model_next_state = {}")
        model_next_state = {}

    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2

    print(f"Starting Lexicographic Hull Value Iteration with lexhull")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")

    while True:
        iteration += 1
        print(f"\nIteration {iteration}")
        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            hull_sizes = []
            for c in env.states_agent_left:
                for p1 in env.states_agent_right:
                    for p2 in env.states_agent_right:

                        hull_sizes.append(V[(c, p1, p2)].shape[0])

                        state = np.array([c, p1, p2])
                        state_translated = env.translate_state(state)
                        state_tuple = (c, p1, p2)

                        old_hull = V[state_tuple].copy()

                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        # For each action, compute the Q-hull
                        action_hulls = []
                        
                        for action in range(n_actions):
                            if (c, p1, p2, action) not in model_next_state:
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

                            q_vectors_for_action = []
                            
                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    # Terminal state: Q = reward
                                    q_vectors_for_action.append(reward_vect)
                                else:
                                    # Non-terminal: Q = reward + gamma * V_hull
                                    next_hull = V[(next_state[0], next_state[1], next_state[2])]
                                    
                                    # For each vector in the next state's hull, compute Q
                                    for v_vector in next_hull:
                                        q_vec = reward_vect + discount_factor * v_vector
                                        q_vectors_for_action.append(q_vec)
                            
                            action_hull = np.array(q_vectors_for_action)
                            action_hulls.append(action_hull)
                            Q_hulls[state_tuple + (action,)] = action_hull

                        all_q_vectors = np.vstack(action_hulls)
                        
                        _, optimal_actions_in_combined = lex_hull_corrected(all_q_vectors, n_objectives=n_objectives)
                        new_hull = all_q_vectors[list(optimal_actions_in_combined)]

                        new_hull = new_hull[np.lexsort(new_hull.T[::-1])]
                        
                        V[state_tuple] = new_hull

                        # Convergence check
                        if old_hull.shape == new_hull.shape:
                            max_diff = np.max(np.abs(new_hull - old_hull))
                        else:
                            max_diff = float('inf')
                        
                        delta = max(delta, max_diff)

                        pbar.update(1)
            
            print(f"\nHull size numbers:")
            print(f"  Min hull size: {np.min(hull_sizes)}")
            print(f"  Max hull size: {np.max(hull_sizes)}")
            print(f"  Mean hull size: {np.mean(hull_sizes):.2f}")
            print(f"  States with hull size 1: {np.sum(np.array(hull_sizes) == 1)}/{len(hull_sizes)}")

        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        if delta < theta:
            print(f"\nConverged in {iteration} iterations")
            break
    
    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")

    if v_hulls_file is not None:
        with open(v_hulls_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"Model saved to {v_hulls_file}")

    if q_hulls_file is not None:
        with open(q_hulls_file, 'wb') as f:
            pickle.dump(Q_hulls, f)
        print(f"Model saved to {q_hulls_file}")

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