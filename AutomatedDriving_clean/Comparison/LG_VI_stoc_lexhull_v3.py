import numpy as np
from tqdm import tqdm
from LG_utils import lex_hull_corrected, generate_all_priority_orders, lex_max
from CH_operations import get_hull, translate_hull, sum_hulls, max_q_value
import os
import pickle
import time

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

    Q_hulls = {}
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                for a in range(n_actions):
                    Q_hulls[(c, p1, p2, a)] = np.array([np.zeros(n_objectives)])

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

    t_start = time.time()
    t_model_done = None # set at end of iteration 1

    while True:
        iteration += 1
        print(f"\nIteration {iteration}")
        delta = 0
        total_hull_vertices = 0
        num_hulls = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            for c in env.states_agent_left:
                for p1 in env.states_agent_right:
                    for p2 in env.states_agent_right:
                        state = np.array([c, p1, p2])
                        state_translated = env.translate_state(state)
                        state_tuple = (c, p1, p2)

                        v_old = V[state_tuple].copy()

                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        for action in range(n_actions):
                            if (c, p1, p2, action) not in model_next_state:
                                outcomes = []

                                if not p1_is_stochastic and not p2_is_stochastic:
                                    env.reset(state_translated[0], state_translated[1], state_translated[2])
                                    next_state, reward, done_array = env.step([action])
                                    done = done_array[0]
                                    prob = 1.0
                                    outcomes.append((next_state, reward, done, prob))

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
                                            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2)
                                            outcomes.append((next_state, reward_vect, done, prob))

                                model_next_state[(c, p1, p2, action)] = outcomes
                            else:
                                outcomes = model_next_state[(c, p1, p2, action)]

                            outcome_hulls = []
                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    # terminal state 
                                    outcome_hull = prob * np.array([reward_vect])
                                else:
                                    next_c, next_p1, next_p2 = next_state
                                    next_state_hull = V[(next_c, next_p1, next_p2)]

                                    # translate_hull does: reward + gamma * hull
                                    outcome_hull = translate_hull(
                                        reward_vect,
                                        discount_factor,
                                        next_state_hull
                                    )

                                    if not isinstance(outcome_hull, np.ndarray):
                                        outcome_hull = np.array(outcome_hull)

                                    outcome_hull = prob * outcome_hull

                                outcome_hulls.append(outcome_hull)

                            # Combine outcomes using Minkowski sum 
                            if len(outcome_hulls) == 1:
                                # Deterministic 
                                new_q_hull = outcome_hulls[0]
                            else:
                                # Stochastic Minkowski sum of scaled hulls
                                combined_hull = outcome_hulls[0]
                                for outcome_hull in outcome_hulls[1:]:
                                    combined_hull = sum_hulls(combined_hull, outcome_hull)
                                new_q_hull = combined_hull

                            if not isinstance(new_q_hull, np.ndarray):
                                new_q_hull = np.array(new_q_hull)

                            if len(new_q_hull) > 1:
                                _, optimal_indices = lex_hull_corrected(new_q_hull, n_objectives=n_objectives)
                                new_q_hull = new_q_hull[list(optimal_indices)]

                            Q_hulls[(c, p1, p2, action)] = new_q_hull

                        all_q_vectors = []
                        for action in range(n_actions):
                            q_hull = Q_hulls[(c, p1, p2, action)]
                            if isinstance(q_hull, np.ndarray):
                                all_q_vectors.extend(q_hull)
                            else:
                                all_q_vectors.extend(list(q_hull))

                        all_q_vectors = np.array(all_q_vectors)

                        if len(all_q_vectors) > 1:
                            _, optimal_indices = lex_hull_corrected(all_q_vectors, n_objectives=n_objectives)
                            new_V = all_q_vectors[list(optimal_indices)]
                        else:
                            new_V = all_q_vectors

                        # Sort for consistent comparison
                        new_V = new_V[np.lexsort(new_V.T[::-1])]

                        V[state_tuple] = new_V

                        # Track 
                        new_hull_size = len(new_V)
                        total_hull_vertices += new_hull_size
                        num_hulls += 1

                        # Convergence 
                        if v_old.shape == new_V.shape:
                            max_diff = np.max(np.abs(new_V - v_old))
                        else:
                            max_diff = float('inf')

                        delta = max(delta, max_diff)

                        pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {round(delta, 3)}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state")

        if iteration == 1:
            t_model_done = time.time()

        if delta < theta:
            print(f"\nConverged in {iteration} iterations")
            t_vi_done = time.time()
            break

    # Save model
    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")

    if v_hulls_file is not None:
        with open(v_hulls_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"V hulls saved to {v_hulls_file}")

    if q_hulls_file is not None:
        with open(q_hulls_file, 'wb') as f:
            pickle.dump(Q_hulls, f)
        print(f"Q hulls saved to {q_hulls_file}")

    print("\nExtracting policies for all lexicographic orders...")
    all_priority_orders = generate_all_priority_orders(n_objectives)
    policies = {}
    extraction_times = {}

    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
        t_policy_start = time.time()

        for c in env.states_agent_left:
            for p1 in env.states_agent_right:
                for p2 in env.states_agent_right:
                    state_tuple = (c, p1, p2)

                    # For each action, get its best Q-vector for this priority
                    q_vectors = []
                    for action in range(n_actions):
                        action_hull = Q_hulls[state_tuple + (action,)]
                        best_idx = lex_max(action_hull, priority=list(priority_order))
                        q_vectors.append(action_hull[best_idx])

                    q_vectors = np.array(q_vectors)

                    # Find best action for this priority
                    best_action = lex_max(q_vectors, priority=list(priority_order))
                    policy[c, p1, p2] = best_action
        
        extraction_times[priority_tuple] = time.time() - t_policy_start
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")
    
    t_extract_done = time.time()

    timing = {
        "model_build":       t_model_done - t_start,
        "remaining_vi":      t_vi_done    - t_model_done,
        "policy_extraction": t_extract_done - t_vi_done,
        "total":             t_extract_done - t_start,
        "extraction_times":  extraction_times,
    }

    return policies, Q_hulls, timing


def get_initial_state_hull(Q_hulls, initial_state_indices, n_actions):
    """
    Get the value hull for the initial state by taking the convex hull
    of all Q-values across actions.
    
    :param Q_hulls: The Q_hulls dictionary from convexhull_VI
    :param initial_state_indices: [c, p1, p2] indices for the initial state
    :param n_actions: Number of actions
    :return: Array of vectors representing the value hull for the initial state
    """
    c, p1, p2 = initial_state_indices
    
    all_vectors = []
    for action in range(n_actions):
        hull = Q_hulls[(c, p1, p2, action)]
        if isinstance(hull, np.ndarray):
            all_vectors.extend(hull)
        else:
            all_vectors.extend(list(hull))
    
    all_vectors = np.array(all_vectors)
    
    # Get the convex hull of all Q-vectors
    if len(all_vectors) > 1:
        value_hull = get_hull(all_vectors)
    else:
        value_hull = all_vectors
    
    return value_hull
