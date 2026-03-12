import numpy as np
from tqdm import tqdm
from CH_operations import get_hull, translate_hull, sum_hulls, max_q_value
import os
import pickle

def convexhull_VI(env, theta=0.01, discount_factor=0.7, MNS_filename='stochastic_policies/CHVI_stochastic_MNS.pkl', q_hulls_file=None):
    """
    CONVERGENCE WITH V HULLS
    """
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_rewards = 3 # 3 rewards: r_car, r_p1, r_p2

    V = {}
    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                V[(c, p1, p2)] = np.array([np.zeros(n_rewards)])

    Q_hulls = {}
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                for a in range(n_actions):
                    Q_hulls[(c, p1, p2, a)] = np.array([np.zeros(n_rewards)])
    
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
    total_states = len(env.states_agent_left) * len(env.states_agent_right) ** 2 # n_cells * n_cells * n_cells

    print(f"Starting CHVI with {total_states} states and {n_actions} actions")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

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
                                            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2) # 0.25
                                            outcomes.append((next_state, reward_vect, done, prob))


                                model_next_state[(c, p1, p2, action)] = outcomes
                                
                            else:
                                outcomes = model_next_state[(c, p1, p2, action)]

                            # Bellman CH
                            # Q(s,a) = CH( U_{outcomes} [prob * (reward_vec + gamma * hull(s'))] )

                            outcome_hulls = []
                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    # Terminal state - Q(s,a) = reward vector scaled by prob
                                    outcome_hull = prob * np.array([reward_vect])
                                else:
                                    next_c, next_p1, next_p2 = next_state

                                    next_state_hull = V[(next_c, next_p1, next_p2)]

                                    outcome_hull = translate_hull(
                                        reward_vect,
                                        discount_factor,
                                        next_state_hull
                                    )

                                    if not isinstance(outcome_hull, np.ndarray):
                                        outcome_hull = np.array(outcome_hull)

                                    outcome_hull = prob * outcome_hull

                                outcome_hulls.append(outcome_hull)
                            
                            if len(outcome_hulls) == 1:
                                # Deterministic
                                new_hull = outcome_hulls[0]
                            else:
                                # Stochastic
                                combined_hull = outcome_hulls[0]
                                for outcome_hull in outcome_hulls[1:]:
                                    combined_hull = sum_hulls(combined_hull, outcome_hull)
                                new_hull = combined_hull

                            if not isinstance(new_hull, np.ndarray):
                                new_hull = np.array(new_hull)
                            
                            if len(new_hull) > 1:
                                new_hull = get_hull(new_hull)
                                                        
                            # Store new Q-hull
                            Q_hulls[(c, p1, p2, action)] = new_hull

                        all_q_vectors = []
                        for action in range(n_actions):
                            q_hull = Q_hulls[(c, p1, p2, action)]
                            if isinstance(q_hull, np.ndarray):
                                all_q_vectors.extend(q_hull)
                            else:
                                all_q_vectors.extend(list(q_hull))
                        
                        all_q_vectors = np.array(all_q_vectors)
                        
                        # Get convex hull of all Q-vectors to form V
                        if len(all_q_vectors) > 1:
                            new_V = get_hull(all_q_vectors)
                        else:
                            new_V = all_q_vectors
                        
                        V[state_tuple] = new_V

                        # Track hull statistics
                        new_hull_size = len(new_V)
                        total_hull_vertices += new_hull_size
                        num_hulls += 1

                        if v_old.shape == new_V.shape:
                            max_diff = np.max(np.abs(new_V - v_old))
                        else:
                            max_diff = float('inf')
                        
                        delta = max(delta, max_diff)

                        pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {delta}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state-action pair")
            

        # Convergence
        if delta < theta:
            print(f"Converged in {iteration} iterations")
            break
    
    with open(MNS_filename, 'wb') as f:
        pickle.dump(model_next_state, f)
    print(f"Model saved to {MNS_filename}")
    
    if q_hulls_file is not None:
        print(f"Saving Q_hulls to {q_hulls_file}...")
        with open(q_hulls_file, 'wb') as f:
            pickle.dump(Q_hulls, f)
    
    return Q_hulls


def extract_policy_for_weights(Q_hulls, weights, env, n_actions):
    """
    Extract the optimal policy for a specific weight vector.
    """
    n_cells = env.map_num_cells
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
    
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    print(f"\nExtracting policy for weights: {weights}")
    print(f"[w_car={weights[0]:.2f}, w_ped1={weights[1]:.2f}, w_ped2={weights[2]:.2f}]")
    
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                best_value = -np.inf
                best_action = 0
                
                # For each action, find the best Q-value for this weight vector
                for action in range(n_actions):
                    hull = Q_hulls[(c, p1, p2, action)]
                    
                    if not isinstance(hull, np.ndarray):
                        hull = np.array(hull)

                    # Extract max Q-value for this weight vector
                    # max_{q in hull} w · q
                    q_value = max_q_value(weights, hull)
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                policy[c, p1, p2] = best_action
    print("Policy extraction complete!")
    return policy

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
