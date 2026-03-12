import numpy as np
from tqdm import tqdm
from CH_operations import get_hull, translate_hull, sum_hulls, max_q_value

def convexhull_VI(env, theta=0.01, discount_factor=0.7):

    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_rewards = 3 # 3 rewards: r_car, r_p1, r_p2

    saved_weights = env.weights
    env.weights = None

    # Q_hulls[c, p1, p2, action] = list of vectors (each vector has 3 components)
    Q_hulls = {}
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                for a in range(n_actions):
                    Q_hulls[(c, p1, p2, a)] = [np.zeros(n_rewards)]
    
    model_next_state = np.zeros([n_cells, n_cells, n_cells, n_actions, 3], dtype=int)
    model_next_reward = np.zeros([n_cells, n_cells, n_cells, n_actions, n_rewards])
    model_next_done = np.zeros([n_cells, n_cells, n_cells, n_actions], dtype=bool)
    
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

                        for action in range(n_actions):
                            # Store old hull for convergence check
                            old_hull = Q_hulls[(c, p1, p2, action)]
                            old_hull_array = np.array(old_hull) if len(old_hull) > 0 else np.zeros((1, n_rewards))
                            # old_hull_size = len(old_hull)

                            if iteration == 1:
                                env.reset(state_translated[0], state_translated[1], state_translated[2])
                                next_state, reward, done_array = env.step([action])
                                done = done_array[0]  # True if terminal state, False otherwise

                                model_next_state[c, p1, p2, action] = next_state
                                model_next_reward[c, p1, p2, action] = reward
                                model_next_done[c, p1, p2, action] = done
                            else:
                                next_state = model_next_state[c, p1, p2, action]
                                reward = model_next_reward[c, p1, p2, action]
                                done = model_next_done[c, p1, p2, action]

                            # Bellman CH
                            if done:
                                # Terminal state - Q(s,a) = reward vector only
                                new_hull = [reward.copy()]
                            else:
                                # Q(s,a) = r(s,a) + γ * hull(∪_{a'} Q(s', a'))
                                # hull[ ∪_{a'} Q_hull(s', a') ]
                                next_c, next_p1, next_p2 = next_state

                                all_next_q_vectors = []
                                for next_action in range(n_actions):
                                    next_hull = Q_hulls[(next_c, next_p1, next_p2, next_action)]

                                    for q_vec in next_hull:
                                        if isinstance(q_vec, np.ndarray):
                                            all_next_q_vectors.append(q_vec)
                                        else:
                                            all_next_q_vectors.append(np.array(q_vec))
                                
                                # Complex hull of union
                                if len(all_next_q_vectors) > 0:
                                    all_next_q_vectors = np.array(all_next_q_vectors)
                                    # CH vertices
                                    next_state_hull = get_hull(all_next_q_vectors)
                                else:
                                    next_state_hull = np.array([np.zeros(n_rewards)])

                                # Q(s,a) = r + γ * hull
                                # translate_hull does: r + γ * hull
                                new_hull = translate_hull(
                                    reward, 
                                    discount_factor, 
                                    next_state_hull
                                )

                                # get CH 
                                if len(new_hull) > 1:
                                    new_hull = get_hull(new_hull)

                                # convert each row to list element
                                if isinstance(new_hull, np.ndarray):
                                    new_hull = [vec for vec in new_hull]

                            # Store new hull
                            Q_hulls[(c,p1,p2,action)] = new_hull

                            new_hull_size = len(new_hull)
                            total_hull_vertices += new_hull_size
                            num_hulls += 1

                            new_hull_array = np.array(new_hull) if len(new_hull) > 0 else np.zeros((1, n_rewards))

                            if old_hull_array.shape == new_hull_array.shape:
                                max_diff = np.max(np.abs(new_hull_array - old_hull_array))
                            else:
                                # Different number of vertices - mark as changed
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
    
    env.weights = saved_weights

    # Extract policy for the weight vector
    weights = np.array(env.weights, dtype=float)
    weights = weights / np.sum(weights)  # Normalize
    
    print(f"\nExtracting policy for environment weights: {weights}")
    print(f"[w_car={weights[0]:.2f}, w_ped1={weights[1]:.2f}, w_ped2={weights[2]:.2f}]")
    
    policy = extract_policy_for_weights(Q_hulls, weights, env, n_actions)
    
    return policy, Q_hulls

def extract_policy_for_weights(Q_hulls, weights, env, n_actions):
    """
    Extract the optimal policy for a specific weight vector.
    """
    n_cells = env.map_num_cells
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
    
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                best_value = -np.inf
                best_action = 0
                
                # For each action, find the best Q-value for this weight vector
                for action in range(n_actions):
                    hull = np.array(Q_hulls[(c, p1, p2, action)])
                    
                    # Extract max Q-value for this weight vector
                    # max_{q ∈ hull} w · q
                    q_value = max_q_value(weights, hull)
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                policy[c, p1, p2] = best_action
    
    return policy
