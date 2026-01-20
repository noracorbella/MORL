import numpy as np
from tqdm import tqdm
from CH_operations import get_hull, translate_hull, sum_hulls, max_q_value

def LG_CH_VI(env, theta=0.01, discount_factor=0.7):

    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_rewards = 3 # 3 rewards: r_car, r_p1, r_p2

    model_next_state = {}  # dict to store multiple possible next states

    # weight_vect = np.array(env.weights) 

    saved_weights = env.weights if env.weights is not None else [1.0, 1.0, 1.0]
    env.weights = None


    # Q_hulls[c, p1, p2, action] = list of vectors (each vector has 3 components)
    Q_hulls = {}
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                for a in range(n_actions):
                    Q_hulls[(c, p1, p2, a)] = np.array([np.zeros(n_rewards)])
    
    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    iteration = 0
    total_states = len(env.states_agent_left) * len(env.states_agent_right) ** 2 # n_cells * n_cells * n_cells

    print(f"Starting LGCHVI with {total_states} states and {n_actions} actions")
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

                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        for action in range(n_actions):
                            # Store old hull for convergence check
                            old_hull = Q_hulls[(c, p1, p2, action)]

                            if iteration == 1:

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
                            # Q(s,a) = CH( ∪_{outcomes} [prob * (reward_vec + γ * hull(s'))] )

                            outcome_hulls = []
                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    # Terminal state - Q(s,a) = reward vector scaled by prob
                                    outcome_hull = prob * np.array([reward_vect])
                                else:
                                    next_c, next_p1, next_p2 = next_state

                                    all_next_q_vectors = []

                                    for next_action in range(n_actions):
                                        next_hull = Q_hulls[(next_c, next_p1, next_p2, next_action)]
                                        all_next_q_vectors.extend(next_hull)                               
                                
                                    # Complex hull of union
                                    if len(all_next_q_vectors) > 0:
                                        all_next_q_vectors = np.array(all_next_q_vectors)
                                        # CH vertices
                                        next_state_hull = get_hull(all_next_q_vectors)
                                    else:
                                        next_state_hull = np.array([np.zeros(n_rewards)])

                                    # For this outcome: reward_vect + γ * hull(s')
                                    # translate_hull does: reward_vect + gamma * hull
                                    outcome_hull = translate_hull(
                                        reward_vect,
                                        discount_factor,
                                        next_state_hull
                                    )

                                    if not isinstance(outcome_hull, np.ndarray):
                                        outcome_hull = np.array(outcome_hull)

                                    # Weight this hull by probability: prob * hull
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
                                                        
                            # Store new hull
                            Q_hulls[(c,p1,p2,action)] = new_hull

                            new_hull_size = len(new_hull)
                            total_hull_vertices += new_hull_size
                            num_hulls += 1

                            # Convergence data
                            if old_hull.shape == new_hull.shape:
                                max_diff = np.max(np.abs(new_hull - old_hull))
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
    
    # env.weights = weight_vect
    env.weights = saved_weights
    
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
                    # max_{q ∈ hull} w · q
                    q_value = max_q_value(weights, hull)
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                policy[c, p1, p2] = best_action
    print("Policy extraction complete!")
    return policy

def lex_max_hull(hulls_per_action, priority='car'):
    """
    Lexicographic maximisation over Q-hulls.
    
    For each action, we have a hull (set of Q-vectors).
    We need to find the action whose hull contains the lexicographically best vector.
    
    Args:
        hulls_per_action: List of hulls, one per action. Each hull is array of shape (n_vertices, n_objectives)
        priority: 'car' or 'pedestrian'
    
    Returns:
        Index of the lexicographically best action
    """
    n_actions = len(hulls_per_action)

    if priority == 'car':
        objective_order = [0,1,2]
    else:
        objective_order = [1,2,0]
    
    best_actions = list(range(n_actions))
    tol = 1e-9

    # for each obj in the priority order
    for obj_idx in objective_order:
        if len(best_actions) <= 1:
            break
        
        # for each action find the max value of this obj in its hull
        max_values_per_action = []
        for action in best_actions:
            hull = hulls_per_action[action]
            if not isinstance(hull, np.ndarray):
                hull = np.array(hull)

            # max value of the objective across all vectors in this hull
            if len(hull) > 0:
                max_val_in_hull = np.max(hull[:, obj_idx])
            else:
                max_val_in_hull = -np.inf
            
            max_values_per_action.append(max_val_in_hull)

        global_max = max(max_values_per_action) # global max for this obj

        # keep actions that achieve global max
        new_best_actions = []
        for i, action in enumerate(best_actions):
            if abs(max_values_per_action[i] - global_max) < tol:
                new_best_actions.append(action)

        best_actions = new_best_actions

        if len(best_actions) == 0:
            print(f"WARNING: No actions left in lex_max_hull! Returning 0")
            return 0
    
    return best_actions[0]

def extract_lexicographic_policy(Q_hulls, priority, env, n_actions):
    """
    Extract a lexicographic policy from Q-hulls.
    
    This is the WEIGHT-FREE approach: for each state, choose the action whose hull
    contains the lexicographically best Q-vector.
    
    Args:
        Q_hulls: Dictionary of Q-hulls from CHVI
        priority: 'car' for lexP or 'pedestrian' for lexA
        env: Environment
        n_actions: Number of actions
    
    Returns:
        policy: Array of shape [n_cells, n_cells, n_cells]
    """
    n_cells = env.map_num_cells
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
    
    print(f"\nExtracting LEXICOGRAPHIC policy from Q-hulls")
    print(f"Priority: {priority.upper()}")
    print("-" * 80)
    
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                # Collect hulls for all actions in this state
                hulls_per_action = []
                for action in range(n_actions):
                    hull = Q_hulls[(c, p1, p2, action)]
                    if not isinstance(hull, np.ndarray):
                        hull = np.array(hull)
                    hulls_per_action.append(hull)
                
                # Find lexicographically best action
                best_action = lex_max_hull(hulls_per_action, priority=priority)
                policy[c, p1, p2] = best_action
    
    return policy

def extract_lexP_policy(Q_hulls, env, n_actions):
    """Extract lexP policy (car priority) from Q-hulls."""
    return extract_lexicographic_policy(Q_hulls, priority='car', env=env, n_actions=n_actions)


def extract_lexA_policy(Q_hulls, env, n_actions):
    """Extract lexA policy (pedestrian priority) from Q-hulls."""
    return extract_lexicographic_policy(Q_hulls, priority='pedestrian', env=env, n_actions=n_actions)
