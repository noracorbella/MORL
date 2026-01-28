import numpy as np
from tqdm import tqdm
from LG_utils import lex_hull, generate_all_priority_orders
from CH_operations import get_hull, translate_hull, sum_hulls

def LG_VI_lexhull(env, theta=1.0, discount_factor=0.7):
    """
    Lexicographic Hull Value Iteration Algorithm.
    
    This algorithm computes optimal policies for all possible lexicographic orderings
    of objectives simultaneously. It maintains Q-values as HULLS (sets of vectors)
    and extracts all policies after convergence.
    
    The key difference from standard CHVI: instead of using weighted scalarization
    during policy extraction, we use lexicographic comparison.
    
    Args:
        env: the environment encoding the MOMDP
        theta: convergence parameter (for maximum change in hull vertices)
        discount_factor: discount factor of the MOMDP
    
    Returns:
        policies: dict mapping priority orders (as tuples) to their optimal policies
        Q_hulls: Q-table where each entry is a hull (set of non-dominated vectors)
    """

    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    # Q_hulls[c, p1, p2, action] = numpy array of vectors (hull vertices)
    # Each vector has n_objectives components
    Q_hulls = {}
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                for a in range(n_actions):
                    Q_hulls[(c, p1, p2, a)] = np.array([np.zeros(n_objectives)])
    
    model_next_state = {}  # dict to store multiple possible next states
    
    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    iteration = 0
    total_states = len(env.states_agent_left) * len(env.states_agent_right) ** 2

    print(f"Starting Lexicographic Hull Value Iteration")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")
    print(f"Computing policies for all {len(generate_all_priority_orders(n_objectives))} lexicographic orders")
    print(f"Total evaluations per iteration: {total_states * n_actions}")

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

                        # Check if pedestrians are in stochastic state
                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        for action in range(n_actions):
                            # Store old hull for convergence check
                            old_hull = Q_hulls[(c, p1, p2, action)]

                            # Build transition model on first iteration
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
                                            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2)
                                            outcomes.append((next_state, reward_vect, done, prob))

                                model_next_state[(c, p1, p2, action)] = outcomes
                            else:
                                outcomes = model_next_state[(c, p1, p2, action)]

                            # Bellman update for convex hull:
                            # Q(s,a) = CH( ∪_{outcomes} [prob * (reward_vec + γ * hull(s'))] )

                            outcome_hulls = []
                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    # Terminal state
                                    outcome_hull = prob * np.array([reward_vect])
                                else:
                                    next_c, next_p1, next_p2 = next_state

                                    # Collect all Q-vectors from all actions at next state
                                    all_next_q_vectors = []
                                    for next_action in range(n_actions):
                                        next_hull = Q_hulls[(next_c, next_p1, next_p2, next_action)]
                                        all_next_q_vectors.extend(next_hull)

                                    # Compute convex hull of union (state value hull)
                                    if len(all_next_q_vectors) > 0:
                                        all_next_q_vectors = np.array(all_next_q_vectors)
                                        next_state_hull = get_hull(all_next_q_vectors)
                                    else:
                                        next_state_hull = np.array([np.zeros(n_objectives)])

                                    # For this outcome: reward_vect + γ * hull(s')
                                    outcome_hull = translate_hull(
                                        reward_vect,
                                        discount_factor,
                                        next_state_hull
                                    )

                                    if not isinstance(outcome_hull, np.ndarray):
                                        outcome_hull = np.array(outcome_hull)

                                    # Weight by probability
                                    outcome_hull = prob * outcome_hull

                                outcome_hulls.append(outcome_hull)

                            # Combine all outcome hulls
                            if len(outcome_hulls) == 1:
                                # Deterministic case
                                new_hull = outcome_hulls[0]
                            else:
                                # Stochastic case - sum hulls
                                combined_hull = outcome_hulls[0]
                                for outcome_hull in outcome_hulls[1:]:
                                    combined_hull = sum_hulls(combined_hull, outcome_hull)
                                new_hull = combined_hull

                            if not isinstance(new_hull, np.ndarray):
                                new_hull = np.array(new_hull)

                            # Take convex hull of the result
                            if len(new_hull) > 1:
                                new_hull = get_hull(new_hull)

                            # Store new hull
                            Q_hulls[(c, p1, p2, action)] = new_hull

                            # Track statistics
                            new_hull_size = len(new_hull)
                            total_hull_vertices += new_hull_size
                            num_hulls += 1

                            # Convergence check
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

        # Check convergence
        if delta < theta:
            print(f"Converged in {iteration} iterations")
            break

    print("\nExtracting policies for all lexicographic orders...")
    
    all_priority_orders = generate_all_priority_orders(n_objectives)
    policies = {}
    
    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
        
        for c in range(n_cells):
            for p1 in range(n_cells):
                for p2 in range(n_cells):
                    best_action = None
                    best_vector = None
                    
                    # For each action, find the lexicographically best vector in its hull
                    action_best_vectors = []
                    for action in range(n_actions):
                        hull = Q_hulls[(c, p1, p2, action)]
                        
                        # Find lexicographically best vector in this hull for this priority
                        lex_best = None
                        for vector in hull:
                            if lex_best is None:
                                lex_best = vector
                            else:
                                # Compare lexicographically according to priority_order
                                for obj_idx in priority_order:
                                    if vector[obj_idx] > lex_best[obj_idx]:
                                        lex_best = vector
                                        break
                                    elif vector[obj_idx] < lex_best[obj_idx]:
                                        break
                        
                        action_best_vectors.append((action, lex_best))
                    
                    # Now compare the best vectors across actions
                    best_action = action_best_vectors[0][0]
                    best_vector = action_best_vectors[0][1]
                    
                    for action, vector in action_best_vectors[1:]:
                        # Lexicographic comparison
                        for obj_idx in priority_order:
                            if vector[obj_idx] > best_vector[obj_idx]:
                                best_action = action
                                best_vector = vector
                                break
                            elif vector[obj_idx] < best_vector[obj_idx]:
                                break
                    
                    policy[c, p1, p2] = best_action
        
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")

    print(f"\nTotal policies extracted: {len(policies)}")
    
    return policies, Q_hulls