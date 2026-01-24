import numpy as np
from tqdm import tqdm
from LG_utils import lex_hull, generate_all_priority_orders


def LG_VI_lexhull_debug(env, theta=1.0, discount_factor=0.7):
    """
    Debug version with detailed logging.
    """

    # Initialize value function and policy
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    # V table stores vectorial values - one vector per state
    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])
    model_next_state = {}  # dict to store multiple possible next states

    # Q-table stores vectorial Q-values
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    iteration = 0
    total_states = len(env.states_agent_left) * len(env.states_agent_right) ** 2

    print(f"Starting Lexicographic Hull Value Iteration (DEBUG MODE)")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")
    print(f"Discount factor: {discount_factor}")
    print(f"Computing policies for all {len(generate_all_priority_orders(n_objectives))} lexicographic orders")

    # Track some sample states
    sample_states = [(2, 3, 3), (3, 3, 4), (0, 0, 0)]
    
    while True:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")

        delta = 0
        max_delta_state = None
        states_with_change = 0

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

                            # Build/retrieve model of state transitions
                            if iteration == 1:
                                outcomes = []  # (next_state, reward, done, probability)

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
                                            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2)
                                            outcomes.append((next_state, reward_vect, done, prob))

                                model_next_state[(c, p1, p2, action)] = outcomes

                            else:
                                outcomes = model_next_state[(c, p1, p2, action)]

                            # Compute Q-vector for this state-action pair
                            q_vector = np.zeros(n_objectives)

                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    q_vector += prob * reward_vect
                                else:
                                    next_value = V[next_state[0], next_state[1], next_state[2]]
                                    q_vector += prob * (reward_vect + discount_factor * next_value)

                            q_vectors[action] = q_vector

                        # For lexicographic hull, we use a reference priority order for convergence
                        lex_optimal_actions = lex_hull(q_vectors, n_objectives=n_objectives)
                        
                        # Use car-priority [0,1,2] as reference for convergence checking
                        reference_priority = (0, 1, 2)
                        best_action_reference = lex_optimal_actions[reference_priority]
                        v_new = q_vectors[best_action_reference]

                        # Calculate change for this state
                        state_delta = np.max(np.abs(v_old - v_new))
                        
                        if state_delta > 1e-9:
                            states_with_change += 1
                        
                        # Update delta - maximum change in value function
                        if state_delta > delta:
                            delta = state_delta
                            max_delta_state = (c, p1, p2)
                        
                        # Update V and Q
                        V[c, p1, p2] = v_new
                        Q[c, p1, p2] = q_vectors
                        
                        # Debug: Track sample states
                        if (c, p1, p2) in sample_states and iteration <= 5:
                            print(f"\n  Sample state {(c, p1, p2)}:")
                            print(f"    V_old: {v_old}")
                            print(f"    V_new: {v_new}")
                            print(f"    Delta: {state_delta:.6f}")
                            print(f"    Best action: {best_action_reference}")

                        pbar.update(1)

        print(f"\nIteration {iteration} Summary:")
        print(f"  Delta (max change): {delta:.6f}")
        print(f"  Max delta at state: {max_delta_state}")
        print(f"  States with change > 1e-9: {states_with_change}/{total_states}")
        print(f"  Theta: {theta}")
        
        # Show the V-value at the max delta state
        if max_delta_state is not None:
            print(f"\n  V-value at max delta state {max_delta_state}:")
            print(f"    V = {V[max_delta_state]}")
            print(f"    Q-values = {Q[max_delta_state]}")
        
        # Sample some V-values
        print(f"\n  Sample V-values:")
        for state in sample_states[:3]:
            print(f"    V{state} = {V[state]}")
        
        # Count non-zero V-values
        nonzero_v_states = np.sum(np.any(V != 0, axis=-1))
        print(f"\n  States with non-zero V-values: {nonzero_v_states}/{total_states}")

        # Check convergence
        if delta < theta:
            print(f"\n{'='*80}")
            print(f"CONVERGED: Delta = {delta:.6f} < Theta = {theta}")
            print(f"Total iterations: {iteration}")
            print(f"{'='*80}")
            break
        
        if iteration > 100:
            print(f"\nWARNING: Stopped after {iteration} iterations without convergence!")
            break

    # Extract policies for ALL lexicographic orders
    print("\nExtracting policies for all lexicographic orders...")
    
    all_priority_orders = generate_all_priority_orders(n_objectives)
    policies = {}
    
    for priority_order in all_priority_orders:
        priority_tuple = tuple(priority_order)
        policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
        
        for c in range(n_cells):
            for p1 in range(n_cells):
                for p2 in range(n_cells):
                    lex_optimal_actions = lex_hull(Q[c, p1, p2], n_objectives=n_objectives)
                    policy[c, p1, p2] = lex_optimal_actions[priority_tuple]
        
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")

    print(f"\nTotal policies extracted: {len(policies)}")
    
    return policies, Q