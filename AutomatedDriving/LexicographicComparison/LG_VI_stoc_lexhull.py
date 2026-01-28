import numpy as np
from tqdm import tqdm
from LG_utils import lex_hull, generate_all_priority_orders

def LG_VI_lexhull(env, theta=1.0, discount_factor=0.7, priority=[0,1,2]):
    """
    Lexicographic Value Iteration Algorithm for all possible priority orders.
    
    This algorithm computes optimal policies for all possible lexicographic orderings
    of objectives simultaneously. Instead of training once per priority order, it
    maintains Q-values as vectors and extracts all policies after convergence.
    
    Args:
        env: the environment encoding the MOMDP
        theta: convergence parameter, the smaller it is the more precise the algorithm
        discount_factor: discount factor of the MOMDP, can be set at discretion
        priority: reference priority order for convergence (default [0,1,2])

    
    Returns:
        policies: dict mapping priority orders (as tuples) to their optimal policies
        Q: Q-table with shape [n_cells, n_cells, n_cells, n_actions, n_objectives]
    """


    # Initialise value function and policy
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])  # V table: each entry represents how good is it to be in this state
    model_next_state = {}  # dict to store multiple possible next states

    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])  # For each state-action pair, what's the expected total reward?

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    reference_priority = tuple(priority)

    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2 #n_cells * n_cells * n_cells

    print(f"Starting Lexicographic Hull Value Iteration")
    print(f"Total states: {total_states}, Actions: {n_actions}, Objectives: {n_objectives}")
    print(f"Reference priority for convergence: {reference_priority}")
    print(f"Computing policies for all {len(generate_all_priority_orders(n_objectives))} lexicographic orders")
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

                        # For lexicographic hull, we need to compute lex_max for ALL orders
                        # and update V accordingly. However, for value iteration convergence,
                        # we can use a representative priority order to check delta.
                        # A common approach: use the first lexicographic order [0,1,2]
                        lex_optimal_actions = lex_hull(q_vectors, n_objectives=n_objectives)
                        

                        # Use car-priority [0,1,2] as reference for convergence checking
                        best_action_reference = lex_optimal_actions[reference_priority]
                        v_new = q_vectors[best_action_reference]

                        V[c, p1, p2] = v_new

                        # Update delta - maximum change in value function
                        delta = max(delta, np.max(np.abs(v_old - v_new)))

                        pbar.update(1)

        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        # Check convergence
        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
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
                    lex_optimal_actions = lex_hull(Q[c, p1, p2], n_objectives=n_objectives)
                    policy[c, p1, p2] = lex_optimal_actions[priority_tuple]
        
        policies[priority_tuple] = policy
        print(f"  Extracted policy for priority order {priority_order}")

    print(f"\nTotal policies extracted: {len(policies)}")
    
    return policies, Q