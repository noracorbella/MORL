import numpy as np
from tqdm import tqdm
from LG_utils import lex_max
from ADS_Environment import Environment

def LG_VI_lexmax(env, theta=1.0, discount_factor=0.7, priority = [0,1,2]):
    """
    Lexicographic Value Iteration Algorithm for a single priority order.

    This is the standard lexicographic VI algorithm that computes the optimal policy
    for one specific lexicographic ordering of objectives.

    Args:
        env: the environment encoding the MOMDP
        theta: convergence parameter, the smaller it is the more precise the algorithm
        discount_factor: discount factor of the MOMDP, can be set at discretion
        priority: lexicographic priority order (e.g., [0,1,2] for car first)
    
    Returns:
        policy: optimal policy for the given priority order
        
    """

    n_cells = env.map_num_cells
    n_actions = env.n_actions
    n_objectives = 3

    V = np.zeros([n_cells, n_cells, n_cells, n_objectives]) 
    model_next_state = {}


    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int) 
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives]) 

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]


    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2 #n_cells * n_cells * n_cells

    print(f"Starting Lexicographic Max Value Iteration")
    print(f"Total states: {total_states}, Actions: {n_actions}")
    print(f"Priority order: {priority}")

    while True:
        iteration += 1
        print(f"\n Iteration {iteration}")

        delta = 0
        states_processed = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            for c in env.states_agent_left:
                for p1 in env.states_agent_right:
                    for p2 in env.states_agent_right:
                        states_processed += 1
                        state = np.array([c, p1, p2])
                        state_translated = env.translate_state(state)

                        v_old = V[c, p1, p2].copy()

                        q_vectors = np.zeros((n_actions, n_objectives))

                        p1_is_stochastic = np.array_equal(state_translated[1], stochastic_state)
                        p2_is_stochastic = np.array_equal(state_translated[2], stochastic_state)

                        for action in range(n_actions):

                            if iteration == 1:
                                
                                outcomes = [] # (next_state, reward, done, probability)

                                if not p1_is_stochastic and not p2_is_stochastic:
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

                            if iteration == 1:
                                terminal_count = 0

                            q_vector = np.zeros(n_objectives)

                            for next_state, reward_vect, done, prob in outcomes:
                                if done:
                                    q_vector += prob * reward_vect
                                else:
                                    next_value = V[next_state[0], next_state[1], next_state[2]]
                                    q_vector += prob * (reward_vect + discount_factor * next_value)

                            q_vectors[action] = q_vector

                        Q[c, p1, p2] = q_vectors

                        best_action = lex_max(q_vectors, priority=priority)
                        V[c, p1, p2] = q_vectors[best_action]

                        delta = max(delta, np.max(np.abs(v_old - V[c, p1, p2])))

                        pbar.update(1)
                    
        if iteration == 1:
            print(f"Total terminal transitions found: {terminal_count}")
            print("\n[DEBUG] Finding source of max value:")
            max_val = np.max(np.abs(V))
            print(f"  Max absolute value in V: {max_val}")
            
            # Find where this max value occurs
            indices = np.where(np.abs(V) >= max_val - 0.01)
            print(f"  Number of locations with max value: {len(indices[0])}")
            
            for i in range(min(5, len(indices[0]))):  # Show first 5
                c, p1, p2, obj = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                print(f"    V[{c},{p1},{p2}] = {V[c,p1,p2]}, objective {obj} has value {V[c,p1,p2,obj]}")

        print(f"States processed: {states_processed}/{total_states}")

        print(f"Delta = {delta} (exact), Theta = {theta}")

        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
            print(f"Converged in {iteration} iterations")

            break

        if done and iteration == 1 and terminal_count <= 5:
            print(f"TERMINAL: state=({c},{p1},{p2}), action={action}, reward={reward_vect}")
            print(f"  This state's V BEFORE update: {V[c, p1, p2]}")
            print(f"  Q-vector for this action: {q_vectors[action] if action < len(q_vectors) else 'not computed yet'}")


    print("\nExtracting policy...")
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                policy[c, p1, p2] = lex_max(Q[c, p1, p2], priority=priority)

    return policy, Q
