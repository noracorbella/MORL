import numpy as np
from tqdm import tqdm

# The Markov property in MDPs means that the future states and rewards are independent of past states and actions,


def value_iteration(env, theta=1.0, discount_factor=0.7):
    """
    Value Iteration Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 4.4,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic environment, and also
    adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the (MO)MPD, can be set at discretion
    :return: policy and Q-table
    """

    # Initialize value function and policy
    n_cells = env.map_num_cells
    n_actions = env.n_actions

    V = np.zeros([n_cells, n_cells, n_cells])  # V table: each entry represents how good is it to be in this state
    model_next_state = np.zeros([n_cells, n_cells, n_cells, n_actions, 3], dtype=int)
    model_next_reward = np.zeros([n_cells, n_cells, n_cells, n_actions])
    model_next_done = np.zeros([n_cells, n_cells, n_cells, n_actions])


    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)  # For each state, which action should we take?
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions])  # For each state-action pair, what's the expected total reward?

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]


    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2 #n_cells * n_cells * n_cells

    print(f"Starting Value Iteration with {total_states} states and {n_actions} actions")
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

                        v_old = V[c, p1, p2]

                        # Q-values for all actions in this state
                        q_values = np.zeros(n_actions)

                        # For this state, try every action
                        for action in range(n_actions):


                            # Take action and observe next state and reward
                            if iteration == 1:
                                # Sets environment to the state we are at
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

                            reward_scalar = reward

                            # Bellman optimality equation:
                            # V(s) <- max(actions) sum(states) T(s'|s, a)[R(s, a, s') + gamma * V(s')]
                            if done:
                                # Terminal state - no future value (q value is the immediate reward)
                                q_values[action] = reward_scalar
                            else:
                                # Q(s,a) = r(s,a) + γ * V(s')
                                q_values[action] = (
                                    reward_scalar
                                    + discount_factor
                                    * V[next_state[0], next_state[1], next_state[2]]
                                )

                        # Store Q-values for this state
                        Q[c, p1, p2] = q_values

                        # Update value function: V(s) = max_a Q(s,a)
                        V[c, p1, p2] = np.max(q_values)

                        # Update delta - maximum change in value function
                        delta = max(delta, abs(v_old - V[c, p1, p2]))

                        pbar.update(1)

        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        # Check convergence
        if delta < theta:
            print(f"\nDelta = {round(delta, 3)} < Theta = {theta}")
            print("Learning Process finished!")
            print(f"Converged in {iteration} iterations")
            break

    # Extract policy: for each state, choose action with best Q-value
    print("\nExtracting policy...")
    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                policy[c, p1, p2] = np.argmax(Q[c, p1, p2])

    return policy, Q
