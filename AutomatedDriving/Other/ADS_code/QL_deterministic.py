import numpy as np
from tqdm import tqdm

# From dummy.py we examina how is our data
# Initial state: [43 45 31]
# Type of state: <class 'numpy.ndarray'>
# State shape: (3,)
# Number of states: 250047
# Number of actions: 6

# After step with action [0]:
# Next state: [44 38 24]
# Type of next_state: <class 'numpy.ndarray'>
# Reward: [-1.  0.  0.]
# Done: [False]
# n_cells: 63

# Implementation of Algorithm 3 Q-learning for MDPs (with epsilon-greedy policies) - Marl book Chapter 2
def q_learning(env, alpha=0.98, gamma=1.0, max_episodes=50000):
    """
    Q-Learning Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 6.5,
    (1998).

    :param env: the environment encoding the (MO)MDP
    :param alpha: the learning rate of the algorithm, can be set at discretion
    :param gamma: discount factor of the (MO)MPD, can be set at discretion (notice that this will change the Q-values)
    :param max_episodes: episodes taken into account in each q_learning
    :return: the learnt policy and its associated state-value (V) and state-action-value (Q) functions
    """

    # Initialize Q-table: [car_cell, ped1_cell, ped2_cell, action]
    n_cells = env.map_num_cells
    n_actions = env.n_actions

    # Q table - [c, p1, p2, action]
    Qtable = np.zeros([n_cells, n_cells, n_cells, n_actions])
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)
    visited_states = np.zeros([n_cells, n_cells, n_cells], dtype=bool)

    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005

    for episode in tqdm(range(1, max_episodes + 1), desc="Training Q-Learning"):

        

        env.reset()
        state = env.get_state()  # Returns [c, p1, p2] of initial positions
        done = False


        while not done: # done = True when car reaches destination
            
            visited_states[state[0], state[1], state[2]] = True

            
            # Epsilon greedy policy
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay_rate * episode
            )
            random_num = np.random.uniform(0, 1)

            if random_num < epsilon:  # Explore
                action = np.random.randint(n_actions) # Choose random action
            else:  # Exploit
                action = np.argmax(Qtable[state[0], state[1], state[2]]) # Choose action  a^t ∈ arg max_a Q(s^t, a) 

            # Take action 
            # env.step([action]): (array([44, 31, 23]), array([-1.,  0.,  0.]), [False])
            next_state, reward, done_array = env.step([action])
            done = done_array[0]  # Extract boolean from array

            # Reward c + p1 + p2
            reward_scalar = reward #np.sum(reward)

            # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            # alpha : estocasticitat
            best_next_action = np.argmax(Qtable[next_state[0], next_state[1], next_state[2]])
            Qtable[state[0], state[1], state[2], action] += alpha * (
                reward_scalar
                + gamma
                * Qtable[next_state[0], next_state[1], next_state[2], best_next_action]
                - Qtable[state[0], state[1], state[2], action]
            )

            state = next_state

    # Extract policy: for each state, choose action with best Q-value
    for c in range(n_cells):
        for p1 in range(n_cells):
            for p2 in range(n_cells):
                policy[c, p1, p2] = np.argmax(Qtable[c, p1, p2])

    total_states = n_cells * n_cells * n_cells
    num_visited_states = np.sum(visited_states)
    print(f"Total number of states: {total_states:,}")
    print(f"States visited: {num_visited_states:,}")
    print(f"Percentage: {100 * num_visited_states / total_states:.2f}%")
    
    return policy, Qtable

# n_cells = 63
# n_actions = 6
# Qtable = np.zeros([n_cells, n_cells, n_cells, n_actions])
# Q[43, 45, 31, 2] = "How good is action 2 when car is at cell 43, ped1 at 45, ped2 at 31?

