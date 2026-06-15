"""
Random MOMDP - Tester (analogous to the Resource Gathering example_execution)
"""

import threading
import time
import numpy as np


def example_execution(env, policy, render=False, n_test_episodes=5):
    """
    Simulation of the random MOMDP without learning.

    :param env: RandMOMDPEnvironment
    :param policy: np.array of shape (n_states,) mapping state -> action (-1 = none)
    :param render: whether to print the textual render after each step
    :param n_test_episodes: number of trajectories to simulate
    """
    max_timesteps = 50
    gym_env = env.gym_env

    for episode in range(n_test_episodes):
        print(f"Episode {episode + 1}")

        obs, info = gym_env.reset()
        state = int(obs)
        if render:
            gym_env.render()
        print(f"Initial state: {state}")

        done = False # Mark episode as not done yet
        timesteps = 0
        total_reward = np.zeros(env.n_rewards)

        while timesteps < max_timesteps and not done:
            action = int(policy[state])

            if action < 0 or not env.action_mask[state, action]:
                print(f" No valid action for state {state}, stopping episode.")
                break

            timesteps += 1
            obs, reward, terminated, truncated, info = gym_env.step(action)

            if render:
                gym_env.render()
                time.sleep(0.4)

            done = terminated or truncated
            state = int(obs)
            total_reward += np.array(reward, dtype=float)

            print(f" t={timesteps:2d} | state={state} | "
                  f"action={action} | reward={np.array(reward, dtype=float)}")

            if done:
                print(f"\nEpisode finished! Total reward: {total_reward}")

    print("\nAll episodes finished.")


class tester:
    """
    Wrapper that runs example_execution in a thread
    """

    def __init__(self, env, policy, render=False, n_test_episodes=5):
        threading.Thread(
            target=example_execution,
            args=(env, policy, render, n_test_episodes),
            daemon=True
        ).start()

        if render:
            try:
                while threading.active_count() > 1:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
