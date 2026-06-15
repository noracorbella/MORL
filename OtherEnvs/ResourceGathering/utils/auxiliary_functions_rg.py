"""
Resource Gathering - Tester (analogous to autonomous driving example_execution)
"""

import threading
import time
import numpy as np
import mo_gymnasium as mo_gym

def example_execution(env, policy, render=False, n_test_episodes=5):
    """
    Simulation of the Resource Gathering environment without learning.

    :param env: the MO-Gymnasium environment
    :param policy: np.array of shape (5, 5, 2, 2) mapping state (row, col, has_gold, has_gem) -> action
    :param render: whether to display the pygame window
    """
    max_timesteps = 50
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    for episode in range(n_test_episodes):
        print(f"Episode {episode + 1}")

        obs, info = env.reset()
        if render:
            env.render()
        state = tuple(int(x) for x in obs)   # (row, col, has_gold, has_gem)
        print(f"Initial state: {state}")

        done = False # Mark episode as not done yet
        timesteps = 0
        total_reward = np.zeros(env.n_rewards)

        while timesteps < max_timesteps and not done:
            timesteps += 1
            action = policy[state[0], state[1], state[2], state[3]]
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()
                time.sleep(0.4)

            done = terminated or truncated
            state = tuple(int(x) for x in obs)
            total_reward += reward

            print(f" t={timesteps:2d} | state={state} | "
                  f"action={action_names[action]} | reward={reward}")

            if done:
                print(f"\nEpisode finished! Total reward: {total_reward}")

            if render:
                time.sleep(0.4)

    print("\nAll episodes finished.")


class tester:
    """
    Wrapper that runs example_execution in a thread so the pygame
    window (launched by mo_gym) stays responsive on the main thread.
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

            finally:
                env.close()


# -- Policies ----------------------------------------------------------------

def make_policy(sea_map):
    """
    Simple policy: always go DOWN or RIGHT
    toward the nearest treasure.
    Shape: (11, 11) -> action int
    """
    policy = np.ones((11, 11), dtype=int)  # default: DOWN
    for col in range(9):
        policy[0, col] = 3   # RIGHT along row 0
    for row in range(11):
        policy[row, 9] = 1   # DOWN along col 9
    return policy



# -- Main ---------------------------------------------------------------------
 
if __name__ == "__main__":
    env = mo_gym.make("resource-gathering-v0", render_mode="human")
 
    # Simple policy: always go up (for demonstration)
    policy = np.zeros((5, 5, 2, 2), dtype=int)
 
    tester(env, policy, render=True)