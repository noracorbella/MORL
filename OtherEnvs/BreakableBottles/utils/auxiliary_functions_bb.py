"""
Breakable Bottles - Tester (analogous to auxiliary_functions_rg.py)
"""

import threading
import time
import os
import sys
import numpy as np
import mo_gymnasium as mo_gym
from BB_Environment import BBEnvironment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from BB_utils import obs_to_state


def example_execution(env, policy, render=False, n_test_episodes=5):
    """
    :param env: the MO-Gymnasium environment
    :param policy: np.array of shape (5, 3, 3, 2, 2, 2)
                   mapping state -> action
    """
    max_timesteps = 100
    action_names  = {0: "LEFT", 1: "RIGHT", 2: "PICKUP"}

    for episode in range(n_test_episodes):
        print(f"Episode {episode + 1}")

        obs, _ = env.reset()
        if render:
            env.render()
        state = obs_to_state(obs)
        print(f"Initial state: {state}")

        done         = False
        timesteps    = 0
        total_reward = np.zeros(3)

        while timesteps < max_timesteps and not done:
            timesteps += 1
            action = policy[state]

            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
                time.sleep(0.4)

            done  = terminated or truncated
            state = obs_to_state(obs)
            total_reward += reward

            print(f" t={timesteps:2d} | state={state} | "
                  f"action={action_names[action]} | reward={reward}")

            if done:
                print(f"\nEpisode finished! Total reward: {total_reward}")

            if render:
                time.sleep(0.4)

    print("\nAll episodes finished.")


class tester:
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


if __name__ == "__main__":
    from mo_gymnasium.envs.breakable_bottles.breakable_bottles import BreakableBottles
    env = BreakableBottles(render_mode="human")
    policy = np.zeros((5, 3, 3, 2, 2, 2), dtype=int)  # always go left
    tester(env, policy, render=True)