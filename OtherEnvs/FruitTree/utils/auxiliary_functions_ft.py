"""
Fruit Tree - Tester 
"""

import threading
import time
import numpy as np
import mo_gymnasium as mo_gym

def example_execution(env, policy, render=False, n_test_episodes=5):

    max_timesteps = 50
    action_names  = {0: "LEFT", 1: "RIGHT"}

    for episode in range(n_test_episodes):
        print(f"Episode {episode + 1}")

        obs, info = env.reset()
        if render:
            env.render()
        state = tuple(int(x) for x in obs)   # (depth, node)
        print(f"Initial state: {state}")

        done = False # Mark episode as not done yet
        timesteps = 0
        total_reward  = np.zeros(6)

        while timesteps < max_timesteps and not done:
            timesteps += 1
            action = policy[state[0], state[1]]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if render:
                env.render()         
                time.sleep(0.4)
            
            done = terminated or truncated
            state = tuple(int(x) for x in obs)
            total_reward += reward

            print(f" t={timesteps:2d} | state={state} | "
                  f"action={action_names[action]} | reward={np.round(reward, 3)}")

            if done:
                print(f"\nLeaf reached! Total reward: {np.round(total_reward, 3)}")

            if render:
                time.sleep(0.4)

    print("\nAll episodes finished.")


class tester:
    """
    Wrapper that runs example_execution in a thread so the pygame
    window (launched by mo_gym) stays responsive on the main thread.
    """

    def __init__(self, env, policy, render=False, n_test_episodes=2):
        threading.Thread(
            target=example_execution,
            args=(env, policy, render),
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


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    env = mo_gym.make("fruit-tree-v0", depth=6, render_mode="human")
 
    tree_depth = env.unwrapped.tree_depth
    max_nodes  = 2 ** tree_depth
 
    # Simple policy: always go left
    policy = np.zeros((tree_depth + 1, max_nodes), dtype=int)
 
    tester(env, policy, render=True)