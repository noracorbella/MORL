import numpy as np
import os
import sys
from tqdm import tqdm
from mo_gymnasium.envs.breakable_bottles.breakable_bottles import BreakableBottles
from BB_Environment import BBEnvironment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))

from BB_utils import obs_to_state

def evaluate_policy(env, policy, n_eval_episodes=100, max_steps=200, discount_factor=0.95):
    """
    Evaluate policy for Breakable Bottles.
    Policy shape: (5, 3, 3, 2, 2, 2) -> action int.
    """
    eval_env = BreakableBottles(
        prob_drop=env.prob_drop,
        time_penalty=env.time_penalty,
        bottle_reward=env.bottle_reward,
        unbreakable_bottles=env.unbreakable_bottles,
    )
    episode_discounted_rewards = []
    episode_lengths            = []

    for episode in tqdm(range(n_eval_episodes), desc="Evaluating policy"):
        obs, _ = eval_env.reset()
        state  = obs_to_state(obs)

        total_discounted_reward = np.zeros(env.n_rewards)
        done = False
        step = 0

        while step < max_steps and not done:
            action = policy[state]
            obs, reward_vector, terminated, truncated, _ = eval_env.step(action)
            done  = terminated or truncated

            total_discounted_reward += (discount_factor ** step) * np.array(reward_vector)
            state = obs_to_state(obs)
            step += 1

        episode_discounted_rewards.append(total_discounted_reward)
        episode_lengths.append(step)

    eval_env.close()
    episode_discounted_rewards = np.array(episode_discounted_rewards)

    return {
        'episode_returns': episode_discounted_rewards,
        'mean_return':     episode_discounted_rewards.mean(axis=0),
        'std_return':      episode_discounted_rewards.std(axis=0),
        'episode_lengths': np.array(episode_lengths),
        'mean_length':     np.mean(episode_lengths),
    }