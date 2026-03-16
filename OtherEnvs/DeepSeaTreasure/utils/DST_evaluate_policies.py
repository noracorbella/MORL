import time
import threading
import numpy as np
from tqdm import tqdm
import pickle
import mo_gymnasium as mo_gym

def evaluate_policy(env, policy, n_eval_episodes=10, max_steps=50, discount_factor=0.7):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param policy: Policy array 
    :param env: The evaluation environment
    :param max_steps: Maximum number of steps per episode
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param max_steps: Maximum steps per episode
    :param discount_factor: Discount factor gamma for computing returns
    :return: dict with episode_returns, mean_return, std_return, episode_lengths, mean_length
    """
    eval_env = mo_gym.make("deep-sea-treasure-v0")
    episode_discounted_rewards = []
    episode_lengths = []

    n_rewards = env.n_rewards

    for episode in tqdm(range(n_eval_episodes), desc="Evaluating policy"):
        obs, info = eval_env.reset()
        state = tuple(obs)

        total_discounted_reward = np.zeros(n_rewards)

        done = False
        step = 0

        while step < max_steps and not done:
            action = policy[state[0], state[1]]

            obs, reward_vector, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            reward_vector = np.array(reward_vector)

            discounted_reward = (discount_factor ** step) * reward_vector
            total_discounted_reward += discounted_reward

            state = tuple(obs)
            step += 1

        
        episode_discounted_rewards.append(total_discounted_reward)  
        episode_lengths.append(step)                                

    eval_env.close()

    episode_discounted_rewards = np.array(episode_discounted_rewards) 

    mean_reward = episode_discounted_rewards.mean(axis=0)
    std_reward  = episode_discounted_rewards.std(axis=0)
 
    results = {
        'episode_returns': episode_discounted_rewards,
        'mean_return':     mean_reward,
        'std_return':      std_reward,
        'episode_lengths': np.array(episode_lengths),
        'mean_length':     np.mean(episode_lengths),
    }
 
    return results