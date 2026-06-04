import time
import threading
import numpy as np
from tqdm import tqdm
import pickle


def evaluate_policy(env, policy, n_eval_episodes=10, max_steps=200, discount_factor=0.7):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param policy: Policy array 
    :param env: The evaluation environment
    :param max_steps: Maximum number of steps per episode
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param max_steps: Maximum steps per episode
    :param discount_factor: Discount factor gamma for computing returns
    """
    episode_discounted_rewards = []
    episode_lengths = []

    n_rewards = 3

    for episode in tqdm(range(n_eval_episodes), desc="Evaluating policy"):
        env.reset()
        state = env.get_state()

        total_discounted_reward = np.zeros(n_rewards)

        done = False
        step = 0

        while step < max_steps and not done:
            action = policy[state[0], state[1], state[2]]

            next_state, reward_vector, done_array = env.step([action])
            done = done_array[0]

            reward_vector = np.array(reward_vector)

            discounted_reward = (discount_factor ** step) * reward_vector
            total_discounted_reward += discounted_reward

            state = next_state
            step += 1

        
        episode_discounted_rewards.append(total_discounted_reward)
        episode_lengths.append(step)

    episode_discounted_rewards = np.array(episode_discounted_rewards)
    
    mean_reward = episode_discounted_rewards.mean(axis=0)
    std_reward = episode_discounted_rewards.std(axis=0)

    results = {
        'episode_returns': episode_discounted_rewards,
        'mean_return': mean_reward,
        'std_return': std_reward,
        'episode_lengths': np.array(episode_lengths),
        'mean_length': np.mean(episode_lengths),
    }
    
    return results
