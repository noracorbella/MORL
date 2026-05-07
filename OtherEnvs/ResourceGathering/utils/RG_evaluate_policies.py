import numpy as np
from tqdm import tqdm
import mo_gymnasium as mo_gym


def evaluate_policy(env, policy, n_eval_episodes=10, max_steps=200, discount_factor=0.7):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.

    :param env: RGEnvironment
    :param policy: np.array of shape (5, 5, 2, 2)
    :param n_eval_episodes: number of episodes to evaluate
    :param max_steps: maximum steps per episode
    :param discount_factor: gamma for computing discounted returns
    :return: dict with episode_returns, mean_return, std_return, episode_lengths, mean_length
    """
    eval_env = mo_gym.make("resource-gathering-v0", enemy_kill_prob=env.enemy_kill_prob)
    episode_discounted_rewards = []
    episode_lengths            = []

    n_rewards = env.n_rewards  # 3

    for episode in tqdm(range(n_eval_episodes), desc="Evaluating policy"):
        obs, _ = eval_env.reset()
        state  = tuple(int(x) for x in obs)   # (row, col, has_gold, has_gem)

        total_discounted_reward = np.zeros(n_rewards)

        done = False
        step = 0

        while step < max_steps and not done:
            action = policy[state[0], state[1], state[2], state[3]]

            obs, reward_vector, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            reward_vector     = np.array(reward_vector)
            discounted_reward = (discount_factor ** step) * reward_vector
            total_discounted_reward += discounted_reward

            state = tuple(int(x) for x in obs)
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