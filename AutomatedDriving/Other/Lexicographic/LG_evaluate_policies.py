import time
import threading
import numpy as np
import pickle
from tqdm import tqdm


def evaluate_policy(env, policy, n_eval_episodes=10, max_steps=200, discount_factor=0.7):
    """
    Evaluate the agent for `n_eval_episodes` episodes and returns average discounted rewards.
    :param policy: Policy array
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param max_steps: Maximum steps per episode
    :param discount_factor: Discount factor gamma for computing returns

    :returns results: dictionary containing evaluation results
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



LG_VI_LEXMAX = 1
LG_VI_LEXHULL = 2

if __name__ == "__main__":
    from ADS_Environment import Environment
    from LG_utils import lex_max

    algorithm_used = LG_VI_LEXHULL

    priority = [0, 1, 2]
    n_eval_episodes = 10000
    max_steps = 200
    discount_factor = 0.7

    print(f"Algorithm: {'LG VI lexmax' if algorithm_used == LG_VI_LEXMAX else 'LG VI lexhull'}")
    print(f"Priority: {priority}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Discount factor: {discount_factor}")

    policy = None
    Q_hulls = None

    if algorithm_used == LG_VI_LEXMAX:
        policy_file = f"lexicographic_policies/LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"
        v_table_file = f"lexicographic_policies/LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_V_table.pkl"

        try:
            policy = np.load(policy_file)
            print(f"\nLoaded VI policy from {policy_file}")
        except FileNotFoundError:
            print(f"\nERROR: Policy file not found: {policy_file}")
            print("Run VI training first or provide correct path.")
            exit(1)
        
        env_tmp = Environment(weights=None)

        try:
            with open(v_table_file, 'rb') as f:
                V = pickle.load(f)
            env_tmp.reset()
            initial_state = env_tmp.get_state()
            theoretical_value = V[initial_state[0], initial_state[1], initial_state[2]]
        except FileNotFoundError:
            theoretical_value = None
        
        del env_tmp

    elif algorithm_used == LG_VI_LEXHULL:
        policy_file = f"lexicographic_policies/LGVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"
        v_hulls_file = f"lexicographic_policies/LGVI_lexhull_V_hulls.pkl"

        try:
            policy = np.load(policy_file)
            print(f"\nLoaded lexhull policy from {policy_file}")
        except FileNotFoundError:
            print(f"\nERROR: Policy file not found: {policy_file}")
            print("Run lexhull training first.")
            exit(1)

        env_temp = Environment(weights=None)
        try:
            with open(v_hulls_file, 'rb') as f:
                V_hulls = pickle.load(f)
            env_temp.reset()
            initial_state = env_temp.get_state()
            state_tuple = (initial_state[0], initial_state[1], initial_state[2])
            v_hull = V_hulls[state_tuple]
            best_idx = lex_max(v_hull, priority=list(priority))
            theoretical_value, value_hull = v_hull[best_idx], v_hull
        except FileNotFoundError:
            theoretical_value, value_hull = None, None
        
        del env_temp

    env = Environment(weights=None)

    results = evaluate_policy(
        env=env,
        policy=policy,
        n_eval_episodes=n_eval_episodes,
        max_steps=max_steps,
        discount_factor=discount_factor
    )

    if theoretical_value is not None:
        print("\n THEORETICAL VALUE")
        print(f"V[initial_state] = {theoretical_value[0]:.4f}, {theoretical_value[1]:.4f}, {theoretical_value[2]:.4f}")
        if algorithm_used == LG_VI_LEXHULL and value_hull is not None:
            print(f"Hull size at initial state: {len(value_hull)} vertices")

    print(f"\nEVALUATE POLICY RESULTS")
    print(f"Number of episodes: {len(results['episode_returns'])}")
    print(f"Mean episode length: {results['mean_length']:.2f}")

    mean_vec = results['mean_return']
    std_vec = results['std_return']

    print(f"\nMean discounted vector return :")
    print(f"  [r_car, r_ped1, r_ped2] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")
    print(f"\nStd discounted vector return:")
    print(f"  [r_car, r_ped1, r_ped2] = [{std_vec[0]:.4f}, {std_vec[1]:.4f}, {std_vec[2]:.4f}]")




