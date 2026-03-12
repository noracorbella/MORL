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


VALUE_ITERATION = 1
CONVEX_HULL_VI = 2 

if __name__ == "__main__":
    from ADS_Environment import Environment
    from CHVI_stochastic import get_initial_state_hull, extract_policy_for_weights

    algorithm_used = VALUE_ITERATION

    weights = [1, 100, 10000]
    n_eval_episodes = 10000
    max_steps = 200
    discount_factor = 0.7

    print(f"Algorithm: {'Value Iteration' if algorithm_used == VALUE_ITERATION else 'Convex Hull VI'}")
    print(f"Weights: {weights}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Discount factor: {discount_factor}")

    policy = None
    Q_hulls = None

    if algorithm_used == VALUE_ITERATION:
        policy_file = f"stochastic_policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        try: 
            policy = np.load(policy_file)
            print(f"\nLoaded VI policy from {policy_file}")
        except FileNotFoundError:
            print(f"\nERROR: Policy file not found: {policy_file}")
            print("Run VI training first or provide correct path.")
            exit(1)

    elif algorithm_used == CONVEX_HULL_VI:
        hulls_file = "stochastic_policies/CH_VI_qhulls.pkl"
        chvi_policy_file = f"stochastic_policies/CHVI_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
        try:
            policy = np.load(chvi_policy_file)
            print(f"\nLoaded existing CHVI policy from {chvi_policy_file}")
        except FileNotFoundError:
            # Extract policy from Q-hulls
            print(f"\nCHVI policy not found. Extracting from Q-hulls")
            try:
                with open(hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
                print(f"Loaded Q-hulls from {hulls_file}")
                
                env_temp = Environment(weights=weights)
                policy = extract_policy_for_weights(Q_hulls, weights, env_temp, env_temp.n_actions)
                
                np.save(chvi_policy_file, policy)
                print(f"Saved extracted policy to {chvi_policy_file}")
                del env_temp

            except FileNotFoundError:
                print(f"\nERROR: Q-hulls file not found: {hulls_file}")
                print("Run CHVI training first.")
                exit(1)

        if Q_hulls is None:
            try:
                with open(hulls_file, 'rb') as f:
                    Q_hulls = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Could not load Q-hulls")
                Q_hulls = None

    if algorithm_used == CONVEX_HULL_VI and Q_hulls is not None:
        env = Environment(weights=weights)
        env.reset()
        initial_state = env.get_state()
        
        value_hull = get_initial_state_hull(Q_hulls, initial_state, env.n_actions)
        
        print(f"\n" + "="*60)
        print(f"VALUE HULL AT INITIAL STATE")
        print("="*60)
        print(f"Initial state indices: {initial_state}")
        print(f"Number of vertices: {len(value_hull)}")
        print(f"\nVertices (Pareto-optimal value vectors):")
        print(f"         [  r_car  ,  r_ped1 ,  r_ped2 ]")
        for i, v in enumerate(value_hull):
            print(f"[{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}]")
        print("="*60)


        del env
    

    env = Environment(weights=weights)

    results = evaluate_policy(
        env=env,
        policy=policy,
        n_eval_episodes=n_eval_episodes,
        max_steps=max_steps,
        discount_factor=discount_factor
    )

    print(f"\nNumber of episodes: {len(results['episode_returns'])}")
    print(f"Mean episode length: {results['mean_length']:.2f}")

    print(f"Weights: {weights}")

    mean_vec = results['mean_return']
    std_vec = results['std_return']
    
    print(f"\nMean discounted vector return :")
    print(f"  [r_car, r_ped1, r_ped2] = [{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")


    if weights is not None:
        weights = np.array(weights)
        scalarised_mean = np.dot(mean_vec, weights)
        scalarised_returns = np.dot(results['episode_returns'], weights)
        print(f"  Mean scalarised return: {scalarised_mean:.4f}")
        print(f"  Std scalarised return:  {np.std(scalarised_returns):.4f}")
        