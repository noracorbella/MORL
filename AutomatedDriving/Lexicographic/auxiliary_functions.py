import time
import threading
import numpy as np
import pickle
from tqdm import tqdm


RIGHT = 0
UP = 1
LEFT = 2



def translate_action(action):
    """
    Specific to the public civility game environmment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    if action < 3:
        part_1 = "MOVE SLOW "
    else:
        part_1 = "MOVE FAST "

    if action % 3 == RIGHT:
        part_2 = "RIGHT"
    elif action % 3 == UP:
        part_2 = "FORWARD"
    else:
        part_2 = "LEFT"

    action_name = part_1 + part_2
    return action_name

# simulation of the car driving using a learned policy
# render = False, to not show the visualisation
def example_execution(env, policy, render=False, stop=False):
    """

    Simulation of the environment without learning.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :param render:
    :param stop:
    :return:
    """
    max_timesteps = 200

    for i in range(10): # run 10 episodes
        timesteps = 0
        env.reset()

        state = env.get_state()

        print("State :", state)
        done = False # Mark episode as not done yet

        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1

            actions = list()
            actions.append(policy[state[0], state[1], state[2]])

            # DEBUGGING
            if stop:
                actions = [LEFT, RIGHT, RIGHT]

            state, rewards, dones = env.step(actions)
            done = dones[0]  # R Agent does not interfere

            print("State :", state)
            print("Action performed : ", translate_action(actions[0]))
            print("Reward received : ", rewards)
            print("Done : ", done)
            print()

            if done:
                print("-- Agent reached goal. Starting new episode --")

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()


class tester:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, env, policy, drawing=False):
        threading.Thread(target=example_execution, args=(env, policy, drawing,)).start()
        if drawing:
            env.render('Evaluating')

def test_policy_vectorial(env, policy, num_episodes=100, verbose=True):
    import numpy as np
    # store vect rewards
    all_r_car = []
    all_r_ped1 = []
    all_r_ped2 = []
    all_steps = []
    all_collisions = []

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()
        done = False
        step_count = 0

        ep_r_car = 0.0
        ep_r_ped1 = 0.0
        ep_r_ped2 = 0.0
        collisions = 0

        while not done and step_count < 200:
            c, p1, p2 = state[0], state[1], state[2]
            action = policy[c, p1, p2]

            next_state, reward_vector, done_array = env.step([action])
            done = done_array[0]

            r_car, r_ped1, r_ped2 = reward_vector[0], reward_vector[1], reward_vector[2]

            ep_r_car += r_car
            ep_r_ped1 += r_ped1
            ep_r_ped2 += r_ped2


            if r_ped1 < 0 or r_ped2 < 0:
                collisions += 1

            state = next_state
            step_count += 1

        all_r_car.append(ep_r_car)
        all_r_ped1.append(ep_r_ped1)
        all_r_ped2.append(ep_r_ped2)
        all_steps.append(step_count)
        all_collisions.append(collisions)

    results = {
        'mean_r_car': np.mean(all_r_car),
        'mean_r_ped1': np.mean(all_r_ped1),
        'mean_r_ped2': np.mean(all_r_ped2),
        'std_r_car': np.std(all_r_car),
        'std_r_ped1': np.std(all_r_ped1),
        'std_r_ped2': np.std(all_r_ped2),
        'mean_steps': np.mean(all_steps),
        'total_collisions': np.sum(all_collisions),
        'collision_rate': np.sum(all_collisions) / num_episodes
    }

    if verbose:
        print(f"\nEpisodes tested: {num_episodes}")
        print(f"Mean steps per episode: {results['mean_steps']:.1f}")
        print("\nMean Vectorial Rewards per Episode:")
        print(f"  r_car :       {results['mean_r_car']:8.2f} ± {results['std_r_car']:.2f}")
        print(f"  r_ped1:     {results['mean_r_ped1']:8.2f} ± {results['std_r_ped1']:.2f}")
        print(f"  r_ped2:     {results['mean_r_ped2']:8.2f} ± {results['std_r_ped2']:.2f}")
        print(f"\nTotal collisions: {results['total_collisions']}")
        print(f"Collision rate: {results['collision_rate']*100:.1f}% of steps")


def example_execution_vect(env, policy, render=False):

    max_timesteps = 200

    for i in range(10): # run 10 episodes
        timesteps = 0
        env.reset()
        state = env.get_state()

        ep_r_c = 0.0
        ep_r_p1 = 0.0
        ep_r_p2 = 0.0

        print(f"Episode {i+1}\n")
        print(f"Initial State: {state}\n")

        done = False # Mark episode as not done yet

        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1

            c, p1, p2 = state[0], state[1], state[2]

            action = policy[c, p1, p2]
            actions = [action]

            # # DEBUGGING
            # if stop:
            #     actions = [LEFT, RIGHT, RIGHT]

            state, rewards, dones = env.step(actions)
            done = dones[0]  # R Agent does not interfere

            r_c, r_p1, r_p2 = rewards[0], rewards[1], rewards[2]
            ep_r_c += r_c
            ep_r_p1 += r_p1
            ep_r_p2 += r_p2

            print(f"Step {timesteps}: State={state}, Action={action}")
            print(f"Reward: [r_car={r_c:.1f}, r_ped1={r_p1:.1f}, r_ped2={r_p2:.1f}]")


            # print("State :", state)
            # print("Action performed : ", translate_action(actions[0]))
            # print("Reward received : ", rewards)
            # print("Done : ", done)
            # print()

            if done:
                print(f"-- Agent reached goal in {timesteps} steps\n")
                print(f"Episoe rewards: [r_c={ep_r_c:.1f}, r_p1={ep_r_p1:.1f}, r_p2={ep_r_p2:.1f}]\n")
                print("Starting new episode --")

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()

class tester_vect:
    def __init__(self, env, policy, drawing=False):
        threading.Thread(target=example_execution_vect, args=(env, policy, drawing,)).start()
        if drawing:
            env.render('Evaluating')


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
        policy_file = f"policies/LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"
        v_table_file = f"policies/LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_V_table.pkl"

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
        policy_file = f"policies/LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"
        v_hulls_file = f"policies/LG_VI_lexhull_V_hulls.pkl"

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




