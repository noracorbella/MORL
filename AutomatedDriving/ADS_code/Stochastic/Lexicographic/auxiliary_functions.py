import time
import threading

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

    # store vect rewards
    all_r_c = []
    all_r_p1 = []
    all_r_p2 = []
    all_steps = []
    all_collisions = []

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()
        done = False
        step_count = 0

        ep_r_c = 0.0
        ep_r_p1 = 0.0
        ep_r_p2 = 0.0
        collisions = 0

        while not done and step_count < 200:
            c, p1, p2 = state[0], state[1], state[2]
            action = policy[c, p1, p2]

            next_state, reward_vector, done_array = env.step([action])
            done = done_array[0]

            r_c, r_p1, r_p2 = reward_vector[0], reward_vector[1], reward_vector[2]

            ep_r_c += r_c
            ep_r_p1 += r_p1
            ep_r_p2 += r_p2
            

            if r_p1 < 0 or r_p2 < 0:
                collisions += 1 
            
            state = next_state
            step_count += 1

        all_r_c.append(ep_r_c)
        all_r_p1.append(ep_r_p1)
        all_r_p2.append(ep_r_p2)
        all_steps.append(step_count)
        all_collisions.append(collisions)

    results = {
        'mean_r_c': np.mean(all_r_c),
        'mean_r_p1': np.mean(all_r_p1),
        'mean_r_p1': np.mean(all_r_p1),
        'std_r_c': np.std(all_r_c),
        'std_r_p1': np.std(all_r_p1),
        'std_r_p2': np.std(all_r_p2),
        'mean_steps': np.mean(all_steps),
        'total_collisions': np.sum(all_collisions),
        'collision_rate': np.sum(all_collisions) / num_episodes
    }

    if verbose:
        print(f"\nEpisodes tested: {num_episodes}")
        print(f"Mean steps per episode: {results['mean_steps']:.1f}")
        print("\nMean Vectorial Rewards per Episode:")
        print(f"  r_c:  {results['mean_r_c']:8.2f} $\pm$ {results['std_r_c']:.2f}")
        print(f"  r_p1: {results['mean_r_p1']:8.2f} $\pm$ {results['std_r_p1']:.2f}")
        print(f"  r_p2: {results['mean_r_p2']:8.2f} $\pm$ {results['std_r_p2']:.2f}")
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

            actions = policy[c, p1, p2]
            actions = [aciton]

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
        threading.Thread(target=example_execution, args=(env, policy, drawing,)).start()
        if drawing:
            env.render('Evaluating')