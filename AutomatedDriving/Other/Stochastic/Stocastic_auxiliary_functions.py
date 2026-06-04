import time
import threading
import numpy as np
from tqdm import tqdm
import pickle


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
        part_1 = "MOVE SLOW"
    else:
        part_1 = "MOVE FAST"

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

