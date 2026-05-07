import mo_gymnasium as mo_gym

env = mo_gym.make("deep-sea-treasure-mirrored-v0")
obs, _ = env.reset()
print("Start state:", tuple(obs))

import numpy as np
import os
import sys
import pickle
import mo_gymnasium as mo_gym


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(BASE_DIR, 'environment'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'algorithms'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))


    from DST_Environment import DSTEnvironment

    env = DSTEnvironment(weights=[1, 1000])
    print('treasure_states:', env.treasure_states)
    print('(1,0) in treasure_states:', (1,0) in env.treasure_states)
    non_terminal = env.non_terminal_states
    print('(1,0) in non_terminal_states:', (1,0) in non_terminal)