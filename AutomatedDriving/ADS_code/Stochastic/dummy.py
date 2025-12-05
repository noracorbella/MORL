from ADS_Environment import Environment
import numpy as np

env = Environment()
env.reset()

pedestrian_stochastic_actions = env.agents[1].move_map[3][3]

print(pedestrian_stochastic_actions, "\n")
print(env.agents[1])
