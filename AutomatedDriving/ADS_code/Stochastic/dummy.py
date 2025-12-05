from ADS_Environment import Environment
import numpy as np

env = Environment()
env.reset()

pedestrian_stochastic_actions = env.agents[1].move_map[3][3]

print(pedestrian_stochastic_actions, "\n")
print(env.agents[1])

stochastic_state = [3, 3]
state = env.get_state()

stochastic_state_translated = env.translate_state(stochastic_state)
state_translated = env.translate_state(state)

state_translated[1] == stochastic_state
print(np.array_equal(state_translated[1], stochastic_state), "\n")

