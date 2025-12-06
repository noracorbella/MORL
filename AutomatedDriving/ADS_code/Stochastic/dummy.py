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

pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
print(len(pedestrian_stochastic_actions))

state_car = env.states_agent_left[1]
state_p1 = env.states_agent_right[2]
state_p2 = env.states_agent_right[3]
action = 1

weights = [1.0, 100.0, 100.0]
next_state, reward_vect, done_array = env.step([action])
print(next_state, reward_vect, done_array, "\n" )


# reward_scalar = np.dot(reward_vect, weight_vect)
# print(reward_scalar)

