from ADS_Environment import Environment
import numpy as np

env = Environment()
env.reset()

print("Initial state:", env.get_state())
print("Type of state:", type(env.get_state()))
print(
    "State shape:",
    env.get_state().shape if hasattr(env.get_state(), "shape") else "not an array",
)
print("Number of states:", env.n_states)
print("Number of actions:", env.n_actions)
print("Number of cells:", env.map_num_cells)
for sl in env.states_agent_left:
    print(f"States agent left: {sl}\n")
for sr in env.states_agent_right:
    print(f"States agent right: {sr}\n")



# Test a step
action = 0
next_state, reward, done = env.step([action])  # Note: with [action]
print("\nenv.step([action]):", env.step([action]))
print("\nAfter step with action [0]:")
print("Next state:", next_state)
print("Type of next_state:", type(next_state))
print("Reward:", reward)
print("Done:", done)

n_cells = env.map_num_cells
print(f"n_cells: {n_cells}\n")



Q_hulls = {}
for c in env.states_agent_left:
    for p1 in env.states_agent_right:
        for p2 in env.states_agent_right:
            for a in range(env.n_actions):
                Q_hulls[(c, p1, p2, a)] = [np.zeros(3)]
                
print(np.shape(Q_hulls))
print(f"\n{np.size(Q_hulls)}")