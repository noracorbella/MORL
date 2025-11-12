from ADS_Environment import Environment

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
print(f"n_cells: {n_cells}")
