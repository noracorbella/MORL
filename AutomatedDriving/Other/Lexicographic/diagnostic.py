from ADS_Environment import Environment
import numpy as np

env = Environment(weights=None)

initial_state = np.array([43, 31, 45])
translated = env.translate_state(initial_state)

print("Initial state indices:", initial_state)
print("Translated positions:")
print("  Car:", translated[0])
print("  Ped1:", translated[1])
print("  Ped2:", translated[2])

# Check if ped2 position 45 is a terminal/special state
print("\nPed2 in states_agent_right?", 45 in env.states_agent_right)