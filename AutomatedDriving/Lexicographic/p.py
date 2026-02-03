from LG_utils import generate_all_priority_orders
import numpy as np
from ADS_Environment import Environment

env = Environment()

print(env.map_num_cells)
print('len states_agent_left', len(env.states_agent_left))
print('len states_agent_right', len(env.states_agent_right))



# actions = [0,1,2,3,4,5]
# for i in range(1, len(actions)):
#     print(i)

# objective_order = [1,2,0]

# for obj_idx in objective_order:
#     print(obj_idx)

# priority_orders = generate_all_priority_orders(3)
# lex_optimal_actions = {}

# for order in priority_orders:
#     print(order, "\t")
#     print(tuple(order), "\n")

#     order_tuple = tuple(order) # from [0,1,2] to (0,1,2)
#     lex_optimal_actions[order_tuple] = np.random.randint(0,10)

# print(lex_optimal_actions)

n_actions=6

best_actions = list(range(n_actions))

obj_values = [0, 10, 10, 5, 3, 10]

max_val = np.max(obj_values)
print('max_val', max_val)
new_best_actions = []

for i, action in enumerate(best_actions):
    print('obj_values[i]', obj_values[i])
    print('action', action)
    if abs(obj_values[i] - max_val) < 1e-9:
        new_best_actions.append(action)
        print('new_best_actions', new_best_actions)


optimal_action_set = set()
print('optimal_action_set', optimal_action_set)


n_cells = env.map_num_cells

V = {}
for c in range(n_cells):
    for p1 in range(n_cells):
        for p2 in range(n_cells):
            V[(c, p1, p2)] = np.zeros((1, 3))
print(V)
