from LG_utils import generate_all_priority_orders
import numpy as np



# actions = [0,1,2,3,4,5]
# for i in range(1, len(actions)):
#     print(i)

# objective_order = [1,2,0]

# for obj_idx in objective_order:
#     print(obj_idx)

# priority_orders = generate_all_priority_orders(3)
# lex_optimal_actions = {}

for order in priority_orders:
    print(order, "\t")
    print(tuple(order), "\n")

    order_tuple = tuple(order) # from [0,1,2] to (0,1,2)
    lex_optimal_actions[order_tuple] = np.random.randint(0,10)

print(lex_optimal_actions)

