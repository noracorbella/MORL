from RandMOMDP_VI import value_iteration
from valueIteration import value_iteration_momdp
from Universal import RandomEnv
import numpy as np



####################################################################
# valueIteration
####################################################################
env = RandomEnv(seed=45, states=100, actions=5, actions_per_state=5, 
                successors=1, objectives=3, terminal_states=1)

n_objectives = env.reward_space.shape[0]

weight_vector = np.ones(n_objectives)

policy_manel, V_manel, Q_manel = value_iteration_momdp(env, weight_vector)

####################################################################
# RandomMOMDP_VI
####################################################################

env = RandomEnv(seed=45, states=100, actions=5, actions_per_state=5,
                    successors=1, objectives=3, terminal_states=1)

n_objectives = env.reward_space.shape[0]
weight_vector = np.ones(n_objectives)

policy_nora, Q_nora = value_iteration(env, weight_vector)



print("="*60)
print("DEBUGGING")
print("="*60)
print(np.allclose(Q_manel, Q_nora))
print(np.max(np.abs(Q_manel - Q_nora)))
print("Policies match:", np.array_equal(policy_manel, policy_nora))

for s in range(Q_manel.shape[0]):
    max_q = np.max(Q_manel[s])
    num_ties = np.sum(Q_manel[s] == max_q)
    if num_ties > 1:
        print(f"State {s}: {num_ties} actions tied at Q={max_q}")

for s in range(Q_nora.shape[0]):
    max_q = np.max(Q_nora[s])
    num_ties = np.sum(Q_nora[s] == max_q)
    if num_ties > 1:
        print(f"State {s}: {num_ties} actions tied at Q={max_q}")