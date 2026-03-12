import numpy as np
from tqdm import tqdm
import pickle
import os

ESTATS ESTOCASTICSS 


def vector_policy_evaluation(env, policy, theta=0.01, discount_factor=0.7, MNS_filename='policies/VI_stoc_MNS.pkl'):
    """
    Evaluate a policy in vector form to get the vaector-value function using cached model_next_state file
    
    :param env: the environment
    :param policy: the policy to evalueate
    :param theta: convergence threshold
    :param discount_factor: discount factor
    :param MNS_filename: filename for cached model_next_state
    :return: V_vector: vector-valued value funciton [n_cells, n_cells, n_cells, 3]
    """

    n_cells = env.map_num_cells
    n_actions = env.n_actions

    V_vect = np.zeros([n_cells, n_cells, n_cells, 3])

    pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
    stochastic_state = [3, 3]

    if not os.path.exists(MNS_filename):
        raise FileNotFoundError(f"Model file {MNS_filename} not found. Run VI training first.")
    
    print(f"Loading model from {MNS_filename}...")
    with open(MNS_filename, 'rb') as f:
        model_next_state = pickle.load(f)

    iteration = 0
    total_states = len(env.states_agent_left)*len(env.states_agent_right)**2

    print(f"Starting Vector Policy Evaluation with {total_states} states")

    while True:
        iteration += 1
        print(f"\nIteration {iteration}")

        delta = 0

        with tqdm(total=total_states, desc=f"Iteration {iteration}") as pbar:
            # Iterate through every possible state
            for c in env.states_agent_left:
                for p1 in env.states_agent_right:
                    for p2 in env.states_agent_right:
                        state = np.array([c, p1, p2])
                        state_translated = env.translate_state(state)

                        v_old = V_vect[c, p1, p2].copy()

                        action = policy[c, p1, p2] # action prescribed by the policy

                        
                        if (c, p1, p2, action) not in model_next_state:
                            # for this state action pair, if not cached, the state was never visited during training --> Keep V as zeros
                            pbar.update(1)
                            continue
                            
                        outcomes = model_next_state[(c, p1, p2, action)]

                        expected_v_vector = np.zeros(3)

                        for next_state, reward_vect, done, prob in outcomes:
                            if done:
                                expected_v_vector += prob * np.array(reward_vect)
                            else:
                                next_v_vector = V_vect[next_state[0], next_state[1], next_state[2]]
                                expected_v_vector += prob * (np.array(reward_vect) + discount_factor * next_v_vector)
                            

                        V_vect[c, p1, p2] = expected_v_vector

                        delta = max(delta, np.max(np.abs(v_old - expected_v_vector)))

                        pbar.update(1)

        
        print(f"Delta = {round(delta, 3)}, Theta = {theta}")

        if delta < theta:
            print(f"\nConverged in {iteration} iterations")
            break
    return V_vect

if __name__ == "__main__":
    from ADS_Environment import Environment

    weights_list = [
        [0, 10, 100],
        [0, 100, 10],
        [10, 0, 100],
        [10, 100, 0],
        [100, 0, 10],
        [100, 10, 0]
    ]

    initial_state = [43, 31, 45]

print("\n" + "#"*70)
print("VECTOR POLICY EVALUATION")
print("#"*70)

for i, weights in enumerate(weights_list, 1):
    print(f"\n{i}. Evaluating policy for weights {weights}")
    print("-" * 60)

    policy_file = f"policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"

    if not os.path.exists(policy_file):
        print(f"Policy file {policy_file} not found")
        continue

    policy = np.load(policy_file)

    env = Environment(weights=weights)

    V_vect = vector_policy_evaluation(env, policy, theta=0.01, discount_factor=0.7)

    # save
    vector_v_file = f"policies/V_vector_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
    with open(vector_v_file, 'wb') as f:
        pickle.dump(V_vect, f)
    print(f"Saved vector V-table to {vector_v_file}")

    # value of initial state
    c, p1, p2 = initial_state
    initial_value_vector = V_vect[c, p1, p2]

    weight_vect = np.array(weights)
    scalar_value = np.dot(initial_value_vector, weight_vect)

    print(f"\nVector Value of initial state: [{initial_value_vector[0]:.3f}, {initial_value_vector[1]:.3f}, {initial_value_vector[2]:.3f}]")
    print(f"Scalarised Value with weights {weights}: {scalar_value:.3f}")

    # compare with stored scalar V table
    v_file = f"policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
    if os.path.exists(v_file):
        with open(v_file, 'rb') as f:
            V_scalar = pickle.load(f)
        stored_scalar = V_scalar[c, p1, p2]
        print(f"Stored scalar V-table value: {stored_scalar:.3f}")
        