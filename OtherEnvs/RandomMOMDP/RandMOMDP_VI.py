import numpy as np
from tqdm import tqdm
import pickle
import os
from Universal import RandomEnv


def value_iteration(
    env, 
    weight_vector, 
    theta=1e-5, 
    discount_factor=0.7, 
    v_table_file=None):
    """
    Value Iteration for a Multi-Objective MDP via linear scalarization.
    Adapted from RG_VI.py to work with the RandomMOMDP tensor interface.

    :param env: RandomEnv
    :param weight_vector: linear scalarization weights, shape (num_objectives,)
    :param theta: convergence threshold (sup-norm on delta)
    :param discount_factor: discount factor
    :param v_table_file: optional path to save V table (pickle)
    :return: policy and Q-table (scalar)
    """

    T = env.get_transition_tensor()      # (S, A, S)
    R = env.get_reward_tensor()          # (K, S, A, S)
    action_mask = env.get_action_mask()  # (S, A)

    S, A, _ = T.shape
    K = R.shape[0]

    weight_vector = np.asarray(weight_vector, dtype=np.float64)
    if weight_vector.shape != (K,):
        raise ValueError(f"weight_vector must have shape ({K},), got {weight_vector.shape}")
    weight_vector = weight_vector / np.linalg.norm(weight_vector, ord=1)

    # Scalarized reward: R_w(s,a,s') = w . R(k,s,a,s')
    R_scalar = np.tensordot(weight_vector, R, axes=(0, 0))  # (S, A, S)

    V = np.zeros(S, dtype=np.float64)           # scalar value function
    V_vec = np.zeros((S, K), dtype=np.float64)  # vector value function
    Q = np.zeros((S, A), dtype=np.float64)
    Q_vec = np.zeros((S, A, K), dtype=np.float64)

    print(f"Starting Value Iteration with {S} states, {A} actions, {K} objectives")

    iteration = 0
    while True:
        iteration += 1
        V_prev = V.copy()
        V_vec_prev = V_vec.copy()

        # Compute Q for all states using old V (Jacobi-style)
        # with tqdm(total=S, desc=f"Iteration {iteration}") as pbar:
        for s in range(S):
            for a in range(A):
                if not action_mask[s, a]:
                    Q[s, a] = -np.inf
                    Q_vec[s, a] = 0.0
                    continue

                # E[R_w + gamma*V | s,a] = sum_s' T(s,a,s') * (R_w(s,a,s') + gamma*V(s'))
                q_scalar = np.sum(T[s, a] * (R_scalar[s, a] + discount_factor * V_prev))

                # Vector version: sum_s' T(s,a,s') * (R(k,s,a,s') + gamma*V_vec(s'))
                q_vector = np.sum(
                    T[s, a, :, None] * (R[:, s, a, :].T + discount_factor * V_vec_prev),
                    axis=0,
                )  # (K,)

                Q[s, a] = q_scalar
                Q_vec[s, a] = q_vector

                # pbar.update(1)

        # Update V after all Q values computed
        for s in range(S):
            best_action = int(np.argmax(Q[s]))
            V[s] = Q[s, best_action]
            V_vec[s] = Q_vec[s, best_action]

        delta = np.max(np.abs(V - V_prev))
        # print(f"Delta = {delta}, Theta = {theta}")
        print("Iteration ", iteration-1, " finished. Delta = ", delta)

        if delta < theta:
            print(f"\nDelta = {delta} < Theta = {theta}")
            print("Learning Process finished!")
            print(f"Converged in {iteration} iterations")
            break

    if v_table_file is not None:
        os.makedirs(os.path.dirname(v_table_file) if os.path.dirname(v_table_file) else '.', exist_ok=True)
        with open(v_table_file, 'wb') as f:
            pickle.dump(V, f)
        print(f"V table (scalar) saved to {v_table_file}")

        v_vec_file = v_table_file.replace('.pkl', '_vec.pkl')
        with open(v_vec_file, 'wb') as f:
            pickle.dump(V_vec, f)
        print(f"V table (vector) saved to {v_vec_file}")

    print("\nExtracting policy...")
    policy = np.array([
        int(np.argmax(Q[s])) if np.any(action_mask[s]) else -1
        for s in range(S)
    ])

    return policy, Q


if __name__ == "__main__":
    env = RandomEnv(seed=45, states=100, actions=5, actions_per_state=5,
                    successors=1, objectives=3, terminal_states=1)

    n_objectives = env.reward_space.shape[0]
    weight_vector = np.ones(n_objectives)

    policy, Q = value_iteration(env, weight_vector)

    print("Finished!")
    print("Policy:", policy)

