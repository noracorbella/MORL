import numpy as np
from Universal import RandomEnv

def value_iteration_momdp(
    env,
    weight_vector: np.ndarray,
    gamma: float = 0.7,
    tol: float = 1e-5,
    max_iterations: int = 10_000,
):
    """
    Value Iteration for a Multi-Objective MDP via linear scalarization.

    Parameters
    ----------
    env : RandomEnv
        Environment exposing get_transition_tensor() and get_reward_tensor()
    weight_vector : np.ndarray, shape (num_objectives,)
        Linear scalarization weights
    gamma : float
        Discount factor
    tol : float
        Convergence tolerance (sup-norm)
    max_iterations : int
        Maximum number of iterations

    Returns
    -------
    policy : np.ndarray, shape (num_states,)
        Deterministic greedy policy
    V : np.ndarray, shape (num_states,)
        Optimal V-function
    Q : np.ndarray, shape (num_states, num_actions)
        Optimal Q-function
    """

    # --------------------------------------------------
    # Extract tensors
    # --------------------------------------------------
    T = env.get_transition_tensor()        # (S, A, S)
    R = env.get_reward_tensor()            # (K, S, A, S)
    action_mask = env.get_action_mask()    # (S, A)

    S, A, _ = T.shape
    K = R.shape[0]

    # --------------------------------------------------
    # Validate weight vector
    # --------------------------------------------------
    weight_vector = np.asarray(weight_vector, dtype=np.float64)

    if weight_vector.shape != (K,):
        raise ValueError(
            f"weight_vector must have shape ({K},), got {weight_vector.shape}"
        )

    # Optional normalization (theoretically irrelevant, numerically useful)
    weight_vector = weight_vector / np.linalg.norm(weight_vector, ord=1)

    # --------------------------------------------------
    # Scalarize reward tensor
    # Result: R_w(s,a,s')
    # --------------------------------------------------
    R_scalar = np.tensordot(
        weight_vector, R, axes=(0, 0)
    )  # shape: (S, A, S)

    # --------------------------------------------------
    # Initialize value function
    # --------------------------------------------------
    V = np.zeros(S, dtype=np.float64)

    # --------------------------------------------------
    # Value Iteration loop
    # --------------------------------------------------
    for iteration in range(max_iterations):
        V_prev = V.copy()

        # Expected immediate reward: E[R | s,a]
        expected_reward = np.sum(T * R_scalar, axis=2)  # (S, A)

        # Expected next-state value: E[V(s')]
        expected_value = np.sum(T * V_prev[None, None, :], axis=2)  # (S, A)

        Q = expected_reward + gamma * expected_value  # (S, A)

        # Mask invalid actions
        Q[~action_mask] = -np.inf

        # Bellman optimality update
        V = np.max(Q, axis=1)

        delta = np.max(np.abs(V - V_prev))

        print("Iteration ", iteration, " finished. Delta = ", delta)
        # Convergence check
        if delta < tol:
            break

    # --------------------------------------------------
    # Extract greedy policy
    # --------------------------------------------------
    policy = np.argmax(Q, axis=1)

    return policy, V, Q


if __name__ == "__main__":

    env = RandomEnv(seed=45, states=100, actions=5, actions_per_state=5, successors=1, objectives=3, terminal_states=1)

    n_objectives = env.reward_space.shape[0]

    weight_vector = np.ones(n_objectives)

    policy, V, Q = value_iteration_momdp(env, weight_vector)

    print("Finished!")
    print("Policy:", policy)