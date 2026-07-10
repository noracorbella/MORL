"""
Value Iteration (VI) algorithm for MOMDPs.

This is Value Iteration from Sutton & Barto, 4.4 applied to a
:class:`~morl.core.env_interface.MOEnv` by first scalarising each vector reward
with a fixed weight vector ``w``: the scalar reward of a transition is
``w · reward_vector``. It finds an optimal policy for the single
scalarised objective ``w`` and depends only on the :class:`MOEnv`.
"""

import numpy as np
from tqdm import tqdm

from morl.core.env_interface import MOEnv


def value_iteration(env: MOEnv, weights, theta=0.01):
    """
    Run Value Iteration and return an optimal policy and values.

    Each vector reward ``r`` is scalarised as ``dot(weights, r)`` and standard
    Value Iteration is run on the resulting scalar MDP. Updates are performed
    over the non-terminal states. Terminal states are never Bellman updated
    and have value ``0`` by definition. Convergence uses the max-norm: the sweep 
    stops once the largest change in any state's value drops below ``theta``.

    Parameters
    ----------
    env : MOEnv
        The environment. :class:`MOEnv`
    weights : sequence of float
        The scalarisation weights, one per objective. Must have length
        ``env.n_objectives``.
    theta : float
        Max-norm convergence threshold. Default ``0.01``.

    Returns
    -------
    policy : dict
        Map of non-terminal states to the greedy action.
    Q : dict
        Map of ``(state, action)`` pair over non-terminal states to its
        scalarised action value.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != env.n_objectives:
        raise ValueError(
            f"weights has length {weights.shape[0]},"
            f"env.n_objectives = {env.n_objectives}"
        )

    gamma = env.gamma

    non_terminal_states = [s for s in env.states() if not env.is_terminal(s)]

    # Scalar value function
    V = {s: 0.0 for s in non_terminal_states}

    def state_value(s):
        # Terminal states have value 0 by definition.
        return V[s] if s in V else 0.0

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        with tqdm(total=len(non_terminal_states), desc=f"Iteration {iteration}") as pbar:
            for s in non_terminal_states:
                v_old = V[s]

                best_q = -np.inf
                for a in env.actions(s):
                    q = 0.0
                    for prob, next_state, reward_vector in env.transitions(s, a):
                        reward_scalar = float(np.dot(weights, reward_vector))
                        q += prob * (reward_scalar + gamma * state_value(next_state))
                    if q > best_q:
                        best_q = q

                V[s] = best_q
                delta = max(delta, abs(v_old - best_q))

                pbar.update(1)

        if delta < theta:
            break

    # Sweep over the converged values
    # Store everything in Q and extract the greedy policy.
    Q = {}
    policy = {}
    for s in non_terminal_states:
        best_q = -np.inf
        best_action = None
        for a in env.actions(s):
            q = 0.0
            for prob, next_state, reward_vector in env.transitions(s, a):
                reward_scalar = float(np.dot(weights, reward_vector))
                q += prob * (reward_scalar + gamma * state_value(next_state))
            Q[(s, a)] = q
            if q > best_q:
                best_q = q
                best_action = a
        policy[s] = best_action

    return policy, Q
