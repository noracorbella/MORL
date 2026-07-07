"""Scalarised Value Iteration (VI) for multi-objective MDPs.

This is classic Value Iteration (Sutton & Barto, 4.4) applied to an
:class:`~morl.core.env_interface.MOEnv` by first scalarising each vector reward
with a fixed weight vector ``w``: the scalar reward of a transition is
``w . reward_vector``. It therefore finds an optimal policy for the single
scalarised objective ``w`` and depends only on the :class:`MOEnv` contract.

The public entry point is :func:`value_iteration`.
"""

import numpy as np
from tqdm import tqdm

from morl.core.env_interface import MOEnv


def value_iteration(env: MOEnv, weights, theta=0.01):
    """Run scalarised Value Iteration and return an optimal policy and values.

    Each vector reward ``r`` is scalarised as ``dot(weights, r)`` and standard
    Value Iteration is run on the resulting scalar MDP. Updates are performed
    in place (Gauss-Seidel) over the non-terminal states; terminal states are
    never backed up and have value ``0`` by definition (the reward for entering
    a terminal state is carried on the transition that leads into it, per the
    :class:`MOEnv` contract). Convergence uses the max-norm: the sweep stops
    once the largest change in any state's value drops below ``theta``.

    Parameters
    ----------
    env : MOEnv
        The environment. Only the :class:`MOEnv` interface is used; in
        particular the weight vector is *not* read from ``env`` (weights are not
        part of the environment).
    weights : sequence of float
        The scalarisation weights, one per objective. Must have length
        ``env.n_objectives``.
    theta : float, optional
        Max-norm convergence threshold. Smaller values give a more precise
        result at the cost of more iterations. Defaults to ``0.01``, matching
        the original DST configuration.

    Returns
    -------
    policy : dict
        Maps each non-terminal state to the greedy action (the action
        maximising the scalarised action value). Terminal states are absent.
    Q : dict
        Maps each ``(state, action)`` pair (over non-terminal states) to its
        converged scalarised action value. The state value is recovered as
        ``V(s) = max_a Q[(s, a)]`` if ever needed. Terminal states are absent.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != env.n_objectives:
        raise ValueError(
            f"weights has length {weights.shape[0]}, expected "
            f"env.n_objectives = {env.n_objectives}"
        )

    gamma = env.gamma

    non_terminal_states = [s for s in env.states() if not env.is_terminal(s)]

    # Scalar value function; terminal states are implicitly 0 and never stored.
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

    # Final sweep over the converged values: store every scalarised action
    # value in Q and extract the greedy policy from it.
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
