"""Lexicographic Value Iteration (LexVI) for multi-objective MDPs.

LexVI solves an MOMDP under a *lexicographic* preference over the objectives:
objectives are ranked by priority, and an action is preferred if it is better on
the highest-priority objective; ties there are broken by the next objective, and
so on. Unlike CHVI it commits to one priority order and carries a single
vector value per state (it is the scalar-style, non-hull algorithm — the old
DST "LGVI-LexMax").

It depends only on the :class:`~morl.core.env_interface.MOEnv` contract.

The public entry point is :func:`lexicographic_vi`.
"""

import numpy as np
from tqdm import tqdm

from morl.core.env_interface import MOEnv


def lex_max(q_vectors, priority, tol=1e-9):
    """Return the index of the lexicographically greatest action value.

    Compares the rows of ``q_vectors`` under the lexicographic order given by
    ``priority``: candidates are first narrowed to those maximising the
    highest-priority objective (within ``tol``), then the next objective breaks
    remaining ties, and so on. If a tie survives all objectives the
    lowest-indexed remaining action is returned.

    Parameters
    ----------
    q_vectors : numpy array of shape ``(n_actions, n_objectives)``
        The candidate action-value vectors, one row per action.
    priority : sequence of int
        Objective indices, highest priority first (see
        :func:`lexicographic_vi` for the full convention).
    tol : float, optional
        Tolerance within which two objective values count as tied.

    Returns
    -------
    int
        The row index of the lexicographically best action.
    """
    best_actions = list(range(q_vectors.shape[0]))
    for obj_idx in priority:
        if len(best_actions) == 1:
            break
        obj_values = [q_vectors[a, obj_idx] for a in best_actions]
        max_val = np.max(obj_values)
        best_actions = [
            a for a, v in zip(best_actions, obj_values) if abs(v - max_val) < tol
        ]
    return best_actions[0]


def lexicographic_vi(env: MOEnv, priority, theta=0.01):
    """Run Lexicographic Value Iteration and return an optimal policy and values.

    Vector-valued Value Iteration whose per-state action selection is a
    lexicographic argmax (:func:`lex_max`) under ``priority``. Updates are
    performed in place (Gauss-Seidel) over the non-terminal states; terminal
    states are never backed up and have the zero vector value (the reward for
    entering a terminal state is carried on the incoming transition, per the
    :class:`MOEnv` contract). Convergence uses the max-norm: the sweep stops once
    the largest per-component change in any state's value vector drops below
    ``theta`` (consistent with :func:`morl.algorithms.vi.value_iteration`).

    Parameters
    ----------
    env : MOEnv
        The environment. Only the :class:`MOEnv` interface is used; the priority
        order is *not* read from ``env``.
    priority : sequence of int
        The lexicographic priority order over objectives, given as a permutation
        of ``range(env.n_objectives)`` using 0-based objective indices, **highest
        priority first**. For example, with two objectives ``[0, 1]`` ranks
        objective 0 above objective 1, and ``[1, 0]`` ranks objective 1 above
        objective 0.
    theta : float, optional
        Max-norm convergence threshold. Defaults to ``0.01``, matching the
        original DST configuration.

    Returns
    -------
    policy : dict
        Maps each non-terminal state to its lexicographically best action.
        Terminal states are absent.
    Q : dict
        Maps each ``(state, action)`` pair (over non-terminal states) to its
        converged vector action value. Terminal states are absent.
    """
    priority = list(priority)
    if sorted(priority) != list(range(env.n_objectives)):
        raise ValueError(
            f"priority must be a permutation of range({env.n_objectives}); "
            f"got {priority}"
        )

    gamma = env.gamma
    n_objectives = env.n_objectives

    non_terminal_states = [s for s in env.states() if not env.is_terminal(s)]

    # Vector value function; terminal states are implicitly the zero vector.
    V = {s: np.zeros(n_objectives) for s in non_terminal_states}

    def action_value(s, a):
        # Vector Q(s, a); terminal successors contribute only their reward.
        q = np.zeros(n_objectives)
        for prob, next_state, reward_vector in env.transitions(s, a):
            reward_vector = np.asarray(reward_vector, dtype=float)
            if env.is_terminal(next_state):
                q += prob * reward_vector
            else:
                q += prob * (reward_vector + gamma * V[next_state])
        return q

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        with tqdm(total=len(non_terminal_states), desc=f"Iteration {iteration}") as pbar:
            for s in non_terminal_states:
                v_old = V[s].copy()

                actions = list(env.actions(s))
                q_vectors = np.array([action_value(s, a) for a in actions])
                best = lex_max(q_vectors, priority)
                V[s] = q_vectors[best]

                delta = max(delta, np.max(np.abs(v_old - V[s])))

                pbar.update(1)

        if delta < theta:
            break

    # Final sweep over the converged values: store every vector action value in
    # Q and extract the lexicographic policy from it.
    Q = {}
    policy = {}
    for s in non_terminal_states:
        actions = list(env.actions(s))
        q_vectors = np.zeros((len(actions), n_objectives))
        for i, a in enumerate(actions):
            q_vectors[i] = action_value(s, a)
            Q[(s, a)] = q_vectors[i]
        policy[s] = actions[lex_max(q_vectors, priority)]

    return policy, Q
