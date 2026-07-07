"""Convex Hull Value Iteration (CHVI) for multi-objective MDPs.

CHVI is the multi-objective analogue of Value Iteration: instead of a single
scalar value per state it maintains, for every state-action pair, the convex
hull of the Pareto-optimal value vectors achievable from there (a "Q-ring").
From this single computation an optimal policy can be recovered for *any* linear
scalarisation, without re-solving.

It depends only on the :class:`~morl.core.env_interface.MOEnv` contract and on
the shared hull algebra in :mod:`morl.core.hull_ops`.

The public entry point is :func:`convexhull_vi`.

Policy extraction is intentionally *not* done inside the algorithm. A convex
hull corresponds to a whole set of optimal policies (one per region of weight
space), not to a single greedy action, so :func:`convexhull_vi` returns
``policy = None``; a caller extracts a concrete policy from the returned Q-ring
as a separate step, choosing the weight vector (or other rule) it wants.
"""

import numpy as np
from tqdm import tqdm

from morl.core.env_interface import MOEnv
from morl.core.hull_ops import (
    get_hull,
    translate_hull,
    weighted_minkowski_sum,
    canonical_order,
    hull_max_norm_diff,
)


def convexhull_vi(env: MOEnv, theta=0.01):
    """Run Convex Hull Value Iteration and return ``(None, Q)``.

    Performs hull-valued Bellman backups in place (Gauss-Seidel) over the
    non-terminal states until convergence. For each state-action pair the
    action hull is

        ``Q(s, a) = hull( sum_outcomes prob * (reward + gamma * V(next_state)) )``

    where ``V(next_state)`` is the state hull of the successor (the zero hull for
    a terminal successor, so its reward is carried unchanged). Convergence uses
    the max-norm hull difference over the per-state hulls, after putting vertices
    in canonical order.

    Parameters
    ----------
    env : MOEnv
        The environment. Only the :class:`MOEnv` interface is used.
    theta : float, optional
        Max-norm convergence threshold. Defaults to ``0.01``, matching the
        original DST configuration.

    Returns
    -------
    policy : None
        Always ``None``. A convex hull represents a set of optimal policies
        rather than one greedy action, so policy extraction is left to the
        caller as a separate step (see the module docstring).
    Q_hulls : dict
        The hull-valued Q-ring: maps each ``(state, action)`` pair (over
        non-terminal states) to a numpy array of the Pareto-optimal value-vector
        vertices of its convex hull. Named ``Q_hulls`` to distinguish it from
        VI's scalar ``Q``; LHVI returns a Q-ring of the same shape. Terminal
        states are absent.
    """
    gamma = env.gamma
    n_objectives = env.n_objectives

    non_terminal_states = [s for s in env.states() if not env.is_terminal(s)]

    # V-ring: the convex hull of Pareto-optimal value vectors per state,
    # initialised to the single zero vector.
    V = {s: np.zeros((1, n_objectives)) for s in non_terminal_states}

    # Q-ring: the hull per (state, action).
    Q_hulls = {}
    for s in non_terminal_states:
        for a in env.actions(s):
            Q_hulls[(s, a)] = np.array([np.zeros(n_objectives)])

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        total_hull_vertices = 0
        num_hulls = 0
        with tqdm(total=len(non_terminal_states), desc=f"Iteration {iteration}") as pbar:
            for s in non_terminal_states:
                v_old = V[s].copy()

                for a in env.actions(s):
                    # Build each outcome's reward-translated successor hull.
                    outcomes = []
                    for prob, next_state, reward_vector in env.transitions(s, a):
                        reward_vector = np.asarray(reward_vector, dtype=float)
                        if env.is_terminal(next_state):
                            # Terminal successor: value is the zero hull, so the
                            # contribution is just the reward vector.
                            outcome_hull = np.array([reward_vector])
                        else:
                            outcome_hull = translate_hull(reward_vector, gamma, V[next_state])
                        outcomes.append((prob, outcome_hull))

                    new_hull = weighted_minkowski_sum(outcomes)
                    if len(new_hull) > 1:
                        new_hull = get_hull(new_hull)
                    Q_hulls[(s, a)] = new_hull

                # State hull V(s) = hull over the union of its action hulls.
                all_q_vertices = np.concatenate(
                    [Q_hulls[(s, a)] for a in env.actions(s)], axis=0
                )
                if len(all_q_vertices) > 1:
                    new_V = get_hull(all_q_vertices)
                else:
                    new_V = all_q_vertices
                new_V = canonical_order(new_V)
                V[s] = new_V

                # Track average hull size over the per-state V-ring hulls.
                total_hull_vertices += len(new_V)
                num_hulls += 1

                delta = max(delta, hull_max_norm_diff(new_V, v_old))

                pbar.update(1)

        avg_hull_size = total_hull_vertices / num_hulls if num_hulls > 0 else 0
        print(f"Delta = {delta}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state")

        if delta < theta:
            break

    return None, Q_hulls
