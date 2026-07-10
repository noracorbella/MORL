"""Convex Hull Value Iteration (CHVI) for MOMDPs.

This algorithm depends only on the :class:`~morl.core.env_interface.MOEnv` 
contract and the hull operations in :mod:`morl.core.hull_ops`.

Policy extraction is *not* done inside the algorithm, so :func:`convexhull_vi` 
returns ``policy = None``. A caller extracts a concrete policy from the returned 
Q-ring as a separate step.
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
    """
    Run Convex Hull Value Iteration and return ``(None, Q)``.

    For each state-action pair the action hull is

        ``Q(s, a) = hull( sum_outcomes prob * (reward + gamma * V(next_state)) )``

    where ``V(next_state)`` is the state hull of the successor (the zero for a 
    terminal successor). Convergence uses,the max-norm hull difference over the 
    per-state hulls, after putting vertices in canonical order.

    Parameters
    ----------
    env : MOEnv
        The environment. Only the :class:`MOEnv` interface is used.
    theta : float, optional
        Max-norm convergence threshold. Default is ``0.01``.

    Returns
    -------
    policy : None
        Policy extraction is left as a separate step.
    Q_hulls : dict
        map of each ``(state, action)`` pair (of non-terminal states) to an 
        array of the optimal value-vector vertices of its convex hull. 
    """
    gamma = env.gamma
    n_objectives = env.n_objectives

    non_terminal_states = [s for s in env.states() if not env.is_terminal(s)]

    # V is the convex hull of optimal value vectors per state,
    # initialised to the single zero vector.
    V = {s: np.zeros((1, n_objectives)) for s in non_terminal_states}

    # Q_hulls is the hull per (state, action).
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
                    # Build each outcome's reward translated successor hull.
                    outcomes = []
                    for prob, next_state, reward_vector in env.transitions(s, a):
                        reward_vector = np.asarray(reward_vector, dtype=float)
                        if env.is_terminal(next_state):
                            # Terminal state: contribution is just the reward vector.
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

                # Track average hull size over the per-state V hulls.
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
