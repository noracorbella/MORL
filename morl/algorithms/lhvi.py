"""
Lexicographic Hull Value Iteration (LHVI) for MOMDPs.

This is the hull-valued lexicographic algorithm. It depends only on the 
:class:`~morl.core.env_interface.MOEnv`, the hull algebra in 
:mod:`morl.core.hull_ops`, and the :func:`lex_max` from :mod:`morl.algorithms.lexvi`.
"""

from itertools import permutations

import numpy as np
from tqdm import tqdm

from morl.core.env_interface import MOEnv
from morl.core.hull_ops import (
    translate_hull,
    weighted_minkowski_sum,
    canonical_order,
    hull_max_norm_diff,
)
from morl.algorithms.lexvi import lex_max


def lex_hull(vertices, n_objectives, tol=1e-9):
    """
    Return the lexicographic hull of ``vertices``.

    Keeps only the rows of ``vertices`` that are the lexicographic maximiser
    (:func:`lex_max`) for at least one priority order over the objectives. 

    Parameters
    ----------
    vertices : numpy array of shape ``(n_points, n_objectives)``
        The candidate value vectors.
    n_objectives : int
        The number of objectives. The priority orders enumerated are the
        permutations of ``range(n_objectives)``.
    tol : float, optional
        Tolerance passed to :func:`lex_max` for tie comparisons.

    Returns
    -------
    numpy array
        The subset of ``vertices`` that survive lexicographic pruning.
    """
    optimal = set()
    for order in permutations(range(n_objectives)):
        optimal.add(lex_max(vertices, list(order), tol=tol))
    return vertices[sorted(optimal)]


def lexicographic_hull_vi(env: MOEnv, theta=0.01):
    """
    Run Lexicographic Hull Value Iteration and return ``(None, Q_hulls)``.

    Parameters
    ----------
    env : MOEnv
        Environment of :class:`MOEnv`.
    theta : float, optional
        Max-norm convergence threshold. Defaults is ``0.01``.

    Returns
    -------
    policy : None
        Always ``None``. Extract a concrete policy for a chosen priority order
        with :func:`extract_lex_policy`.
    Q_hulls : dict
        The lexicographic Q: maps each ``(state, action)`` pair (over
        non-terminal states) to a numpy array of its lexicographic-hull vertices.
    """
    gamma = env.gamma
    n_objectives = env.n_objectives

    non_terminal_states = [s for s in env.states() if not env.is_terminal(s)]

    # V hull per state, initialised to the single zero vector.
    V = {s: np.zeros((1, n_objectives)) for s in non_terminal_states}

    # Q hull per (state, action).
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
                    # Build each outcome's reward-translated successor hull
                    outcomes = []
                    for prob, next_state, reward_vector in env.transitions(s, a):
                        reward_vector = np.asarray(reward_vector, dtype=float)
                        if env.is_terminal(next_state):
                            outcome_hull = np.array([reward_vector])
                        else:
                            outcome_hull = translate_hull(reward_vector, gamma, V[next_state])
                        outcomes.append((prob, outcome_hull))

                    new_hull = weighted_minkowski_sum(outcomes)
                    if len(new_hull) > 1:
                        new_hull = lex_hull(new_hull, n_objectives)
                    Q_hulls[(s, a)] = new_hull

                # State hull V(s) = lexicographic hull over the union of action hulls.
                all_q_vertices = np.concatenate(
                    [Q_hulls[(s, a)] for a in env.actions(s)], axis=0
                )
                if len(all_q_vertices) > 1:
                    new_V = lex_hull(all_q_vertices, n_objectives)
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
        print(f"Delta = {round(delta, 3)}, Theta = {theta}")
        print(f"Average hull size: {avg_hull_size:.2f} vertices per state")

        if delta < theta:
            break

    return None, Q_hulls


def extract_lex_policy(Q_hulls, env: MOEnv, priority):
    """
    Extract a single policy from a lexicographic Q hull for one priority order.

    For each non-terminal state and each action, takes the action's
    lexicographically best hull vertex (:func:`lex_max` under ``priority``), then
    chooses the action whose best vertex is lexicographically best overall. This
    is the per-priority extraction.

    Parameters
    ----------
    Q_hulls : dict
        Lexicographic Q hull as returned by :func:`lexicographic_hull_vi`,
        keyed by ``(state, action)``.
    env : MOEnv
        Environment (used for its state/action sets and ``n_objectives``).
    priority : sequence of int
        Lexicographic priority order, a permutation of
        ``range(env.n_objectives)``, 0-based, highest priority first.

    Returns
    -------
    dict
        Map of each non-terminal state to its chosen action. 
    """
    priority = list(priority)
    if sorted(priority) != list(range(env.n_objectives)):
        raise ValueError(
            f"priority must be a permutation of range({env.n_objectives}); "
            f"got {priority}"
        )

    policy = {}
    for s in env.states():
        if env.is_terminal(s):
            continue
        actions = list(env.actions(s))
        best_vectors = np.array([
            Q_hulls[(s, a)][lex_max(Q_hulls[(s, a)], priority)] for a in actions
        ])
        policy[s] = actions[lex_max(best_vectors, priority)]
    return policy
