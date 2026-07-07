"""Convex-hull operations shared by the hull-based algorithms (CHVI, LHVI).

These are the multi-objective analogue of scalar ``max``/``+`` in a Bellman
backup: instead of a single value per state we carry the convex hull of the
Pareto-optimal value vectors. The operations here are exactly those the
Convex Hull and Lexicographic Hull value-iteration algorithms need — nothing
more:

* :func:`get_hull` / :func:`non_dominated` — convex-hull computation with
  Pareto (upper-hull) filtering;
* :func:`translate_hull` — the reward + discount translation/scaling of a hull;
* :func:`sum_hulls` / :func:`weighted_minkowski_sum` — the probability-weighted
  Minkowski sum that combines the successor hulls of a (possibly stochastic)
  transition;
* :func:`canonical_order` — the ``np.lexsort`` vertex ordering applied before
  comparing hulls;
* :func:`hull_max_norm_diff` — the max-norm hull difference used for
  convergence.

The hull algebra follows Barrett & Narayanan, *Learning All Optimal Policies
with Multiple Criteria* (2008). Two subtle but important behaviours are carried
over deliberately: hulls are ``.copy()``-ed before in-place scaling so that
successor hulls are never mutated by aliasing, and vertices are put in a
canonical ``lexsort`` order before comparison so that vertex-order
nondeterminism does not cause spurious non-convergence.
"""

from scipy.spatial import ConvexHull
import numpy as np


def non_dominated(solutions):
    """Return the Pareto-optimal (non-dominated) subset of ``solutions``.

    A point is dominated if another point is greater-or-equal in every
    objective (objectives are maximised). Keeps the upper Pareto frontier and
    always retains at least the point itself, so the result is never empty.

    :param solutions: numpy array of shape ``(n_points, n_objectives)``
    :return: numpy array of the non-dominated rows of ``solutions``
    """
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove points dominated by c (this also removes c itself)...
            dominated = (np.asarray(solutions[is_efficient]) <= c).all(axis=1)
            is_efficient[is_efficient] = np.invert(dominated)
            # ...then keep c, otherwise we would drop the point we compared against.
            is_efficient[i] = 1
    return solutions[is_efficient]


def get_hull(points):
    """Compute the convex hull of ``points``, keeping only Pareto-optimal vertices.

    Points are first reduced to their non-dominated set, then the convex hull is
    taken and its vertices reduced to non-dominated again. If there are too few
    points for a hull (``scipy`` raises), the non-dominated points are returned
    as-is.

    :param points: list / numpy array of vectors
    :return: numpy array of the hull's Pareto-optimal vertices
    """
    points = non_dominated(np.array(points))

    try:
        hull = ConvexHull(points)
        hull_points = [points[vertex] for vertex in hull.vertices]
        vertices = non_dominated(np.array(hull_points))
    except Exception:
        # Fewer points than needed for a hull (degenerate): keep them as-is.
        vertices = points

    return np.array(vertices)


def translate_hull(point, gamma, hull):
    """Scale ``hull`` by ``gamma`` and translate it by ``point``.

    Computes ``point + gamma * hull`` (Barrett & Narayanan, definition 1). Used
    to turn a successor-state hull into the contribution of a transition:
    ``reward_vector + gamma * V(next_state)``. An empty ``hull`` degenerates to
    the single point ``[point]``.

    :param point: numpy array, the reward vector added to every vertex
    :param gamma: real discount factor multiplying the hull
    :param hull: numpy array of vertices (not mutated by the caller if copied)
    :return: numpy array of the translated/scaled vertices
    """
    if len(hull) == 0:
        hull = np.array([point])
    else:
        hull = np.multiply(hull, gamma, casting="unsafe")
        if len(point) > 0:
            hull = np.add(hull, point, casting="unsafe")
    return hull


def sum_hulls(hull_1, hull_2):
    """Minkowski sum of two hulls (Barrett & Narayanan, definition 2).

    Returns the convex hull of ``{a + b : a in hull_1, b in hull_2}``. Each
    successor hull is ``.copy()``-ed before use so the inputs are never mutated
    by aliasing. An empty operand acts as the identity.

    :param hull_1: numpy array of vertices
    :param hull_2: numpy array of vertices
    :return: numpy array of the summed hull's vertices
    """
    if len(hull_1) == 0:
        return hull_2
    elif len(hull_2) == 0:
        return hull_1

    new_points = None
    for i in range(len(hull_1)):
        translated = translate_hull(hull_1[i].copy(), 1, hull_2.copy())
        if new_points is None:
            new_points = translated
        else:
            new_points = np.concatenate((new_points, translated), axis=0)

    return get_hull(new_points)


def weighted_minkowski_sum(outcomes):
    """Probability-weighted Minkowski sum of successor hulls.

    Combines the outcomes of a (possibly stochastic) transition into a single
    hull: each outcome's hull is scaled by its probability and the scaled hulls
    are Minkowski-summed. For a deterministic transition (a single outcome with
    probability ``1``) this simply returns that outcome's hull unchanged, which
    keeps the deterministic path free of unnecessary hull recomputation.

    :param outcomes: list of ``(prob, hull)`` pairs, where ``hull`` is the
        already reward-translated successor hull for that outcome
    :return: numpy array, the combined hull (not yet re-hulled by ``get_hull``;
        the caller decides whether a final :func:`get_hull` is needed)
    """
    scaled = [prob * np.asarray(hull) for prob, hull in outcomes]

    if len(scaled) == 1:
        return scaled[0]

    combined = scaled[0]
    for hull in scaled[1:]:
        combined = sum_hulls(combined, hull)
    return combined


def canonical_order(hull):
    """Return ``hull`` with its vertices in a canonical ``np.lexsort`` order.

    Convex-hull computation does not fix the order of the returned vertices, so
    two runs can produce the same vertex set in different orders. Sorting into a
    canonical order before comparison prevents that order nondeterminism from
    being read as a change (and thus causing spurious non-convergence).

    :param hull: numpy array of vertices
    :return: numpy array with the same vertices in canonical order
    """
    hull = np.asarray(hull)
    return hull[np.lexsort(hull.T)]


def hull_max_norm_diff(new_hull, old_hull):
    """Max-norm difference between two hulls, used for convergence testing.

    If the two hulls have the same shape, returns the largest absolute
    component-wise difference between them (both should already be in
    :func:`canonical_order`). If the shapes differ, the hull changed size this
    sweep and the difference is infinite, forcing another iteration.

    :param new_hull: numpy array of vertices (canonically ordered)
    :param old_hull: numpy array of vertices (canonically ordered)
    :return: float, the max-norm difference (``inf`` on shape mismatch)
    """
    new_hull = np.asarray(new_hull)
    old_hull = np.asarray(old_hull)
    if new_hull.shape == old_hull.shape:
        return float(np.max(np.abs(new_hull - old_hull)))
    return float("inf")
