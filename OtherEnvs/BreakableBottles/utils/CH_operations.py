from scipy.spatial import ConvexHull
import numpy as np


def non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            dominated = (np.asarray(solutions[is_efficient]) <= c).all(axis=1)
            is_efficient[is_efficient] = np.invert(dominated)
            # keep the point itself, otherwise we would get an empty list
            is_efficient[i] = 1

    return solutions[is_efficient]


def get_hull(points):
    """
    From a set of points, computes its associated convex hull.
    Handles degenerate configurations where points don't span the full ambient dimension
    by projecting to the intrinsic subspace before calling ConvexHull.

    :param points: a list / numpy array of vectors
    :return: a numpy array of vectors
    """

    # pareto pre filter
    points = non_dominated(np.array(points))
    n = len(points)

    if n <= 1:
        return np.array(points)

    # Determine intrinsic rank via SVD to handle low-rank / degenerate point sets
    centered = points - points.mean(axis=0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    tol = np.finfo(float).eps * max(centered.shape) * (s[0] if s.size > 0 else 1.0)
    rank = int(np.sum(s > tol))

    if rank == 0:
        # All points are numerically identical
        return np.array([points[0]])

    if rank == 1:
        # Points are collinear — hull is just the two endpoints
        proj = centered @ Vt[0]
        vertices = points[[np.argmin(proj), np.argmax(proj)]]
        return non_dominated(np.unique(vertices, axis=0))

    # Project to intrinsic subspace and compute hull there;
    # vertex indices map back to the original (unrotated) points unchanged
    proj = centered @ Vt[:rank].T
    try:
        hull = ConvexHull(proj)
        return non_dominated(points[hull.vertices])
    except Exception:
        return non_dominated(points)
    


def translate_hull(point, gamma, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)
    Translation and scaling operation of convex hulls (definition 1 of the paper).
    :param point: a numpy array
    :param gamma: a real number
    :param hull: a set of points, they need to be numpy arrays
    :return: the new convex hull, a new set of points
    """
    if len(hull) == 0:
        hull = np.array([point])
        # hull = [point]
    else:
       hull = np.multiply(hull, gamma, casting="unsafe")
       if len(point) > 0:
            hull = np.add(hull, point, casting="unsafe")
    return hull




def sum_hulls(hull_1, hull_2):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Sum operation of convex hulls (definition 2 of the paper)

    :param hull_1: a set of 2-D points, they need to be numpy arrays
    :param hull_2: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """
    if len(hull_1) == 0:
        return hull_2
    elif len(hull_2) == 0:
        return hull_1

    new_points = None

    for i in range(len(hull_1)):
        if new_points is None:
            new_points = translate_hull(hull_1[i].copy(), 1,  hull_2.copy())
        else:
            new_points = np.concatenate((new_points, translate_hull(hull_1[i].copy(), 1, hull_2.copy())), axis=0)

    return get_hull(new_points)


def max_q_value(weight, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Extraction of the Q-value (definition 3 of the paper)

    :param weight: a weight vector, can be simply a list of floats
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: a real number, the best Q-value of the hull for the given weight vector
    """
    scalarised = []

    for i in range(len(hull)):
        f = np.dot(weight,hull[i])
        scalarised.append(f)

    scalarised = np.array(scalarised)

    return np.max(scalarised)


if __name__ == "__main__":

    print("We provide some examples of convex hull computations.")

    points = np.random.rand(42, 2)   # 15 random points in 2-D

    puntitos = -1*np.array([[0, 0, 4], [0, 5, 3], [1, 7, 0], [2, 1, 4], [3, 4, 5], [4, 2, 3], [4, 4, 6], [4, 6, 7], [5, 0, 2], [6, 4, 1], [6, 5, 1], [6, 7, 0], [7, 4, 3]])
    print(puntitos)

    v_function = [[3., 2.],
                  [5., -1.5],
                  [-20., 4.]]

    v_functionE = [[-3., 0.],
                   [4., 2.],
                   [5., 3.]]

    v_function = np.array(v_function)
    v_functionE = np.array(v_functionE)

    vertices = get_hull(puntitos)

    print("resulting vertices")
    print(vertices)
    import matplotlib.pyplot as plt

    plt.plot(vertices[:,0], vertices[:,1], 'k-')
    #max_q_value([1.0,0.4],vertices)
    plt.show()