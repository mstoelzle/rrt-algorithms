import numpy as np

from rrt_algorithms.rrt.rrt_base import RRTBase
from rrt_algorithms.search_space.search_space import SearchSpace


def _build_space():
    return SearchSpace(np.array([(0, 10), (0, 10)]))


def _seed_tree(rrt, vertices):
    for v in vertices:
        rrt.add_vertex(0, v)


def test_default_distance_matches_rtree_nearest():
    """Default behavior should remain identical to the legacy Euclidean/rtree ordering."""
    X = _build_space()
    vertices = [(5.0, 5.0), (2.0, 2.0), (8.0, 8.0)]
    query = (0.0, 0.0)

    rrt_default = RRTBase(X, q=1.0, x_init=query, x_goal=(9.0, 9.0),
                          max_samples=50, r=1.0)
    _seed_tree(rrt_default, vertices)

    nearest_default = list(rrt_default.nearby(0, query, len(vertices)))

    # Recreate the old behavior explicitly via the rtree index
    old_order = list(rrt_default.trees[0].V.nearest(query, num_results=len(vertices), objects="raw"))

    assert nearest_default == old_order


def test_custom_distance_function_overrides_default_ordering():
    """Supplying a custom distance function should dictate the neighbor order."""
    def axis_distance(a, b):
        return abs(a[0] - b[0])

    X = _build_space()
    vertices = [(5.0, 5.0), (2.0, 2.0), (8.0, 8.0)]
    query = (4.0, 4.0)

    rrt_custom = RRTBase(X, q=1.0, x_init=query, x_goal=(9.0, 9.0),
                         max_samples=50, r=1.0, distance_fn=axis_distance)
    _seed_tree(rrt_custom, vertices)

    nearest_custom = list(rrt_custom.nearby(0, query, len(vertices)))

    expected = sorted(vertices, key=lambda v: axis_distance(v, query))
    assert nearest_custom == expected


def test_goal_distance_mode_tracks_best_candidate_and_path():
    """use_goal_distance should record the best vertex within the supplied threshold and return its path."""
    def goal2distance(v):
        return abs(v[0] - 5.0)

    X = _build_space()
    x_init = (0.0, 0.0)
    rrt = RRTBase(X, q=1.0, x_init=x_init, x_goal=None,
                  max_samples=50, r=1.0,
                  goal_distance_estimator=goal2distance,
                  goal_distance_threshold=0.2,
                  use_goal_distance=True)

    rrt.add_vertex(0, x_init)
    rrt.add_edge(0, x_init, None)

    edges = [((1.0, 0.0), x_init),
             ((2.0, 0.0), (1.0, 0.0)),
             ((4.9, 0.0), (2.0, 0.0)),
             ((5.0, 0.0), (4.9, 0.0))]

    for child, parent in edges:
        rrt.add_vertex(0, child)
        rrt.add_edge(0, child, parent)

    assert rrt.best_goal_vertex == (5.0, 0.0)
    assert rrt.best_goal_distance == 0.0

    solved, path = rrt.check_solution()
    assert solved is True
    assert path == [x_init, (1.0, 0.0), (2.0, 0.0), (4.9, 0.0), (5.0, 0.0)]


def test_goal_distance_mode_respects_threshold_and_max_samples():
    """When no vertex meets the goal threshold, the planner should return None after exhausting samples."""
    def goal2distance(v):
        return abs(v[0] - 8.0)

    X = _build_space()
    x_init = (0.0, 0.0)
    rrt = RRTBase(X, q=1.0, x_init=x_init, x_goal=None,
                  max_samples=5, r=1.0,
                  goal_distance_estimator=goal2distance,
                  goal_distance_threshold=0.25,
                  use_goal_distance=True)

    rrt.add_vertex(0, x_init)
    rrt.add_edge(0, x_init, None)

    # Add a vertex that does not satisfy the threshold.
    rrt.add_vertex(0, (2.0, 0.0))
    rrt.add_edge(0, (2.0, 0.0), x_init)

    solved, path = rrt.check_solution()
    assert solved is False
    assert path is None

    rrt.samples_taken = rrt.max_samples
    solved, path = rrt.check_solution()
    assert solved is True
    assert path is None
