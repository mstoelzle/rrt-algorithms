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


def test_goal_distance_function_only_applies_to_goal_queries():
    """distance2goal_fn should override distance_fn only when querying the goal."""
    def axis_distance(a, b):
        return abs(a[0] - b[0])

    def goal_distance(a, b):
        return abs(a[1] - b[1])

    X = _build_space()
    vertices = [(5.0, 2.0), (2.0, 5.0), (8.0, 1.0)]
    query = (4.0, 4.0)
    goal = (9.0, 9.0)

    rrt_custom = RRTBase(X, q=1.0, x_init=query, x_goal=goal,
                         max_samples=50, r=1.0,
                         distance_fn=axis_distance,
                         distance2goal_fn=goal_distance)
    _seed_tree(rrt_custom, vertices)

    non_goal_neighbors = list(rrt_custom.nearby(0, query, len(vertices)))
    expected_non_goal = sorted(vertices, key=lambda v: axis_distance(v, query))
    assert non_goal_neighbors == expected_non_goal

    goal_neighbors = list(rrt_custom.nearby(0, goal, len(vertices), check_goal=True))
    expected_goal = sorted(vertices, key=lambda v: goal_distance(v, goal))
    assert goal_neighbors == expected_goal


def test_goal_distance_without_distance_fn_preserves_default_behavior_elsewhere():
    """distance2goal_fn should fall back to default ordering for non-goal queries when distance_fn is absent."""
    def goal_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    X = _build_space()
    vertices = [(5.0, 5.0), (2.0, 8.0), (8.0, 2.0)]
    query = (1.0, 1.0)
    goal = (9.0, 9.0)

    rrt_default = RRTBase(X, q=1.0, x_init=query, x_goal=goal,
                          max_samples=50, r=1.0)
    rrt_goal_only = RRTBase(X, q=1.0, x_init=query, x_goal=goal,
                            max_samples=50, r=1.0,
                            distance2goal_fn=goal_distance)
    _seed_tree(rrt_default, vertices)
    _seed_tree(rrt_goal_only, vertices)

    expected_non_goal = list(rrt_default.nearby(0, query, len(vertices)))
    actual_non_goal = list(rrt_goal_only.nearby(0, query, len(vertices)))
    assert actual_non_goal == expected_non_goal

    goal_neighbors = list(rrt_goal_only.nearby(0, goal, len(vertices), check_goal=True))
    expected_goal = sorted(vertices, key=lambda v: goal_distance(v, goal))
    assert goal_neighbors == expected_goal
