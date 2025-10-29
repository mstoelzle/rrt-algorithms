# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot


X_DIMENSIONS = np.array([(0, 100), (0, 100), (0, 100)])
OBSTACLES = np.array(
    [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
     (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])

x_init = (0, 0, 0)
# In many soft-robot settings the configuration-space goal is unknown; we only get a
# target pose in task space via sensing. We keep that target only for plotting.
task_space_goal = np.array((95.0, 95.0, 95.0))


def task_space_distance(config):
    """
    Example task-space distance estimator. Replace this with the distance-to-goal
    signal available in your problem (e.g., vision-based end-effector tracking).
    """
    return np.linalg.norm(np.asarray(config) - task_space_goal)


q = 16
r = 1
max_samples = 4096
rewire_count = 32
prc = 0.1
goal_tolerance = 5.0  # Stop once we are within 5 units in task space.

X = SearchSpace(X_DIMENSIONS, OBSTACLES)

rrt = RRTStar(
    X, q, x_init, x_goal=None, max_samples=max_samples, r=r, prc=prc,
    rewire_count=rewire_count,
    goal_distance_estimator=task_space_distance,
    goal_distance_threshold=goal_tolerance,
    use_goal_distance=True
)
path = rrt.rrt_star()

plot = Plot("rrt_star_3d_goal_distance")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, OBSTACLES)
plot.plot_start(X, x_init)
plot.plot_goal(X, tuple(task_space_goal))
plot.draw(auto_open=True)
