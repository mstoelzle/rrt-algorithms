from rrt_algorithms.rrt.rrt_base import RRTBase


class RRT(RRTBase):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01,
                 distance_fn=None, goal_distance_estimator=None,
                 goal_distance_threshold=None, use_goal_distance=False):
        """
        Template RRT planner
        :param X: Search Space
        :param q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param distance_fn: optional callable used to measure distance between vertices
        :param goal_distance_estimator: optional callable mapping a vertex to an estimated distance-to-goal in task space
        :param goal_distance_threshold: threshold on goal_distance_estimator used when use_goal_distance is True
        :param use_goal_distance: when True, terminate once goal_distance_estimator falls below the provided threshold
        """
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc,
                         distance_fn=distance_fn,
                         goal_distance_estimator=goal_distance_estimator,
                         goal_distance_threshold=goal_distance_threshold,
                         use_goal_distance=use_goal_distance)

    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            x_new, x_nearest = self.new_and_near(0, self.q)

            if x_new is None:
                continue

            # connect shortest valid edge
            self.connect_to_point(0, x_nearest, x_new)

            solution = self.check_solution()
            if solution[0]:
                return solution[1]
