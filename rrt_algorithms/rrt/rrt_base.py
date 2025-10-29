import numpy as np
import random
from typing import Callable, Optional

from rrt_algorithms.rrt.heuristics import path_cost
from rrt_algorithms.rrt.tree import Tree
from rrt_algorithms.utilities.geometry import steer


class RRTBase(object):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01,
                 distance_fn: Optional[Callable] = None, 
                 goal_distance_estimator: Optional[Callable] = None,
                 goal_distance_threshold: Optional[float] = None, 
                 use_goal_distance: bool = False
                 ):
        """
        Template RRT planner
        :param X: Search Space
        :param q: length of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param distance_fn: optional callable used to measure distance between vertices
        :param goal_distance_estimator: optional callable mapping a vertex to a goal-distance estimate in task space
        :param goal_distance_threshold: maximum acceptable distance returned by goal_distance_estimator for a vertex to be considered a goal candidate
        :param use_goal_distance: when True, terminate once a vertex falls within goal_distance_threshold instead of connecting to x_goal
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.q = q
        self.r = r
        self.prc = prc
        self.x_init = x_init
        self.x_goal = x_goal
        self.distance_fn = distance_fn
        self.goal_distance_estimator = goal_distance_estimator
        self.goal_distance_threshold = goal_distance_threshold
        self.use_goal_distance = use_goal_distance
        self.best_goal_vertex = None
        self.best_goal_distance = None
        self.best_goal_cost = None
        if self.use_goal_distance:
            if self.goal_distance_estimator is None:
                raise ValueError("goal_distance_estimator must be provided when use_goal_distance is True.")
            if self.goal_distance_threshold is None:
                raise ValueError("goal_distance_threshold must be provided when use_goal_distance is True.")

        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(Tree(self.X))

    def _register_vertex(self, tree, v):
        """
        Internal helper to register a vertex with the tree's spatial index and cache.
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        :return: bool, True if vertex was newly added
        """
        if self.trees[tree].V.count(v) != 0:
            return False
        self.trees[tree].points.append(v)
        self.trees[tree].V.insert(0, v + v, v)
        self.trees[tree].V_count += 1
        return True

    def add_vertex(self, tree, v):
        """
        Add vertex to corresponding tree
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        """
        if self._register_vertex(tree, v):
            self.samples_taken += 1  # increment number of samples taken

    def add_edge(self, tree, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
        self.trees[tree].E[child] = parent
        if self.use_goal_distance and tree == 0:
            self._update_goal_candidate(tree, child)

    def nearby(self, tree, x, n):
        """
        Return nearby vertices
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :param n: int, max number of neighbors to return
        :return: list of nearby vertices
        """
        metric = self.distance_fn

        if metric is None:
            return self.trees[tree].V.nearest(x, num_results=n, objects="raw")

        vertices = self.trees[tree].points
        if n is None or n <= 0:
            return iter(sorted(vertices, key=lambda v: metric(v, x)))

        ordered = sorted(vertices, key=lambda v: metric(v, x))
        return iter(ordered[:n])

    def get_nearest(self, tree, x):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """
        return next(self.nearby(tree, x, 1))

    def new_and_near(self, tree, q):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
        """
        x_rand = self.X.sample_free()
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q))
        # check if new point is in X_free and not already in V
        if not self.trees[0].V.count(x_new) == 0 or not self.X.obstacle_free(x_new):
            return None, None
        self.samples_taken += 1
        return x_new, x_nearest

    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph
        :param tree: rtree of all Vertices
        :return: True if can be added, False otherwise
        """
        if self.use_goal_distance:
            return self.best_goal_vertex is not None
        x_nearest = self.get_nearest(tree, self.x_goal)
        if self.x_goal in self.trees[tree].E and x_nearest in self.trees[tree].E[self.x_goal]:
            # tree is already connected to goal using nearest vertex
            return True
        # check if obstacle-free
        if self.X.collision_free(x_nearest, self.x_goal, self.r):
            return True
        return False

    def get_path(self):
        """
        Return path through tree from start to goal
        :return: path if possible, None otherwise
        """
        if self.use_goal_distance:
            if self.best_goal_vertex is not None:
                print("Found a goal-distance compliant vertex, extracting path")
                return self.reconstruct_path(0, self.x_init, self.best_goal_vertex)
            print("Could not find a vertex within the goal-distance threshold")
            return None
        if self.can_connect_to_goal(0):
            print("Can connect to goal")
            self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        print("Could not connect to goal")
        return None

    def connect_to_goal(self, tree):
        """
        Connect x_goal to graph
        (does not check if this should be possible, for that use: can_connect_to_goal)
        :param tree: rtree of all Vertices
        """
        if self.use_goal_distance:
            raise RuntimeError("connect_to_goal is not valid when use_goal_distance is enabled.")
        x_nearest = self.get_nearest(tree, self.x_goal)
        self._register_vertex(tree, self.x_goal)
        self.trees[tree].E[self.x_goal] = x_nearest

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        path = [x_goal]
        current = x_goal
        if x_init == x_goal:
            return path
        while not self.trees[tree].E[current] == x_init:
            path.append(self.trees[tree].E[current])
            current = self.trees[tree].E[current]
        path.append(x_init)
        path.reverse()
        return path

    def check_solution(self):
        # probabilistically check if solution found
        if self.use_goal_distance:
            if self.best_goal_vertex is not None:
                return True, self.reconstruct_path(0, self.x_init, self.best_goal_vertex)
            if self.samples_taken >= self.max_samples:
                return True, self.get_path()
            return False, None
        if self.prc and random.random() < self.prc:
            print("Checking if can connect to goal at", str(self.samples_taken), "samples")
            path = self.get_path()
            if path is not None:
                return True, path
        # check if can connect to goal after generating max_samples
        if self.samples_taken >= self.max_samples:
            return True, self.get_path()
        return False, None

    def bound_point(self, point):
        # if point is out-of-bounds, set to bound
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)

    def _goal_distance(self, vertex):
        if self.goal_distance_estimator is not None:
            return self.goal_distance_estimator(vertex)
        if self.distance_fn is not None:
            return self.distance_fn(vertex, self.x_goal)
        return np.linalg.norm(np.array(vertex) - np.array(self.x_goal))

    def _update_goal_candidate(self, tree, vertex):
        if self.goal_distance_estimator is None:
            return
        goal_distance = self._goal_distance(vertex)
        if goal_distance is None or goal_distance > self.goal_distance_threshold:
            return
        cost = path_cost(self.trees[tree].E, self.x_init, vertex)
        if (self.best_goal_vertex is None or
                goal_distance < self.best_goal_distance or
                (goal_distance == self.best_goal_distance and cost < self.best_goal_cost)):
            self.best_goal_vertex = vertex
            self.best_goal_cost = cost
            self.best_goal_distance = goal_distance
