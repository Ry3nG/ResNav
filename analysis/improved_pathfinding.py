#!/usr/bin/env python3
"""Improved pathfinding algorithm that considers robot dynamics and turning radius.

This module provides enhanced pathfinding capabilities that account for:
- Robot actual size and shape
- Minimum turning radius
- Dynamic constraints
- Multiple path options
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import math


class ImprovedPathfinder:
    """Enhanced pathfinding with robot dynamics consideration."""

    def __init__(self, robot_radius: float = 0.25, min_turning_radius: float = 0.5,
                 resolution: float = 0.2, safety_margin: float = 0.1):
        """Initialize the pathfinder with robot constraints.

        Args:
            robot_radius: Robot radius in meters
            min_turning_radius: Minimum turning radius in meters
            resolution: Grid resolution in meters per cell
            safety_margin: Additional safety margin in meters
        """
        self.robot_radius = robot_radius
        self.min_turning_radius = min_turning_radius
        self.resolution = resolution
        self.safety_margin = safety_margin
        self.total_radius = robot_radius + safety_margin

    def find_path(self, grid: np.ndarray, start_pose: Tuple[float, float, float],
                  goal_xy: Tuple[float, float]) -> Tuple[bool, Optional[List[Tuple[float, float]]], Dict]:
        """Find a feasible path considering robot dynamics.

        Args:
            grid: Occupancy grid (True=occupied)
            start_pose: (x, y, theta) in meters and radians
            goal_xy: (x, y) in meters

        Returns:
            (success, path, info)
        """
        start_x, start_y, start_theta = start_pose
        goal_x, goal_y = goal_xy

        # Convert to grid coordinates
        start_i = int(np.floor(start_y / self.resolution))
        start_j = int(np.floor(start_x / self.resolution))
        goal_i = int(np.floor(goal_y / self.resolution))
        goal_j = int(np.floor(goal_x / self.resolution))

        H, W = grid.shape

        # Check bounds
        if (start_i < 0 or start_i >= H or start_j < 0 or start_j >= W or
            goal_i < 0 or goal_i >= H or goal_j < 0 or goal_j >= W):
            return False, None, {"error": "Start or goal out of bounds"}

        # Check if start/goal are in obstacles
        if grid[start_i, start_j] or grid[goal_i, goal_j]:
            return False, None, {"error": "Start or goal in obstacle"}

        # Try multiple pathfinding strategies
        strategies = [
            self._astar_with_dynamics,
            self._rrt_like_search,
            self._corridor_following
        ]

        for strategy in strategies:
            success, path, info = strategy(grid, start_pose, goal_xy)
            if success:
                return success, path, info

        return False, None, {"error": "No feasible path found with any strategy"}

    def _astar_with_dynamics(self, grid: np.ndarray, start_pose: Tuple[float, float, float],
                           goal_xy: Tuple[float, float]) -> Tuple[bool, Optional[List[Tuple[float, float]]], Dict]:
        """A* with consideration of robot dynamics and turning constraints."""

        start_x, start_y, start_theta = start_pose
        goal_x, goal_y = goal_xy

        # Convert to grid coordinates
        start_i = int(np.floor(start_y / self.resolution))
        start_j = int(np.floor(start_x / self.resolution))
        goal_i = int(np.floor(goal_y / self.resolution))
        goal_j = int(np.floor(goal_x / self.resolution))

        H, W = grid.shape

        # Priority queue: (f_cost, g_cost, (i, j, theta), parent)
        from heapq import heappush, heappop
        open_set = [(0, 0, (start_i, start_j, start_theta), None)]
        closed_set = set()
        g_costs = {}
        parents = {}

        # 8-connected directions with orientation changes
        directions = [
            (-1, -1, 0), (-1, 0, 0), (-1, 1, 0),
            (0, -1, 0), (0, 1, 0),
            (1, -1, 0), (1, 0, 0), (1, 1, 0),
            # Add orientation changes
            (0, 0, np.pi/4), (0, 0, -np.pi/4), (0, 0, np.pi/2), (0, 0, -np.pi/2)
        ]

        while open_set:
            f_cost, g_cost, (i, j, theta), parent = heappop(open_set)

            if (i, j) == (goal_i, goal_j):
                # Reconstruct path
                path = self._reconstruct_path(parents, (i, j, theta), start_pose, goal_xy)
                return True, path, {"method": "astar_with_dynamics", "cost": g_cost}

            if (i, j, theta) in closed_set:
                continue

            closed_set.add((i, j, theta))

            for di, dj, dtheta in directions:
                ni, nj = i + di, j + dj
                ntheta = (theta + dtheta) % (2 * np.pi)

                if (0 <= ni < H and 0 <= nj < W and
                    not grid[ni, nj] and (ni, nj, ntheta) not in closed_set):

                    # Check if path from current to next position is collision-free
                    if self._is_path_clear(grid, i, j, ni, nj, theta, ntheta):
                        tentative_g = g_cost + self._calculate_cost(i, j, ni, nj, theta, ntheta)

                        if (ni, nj, ntheta) not in g_costs or tentative_g < g_costs[(ni, nj, ntheta)]:
                            g_costs[(ni, nj, ntheta)] = tentative_g
                            h_cost = self._heuristic(ni, nj, goal_i, goal_j)
                            f_cost = tentative_g + h_cost

                            parents[(ni, nj, ntheta)] = (i, j, theta)
                            heappush(open_set, (f_cost, tentative_g, (ni, nj, ntheta), (i, j, theta)))

        return False, None, {"error": "A* with dynamics failed"}

    def _rrt_like_search(self, grid: np.ndarray, start_pose: Tuple[float, float, float],
                        goal_xy: Tuple[float, float]) -> Tuple[bool, Optional[List[Tuple[float, float]]], Dict]:
        """RRT-like search for more flexible pathfinding."""

        start_x, start_y, start_theta = start_pose
        goal_x, goal_y = goal_xy

        # Convert to grid coordinates
        start_i = int(np.floor(start_y / self.resolution))
        start_j = int(np.floor(start_x / self.resolution))
        goal_i = int(np.floor(goal_y / self.resolution))
        goal_j = int(np.floor(goal_x / self.resolution))

        H, W = grid.shape

        # Tree: node -> parent
        tree = {(start_i, start_j, start_theta): None}
        max_iterations = 1000

        for _ in range(max_iterations):
            # Sample random configuration
            if np.random.random() < 0.1:  # 10% chance to sample goal
                target_i, target_j = goal_i, goal_j
            else:
                target_i = np.random.randint(0, H)
                target_j = np.random.randint(0, W)

            # Find nearest node in tree
            nearest_node = self._find_nearest_node(tree, target_i, target_j)
            if nearest_node is None:
                continue

            # Try to extend towards target
            new_node = self._extend_towards(grid, nearest_node, target_i, target_j)
            if new_node is not None:
                tree[new_node] = nearest_node

                # Check if we reached the goal
                if abs(new_node[0] - goal_i) <= 1 and abs(new_node[1] - goal_j) <= 1:
                    path = self._reconstruct_rrt_path(tree, new_node, start_pose, goal_xy)
                    return True, path, {"method": "rrt_like", "iterations": _}

        return False, None, {"error": "RRT-like search failed"}

    def _corridor_following(self, grid: np.ndarray, start_pose: Tuple[float, float, float],
                           goal_xy: Tuple[float, float]) -> Tuple[bool, Optional[List[Tuple[float, float]]], Dict]:
        """Corridor-following strategy for narrow passages."""

        start_x, start_y, start_theta = start_pose
        goal_x, goal_y = goal_xy

        # Convert to grid coordinates
        start_i = int(np.floor(start_y / self.resolution))
        start_j = int(np.floor(start_x / self.resolution))
        goal_i = int(np.floor(goal_y / self.resolution))
        goal_j = int(np.floor(goal_x / self.resolution))

        H, W = grid.shape

        # Find corridor centerline
        corridor_center = self._find_corridor_center(grid, start_i, start_j, goal_i, goal_j)
        if corridor_center is None:
            return False, None, {"error": "No corridor center found"}

        # Follow corridor centerline
        path = self._follow_corridor_centerline(grid, corridor_center, start_pose, goal_xy)
        if path is not None:
            return True, path, {"method": "corridor_following"}

        return False, None, {"error": "Corridor following failed"}

    def _is_path_clear(self, grid: np.ndarray, i1: int, j1: int, i2: int, j2: int,
                      theta1: float, theta2: float) -> bool:
        """Check if path between two points is collision-free."""

        # Simple line-of-sight check
        steps = max(abs(i2 - i1), abs(j2 - j1))
        if steps == 0:
            return True

        for t in np.linspace(0, 1, steps + 1):
            i = int(i1 + t * (i2 - i1))
            j = int(j1 + t * (j2 - j1))

            if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
                return False

            if grid[i, j]:
                return False

        return True

    def _calculate_cost(self, i1: int, j1: int, i2: int, j2: int,
                       theta1: float, theta2: float) -> float:
        """Calculate cost of moving from one configuration to another."""

        # Distance cost
        distance = np.sqrt((i2 - i1)**2 + (j2 - j1)**2) * self.resolution

        # Orientation change cost
        theta_diff = abs(theta2 - theta1)
        theta_diff = min(theta_diff, 2 * np.pi - theta_diff)
        orientation_cost = theta_diff * self.min_turning_radius

        return distance + orientation_cost

    def _heuristic(self, i: int, j: int, goal_i: int, goal_j: int) -> float:
        """Heuristic function for A*."""
        return np.sqrt((i - goal_i)**2 + (j - goal_j)**2) * self.resolution

    def _find_nearest_node(self, tree: Dict, target_i: int, target_j: int) -> Optional[Tuple[int, int, float]]:
        """Find nearest node in tree to target."""
        if not tree:
            return None

        min_dist = float('inf')
        nearest = None

        for (i, j, theta) in tree.keys():
            dist = np.sqrt((i - target_i)**2 + (j - target_j)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (i, j, theta)

        return nearest

    def _extend_towards(self, grid: np.ndarray, from_node: Tuple[int, int, float],
                       target_i: int, target_j: int) -> Optional[Tuple[int, int, float]]:
        """Extend tree towards target."""

        i, j, theta = from_node

        # Calculate direction to target
        di = target_i - i
        dj = target_j - j
        dist = np.sqrt(di**2 + dj**2)

        if dist == 0:
            return None

        # Normalize and scale by step size
        step_size = 2  # Grid cells
        di = int(di / dist * step_size)
        dj = int(dj / dist * step_size)

        ni, nj = i + di, j + dj

        # Check bounds and collision
        if (ni < 0 or ni >= grid.shape[0] or nj < 0 or nj >= grid.shape[1] or
            grid[ni, nj]):
            return None

        # Calculate new orientation
        new_theta = np.arctan2(dj, di)

        return (ni, nj, new_theta)

    def _find_corridor_center(self, grid: np.ndarray, start_i: int, start_j: int,
                             goal_i: int, goal_j: int) -> Optional[List[Tuple[int, int]]]:
        """Find corridor centerline between start and goal."""

        # Simple straight line for now
        steps = max(abs(goal_i - start_i), abs(goal_j - start_j))
        if steps == 0:
            return [(start_i, start_j)]

        centerline = []
        for t in np.linspace(0, 1, steps + 1):
            i = int(start_i + t * (goal_i - start_i))
            j = int(start_j + t * (goal_j - start_j))
            centerline.append((i, j))

        return centerline

    def _follow_corridor_centerline(self, grid: np.ndarray, centerline: List[Tuple[int, int]],
                                   start_pose: Tuple[float, float, float],
                                   goal_xy: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Follow corridor centerline to create path."""

        path = []
        for i, j in centerline:
            x = j * self.resolution + self.resolution / 2
            y = i * self.resolution + self.resolution / 2
            path.append((x, y))

        return path if path else None

    def _reconstruct_path(self, parents: Dict, goal_node: Tuple[int, int, float],
                         start_pose: Tuple[float, float, float],
                         goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Reconstruct path from parents dictionary."""

        path = []
        current = goal_node

        while current is not None:
            i, j, theta = current
            x = j * self.resolution + self.resolution / 2
            y = i * self.resolution + self.resolution / 2
            path.append((x, y))
            current = parents.get(current)

        path.reverse()
        return path

    def _reconstruct_rrt_path(self, tree: Dict, goal_node: Tuple[int, int, float],
                             start_pose: Tuple[float, float, float],
                             goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Reconstruct path from RRT tree."""

        path = []
        current = goal_node

        while current is not None:
            i, j, theta = current
            x = j * self.resolution + self.resolution / 2
            y = i * self.resolution + self.resolution / 2
            path.append((x, y))
            current = tree.get(current)

        path.reverse()
        return path


def test_improved_pathfinding():
    """Test the improved pathfinding algorithm."""

    print("Testing Improved Pathfinding Algorithm")
    print("=" * 50)

    # Create a simple test grid (50x50 cells, 0.2m resolution = 10x10m)
    grid = np.zeros((50, 50), dtype=bool)

    # Add some obstacles
    grid[20:30, 20:30] = True  # Block in middle
    grid[10:15, 35:40] = True  # Another block

    # Test cases (x, y, theta) in meters (within 10x10m grid)
    test_cases = [
        ((1.0, 1.0, 0), (9.0, 9.0)),  # Simple case
        ((1.0, 5.0, 0), (9.0, 5.0)), # Through narrow passage
        ((5.0, 1.0, 0), (5.0, 9.0)), # Around obstacle
    ]

    pathfinder = ImprovedPathfinder(robot_radius=0.25, min_turning_radius=0.5)

    for i, (start_pose, goal_xy) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  Start: {start_pose}")
        print(f"  Goal: {goal_xy}")

        success, path, info = pathfinder.find_path(grid, start_pose, goal_xy)

        if success:
            print(f"  ✅ Success! Path length: {len(path)}")
            print(f"  Method: {info.get('method', 'unknown')}")
        else:
            print(f"  ❌ Failed: {info.get('error', 'unknown error')}")


if __name__ == "__main__":
    test_improved_pathfinding()
