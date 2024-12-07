import numpy as np
import random
import math
from shapely.geometry import Point, Polygon
from abc import ABC, abstractmethod
from visualisations import plot_random_walk

class Boundary(ABC):
    def __init__(self, points):
        self.points = points

    @abstractmethod
    def is_inside(self, p):
        pass
    
    @abstractmethod
    def get_closest_distance_from_edge(self, p):
        pass
    
    def generate_polygon_points(self):
        return self.points


class PolygonBoundary(Boundary):
    def __init__(self, points):
        super().__init__(points)
        self.polygon = Polygon(points)

    def is_inside(self, p):
        point = Point(p)
        return self.polygon.contains(point)

    def get_closest_distance_from_edge(self, p):
        point = Point(p)
        return self.polygon.exterior.distance(point)

class IntervalBoundary:
    def __init__(self, interval):
        self.interval = interval

    def is_inside(self, p):
        return self.interval[0] <= p <= self.interval[1]

    def get_closest_distance_from_edge(self, p):
        closest_distance = float('inf')
        if self.is_inside(p):
            return min(abs(p - self.min), abs(self.max - p))
        return closest_distance

class NoBoundary(Boundary):
    def __init__(self):
        super().__init__([])

    def is_inside(self, p):
        return True 

    def get_closest_distance_from_edge(self, p):
        return np.inf

class WalkStrategy(ABC):
    @abstractmethod
    def compute_step(self, p, boundary):
        pass

class SimpleRandomWalkStrategy(WalkStrategy):
    def compute_step(self, p, boundary):
        dp = np.random.uniform(-1, 1, p.shape)
        
        # Update position ensuring it stays inside the boundary
        new_p = p + dp 
        if not boundary.is_inside(p):
            new_p = p 
        
        return new_p 


class LevyFlightStrategy(WalkStrategy):
    def __init__(self, dim=2, alpha=1.5, bias=1, momentum=0.9):
        self.dim = dim
        self.alpha = alpha
        self.bias = bias
        self.momentum = momentum

        # Initialize previous velocities with zeros
        self.v_prev = np.zeros(dim)
    
    def compute_step(self, p, boundary):
        # Generate step length from a power-law distribution
        dp = (np.random.pareto(self.alpha, self.dim) + self.bias) * np.random.choice([-1, 1], size=p.shape)
        
        # Apply momentum by preserving contribution from previous values
        dp = self.momentum * self.v_prev + (1 - self.momentum) * dp

        # Update previous velocities
        self.vp_prev = dp
        
        # Update position ensuring it stays inside the boundary
        new_p = p + dp
        if not boundary.is_inside(new_p):
            new_p = p
        
        return new_p

# Function to simulate the random walk with a strategy
def random_walk(dim: int, steps: int, strategy: WalkStrategy, boundary: Boundary):
    agent_p = np.zeros(dim) 
    positions = []
    # positions.append((agent_p))
    for _ in range(steps):
        agent_p = strategy.compute_step(agent_p, boundary)
        positions.append((agent_p))
        
    return np.array(positions)

def positions_to_p_v_pairs(positions: np.array):
    velocities = np.diff(positions, axis=0, prepend=np.zeros((1, positions.shape[1])))
    return positions, velocities 


# Define a set of points for the boundary (example: hexagon centered at (0, 0))
def generate_polygon_points(n, radius, rotation=0):
    """
    Generate points representing a regular polygon shape with n sides, the given radius, and rotation.
    The polygon will be centered around the origin (0,0).
    
    :param n: Number of sides of the polygon
    :param radius: Radius of the polygon
    :param rotation: Rotation of the polygon in radians
    :return: List of points (tuples) representing the vertices of the polygon
    """
    points = []
    angle_step = 2 * math.pi / n

    for i in range(n):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        # Apply rotation
        rotated_x = x * math.cos(rotation) - y * math.sin(rotation)
        rotated_y = x * math.sin(rotation) + y * math.cos(rotation)

        points.append((rotated_x, rotated_y))

    return points


def stretch_polygon(points, stretch_x, stretch_y):
    """
    Stretch the polygon points along the x and y axes.
    
    :param points: List of points (tuples) representing the vertices of the polygon
    :param stretch_x: Stretch factor along the x-axis
    :param stretch_y: Stretch factor along the y-axis
    :return: List of points (tuples) representing the stretched polygon
    """
    stretched_points = [(x * stretch_x, y * stretch_y) for x, y in points]
    return stretched_points


if __name__ == '__main__':
    dim = 1
    # -------------------------------------------------------------
    # boundary = NoBoundary()
    boundary = IntervalBoundary([-.1, .1]) 
    strategy = LevyFlightStrategy(dim=dim, alpha=3, momentum=0.9, bias=0)

    dim = 2
    # -------------------------------------------------------------
    # boundary = NoBoundary()
    square = generate_polygon_points(4, .1, rotation=np.pi/4) 
    boundary = PolygonBoundary(points=square)
    # strategy = SimpleRandomWalkStrategy(dim=dim)
    strategy = LevyFlightStrategy(dim=dim, alpha=3, momentum=0.9, bias=0)

    # Simulate the walk
    steps = 5
    positions = random_walk(dim, steps, strategy, boundary)
    plot_random_walk(positions, boundary)