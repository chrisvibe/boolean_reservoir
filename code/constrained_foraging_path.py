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
    def is_inside(self, x, y):
        pass
    
    @abstractmethod
    def get_closest_distance_from_edge(self, x, y):
        pass
    
    def generate_polygon_points(self):
        return self.points


class PolygonBoundary(Boundary):
    def __init__(self, points):
        super().__init__(points)
        self.polygon = Polygon(points)

    def is_inside(self, x, y):
        point = Point(x, y)
        return self.polygon.contains(point)

    def get_closest_distance_from_edge(self, x, y):
        point = Point(x, y)
        return self.polygon.exterior.distance(point)

class NoBoundary(Boundary):
    def __init__(self):
        super().__init__([])

    def is_inside(self, x, y):
        return True 

    def get_closest_distance_from_edge(self, x, y):
        return np.inf

class WalkStrategy(ABC):
    @abstractmethod
    def compute_step(self, x, y, boundary):
        pass

class SimpleRandomWalkStrategy(WalkStrategy):
    def compute_step(self, x, y, boundary):
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, 1)
        
        # Update position ensuring it stays inside the boundary
        new_x, new_y = x + dx, y + dy
        if not boundary.is_inside(new_x, new_y):
            new_x, new_y = x, y
        
        return new_x, new_y


class LevyFlightStrategy(WalkStrategy):
    def __init__(self, alpha=1.5, bias=1, momentum=0.9):
        self.alpha = alpha
        self.bias = bias
        self.momentum = momentum

        # Initialize previous velocities with zeros
        self.vx_prev = 0
        self.vy_prev = 0
    
    def compute_step(self, x, y, boundary):
        # Generate step length from a power-law distribution
        step_length = (np.random.pareto(self.alpha) + self.bias)
        
        # Generate a random angle
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Compute the displacement in x and y directions
        dx = step_length * math.cos(angle)
        dy = step_length * math.sin(angle)

        # Apply momentum by preserving contribution from previous values
        dx = self.momentum * self.vx_prev + (1 - self.momentum) * dx
        dy = self.momentum * self.vy_prev + (1 - self.momentum) * dy

        # Update previous velocities
        self.vx_prev = dx
        self.vy_prev = dy
        
        # Update position ensuring it stays inside the boundary
        new_x, new_y = x + dx, y + dy
        if not boundary.is_inside(new_x, new_y):
            new_x, new_y = x, y
        
        return new_x, new_y

# Function to simulate the random walk with a strategy
def random_walk(steps, strategy: WalkStrategy, boundary: Boundary):
    mouse_x, mouse_y = 0, 0
    
    positions = []
    
    for _ in range(steps):
        mouse_x, mouse_y = strategy.compute_step(mouse_x, mouse_y, boundary)
        positions.append((mouse_x, mouse_y))
        
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
    # boundary = NoBoundary()
    square = generate_polygon_points(4, 500, rotation=np.pi/4) 
    # rectangle = stretch_polygon(square, 2, 1/2) 
    # triangle = generate_polygon_points(3, 500) 
    # pentagon = generate_polygon_points(5, 500) 
    # hexagon = generate_polygon_points(6, 500) 
    # circle = generate_polygon_points(20, 10) 
    boundary = PolygonBoundary(points=square)

    # Choose a strategy
    # strategy = SimpleRandomWalkStrategy()
    # strategy = LevyFlightStrategy(momentum=0, bias=0)
    strategy = LevyFlightStrategy(momentum=0.9, bias=0)

    # Simulate the walk
    steps = 25
    positions = random_walk(steps, strategy, boundary)
    
    plot_random_walk(positions, boundary)

