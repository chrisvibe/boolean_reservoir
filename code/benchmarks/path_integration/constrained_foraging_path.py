import numpy as np
import math
from shapely.geometry import Point, Polygon
from abc import ABC, abstractmethod
from benchmarks.path_integration.visualizations import plot_random_walk

class Boundary(ABC):
    def __init__(self, points):
        self.points = points

    @abstractmethod
    def is_inside(self, p: np.array):
        pass
    
    @abstractmethod
    def get_closest_distance_from_edge(self, p):
        pass
    
    def get_points(self):
        return self.points
    
    def __str__(self):
        return f'{self.__class__.__name__}'

    @staticmethod
    def _ensure_numpy_array(p):
        """Ensure input is a numpy array with proper shape"""
        p = np.asarray(p)
        # Handle scalar case (0D array) by converting to 1D
        if p.ndim == 0:
            p = np.array([p.item()])
        return p


class PolygonBoundary(Boundary):
    def __init__(self, points):
        super().__init__(points)
        self.polygon = Polygon(points)

    def is_inside(self, p):
        return self.polygon.contains(Point(p))

    def get_closest_distance_from_edge(self, p):
        return self.polygon.exterior.distance(Point(p))

class IntervalBoundary(Boundary):
    def __init__(self, interval, center=None):
        self.interval = interval
        self.center = 0 if center is None else center
        self.effective_interval = tuple(x + self.center for x in self.interval)
        self.min = min(self.effective_interval)
        self.max = max(self.effective_interval)
        self.points = (self.min, self.max)
    
    def is_inside(self, p):
        return self.effective_interval[0] <= p <= self.effective_interval[1]
    
    def get_closest_distance_from_edge(self, p):
        if self.is_inside(p):
            return min(abs(p - self.min), abs(p - self.max))
        else:
            if p < self.min:
                return self.min - p
            else:
                return p - self.max
    
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

    def __str__(self):
        return f'{self.__class__.__name__}'

class SimpleRandomWalkStrategy(WalkStrategy):
    def __init__(self):
        pass

    @staticmethod
    def compute_step(p, boundary):
        dp = np.random.uniform(-1, 1, p.shape)
        
        # Update position ensuring it stays inside the boundary
        new_p = p + dp 
        if not boundary.is_inside(new_p):
            new_p = p 
        
        return new_p 

class LevyFlightStrategy(WalkStrategy):
    def __init__(self, dim=2, alpha=1.5, momentum=0.9, momentum_bias=0, bias_direction=None):
        self.dim = dim
        self.alpha = alpha
        self.momentum = momentum
        self.momentum_bias = momentum_bias  # Scalar magnitude
        
        # Set bias direction vector (unit vector)
        if bias_direction is None:
            # Default: no directional bias
            self.bias_direction = np.zeros(dim)
        else:
            # Normalize the direction vector
            direction = np.array(bias_direction)
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.bias_direction = direction / norm
            else:
                self.bias_direction = np.zeros(dim)
        
        # Create the bias vector (like a gravity/force vector)
        self.bias_vector = self.momentum_bias * self.bias_direction
        
        # Initialize previous velocities with zeros
        self.v_prev = np.zeros(dim)
    
    def compute_step(self, p, boundary):
        # Generate step length from a power-law distribution (random impulse)
        dp = np.random.pareto(self.alpha, self.dim) * np.random.choice([-1, 1], size=p.shape)
        
        # Apply momentum: preserve previous velocity, add random impulse, and add constant bias
        dp = self.momentum * self.v_prev + (1 - self.momentum) * dp + self.bias_vector
        
        # Update previous velocities
        self.v_prev = dp
        
        # Update position ensuring it stays inside the boundary
        new_p = p + dp
        if not boundary.is_inside(new_p):
            new_p = p
            # reset momentum when hitting boundary
            self.v_prev = np.zeros(self.dim)
        
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


def generate_polygon_points(n, radius, rotation=0, center=None):
    """
    Generate points representing a regular polygon shape with n sides, the given radius, and rotation.
    The polygon will be centered around the specified center point.
    :param n: Number of sides of the polygon
    :param radius: Radius of the polygon
    :param rotation: Rotation of the polygon in radians
    :param center: Center point of the polygon, defaults to (0, 0)
    :return: List of points (tuples) representing the vertices of the polygon
    """
    if center is None: center = (0, 0)
    points = []
    angle_step = 2 * math.pi / n
    for i in range(n):
        # Calculate the base angle and apply rotation
        angle = i * angle_step + rotation
        
        # Generate point relative to center
        x = radius * math.cos(angle) + center[0]
        y = radius * math.sin(angle) + center[1]
        
        points.append((x, y))
    return points

def stretch_polygon(polygon_boundary: PolygonBoundary, stretch_x: float, stretch_y: float):
    """
    Stretch the polygon points along the x and y axes.
    """
    stretched_points = [(x * stretch_x, y * stretch_y) for x, y in polygon_boundary.get_points()]
    polygon_boundary.points = stretched_points
    polygon_boundary.polygon = Polygon(stretched_points)
    return polygon_boundary


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
    strategy = SimpleRandomWalkStrategy()
    # strategy = LevyFlightStrategy(dim=dim, alpha=3, momentum=0.9, bias=0)

    # Simulate the walk
    steps = 5
    positions = random_walk(dim, steps, strategy, boundary)
    plot_random_walk('/out/', positions, strategy, boundary)