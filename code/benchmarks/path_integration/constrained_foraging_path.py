import numpy as np
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from abc import ABC, abstractmethod
from benchmarks.path_integration.visualizations import plot_random_walk

class Boundary(ABC):
    def __init__(self, points):
        self.points = points

    @abstractmethod
    def is_inside(self, p: np.array):
        pass

    @abstractmethod
    def handle_boundary_crossing(self, p_start, p_end):
        """Handle when step crosses boundary. Returns valid end position."""
        pass
        
    def get_points(self):
        return self.points
    
    def __str__(self):
        return f'{self.__class__.__name__}'

class NoBoundary(Boundary):
    def __init__(self):
        super().__init__([])

    def is_inside(self, p):
        return True 

    def handle_boundary_crossing(self, p_start, p_end):
        return p_end

class IntervalBoundary(Boundary):
    def __init__(self, interval, center=None, boundary_tolerance=1e-10):
        self.interval = interval
        self.center = 0 if center is None else center
        self.effective_interval = tuple(x + self.center for x in self.interval)
        self.min = min(self.effective_interval)
        self.max = max(self.effective_interval)
        self.points = (self.min, self.max)
        self.boundary_tolerance = boundary_tolerance
    
    def is_inside(self, p):
        """Works with both scalar and array input"""
        val = np.asarray(p).flat[0]
        return self.min <= val <= self.max
    
    def is_on_boundary(self, p):
        val = np.asarray(p).flat[0]
        return (abs(val - self.min) < self.boundary_tolerance or 
                abs(val - self.max) < self.boundary_tolerance)
    
    def handle_boundary_crossing(self, p_start, p_end):
        """Preserves input shape/type"""
        # Work with arrays consistently
        p_end = np.asarray(p_end)
        
        if self.is_inside(p_end):
            return p_end
        
        # Clamp to boundary, preserving shape
        end_val = p_end.flat[0]
        if end_val < self.min:
            clamped = self.min
        else:
            clamped = self.max
        
        # Return with same shape as p_end
        result = p_end.copy()
        result.flat[0] = clamped
        return result
    
class PolygonBoundary(Boundary):
    def __init__(self, points, boundary_tolerance=1e-10):
        super().__init__(points)
        self.polygon = Polygon(points)
        self.boundary = self.polygon.boundary
        self.boundary_tolerance = boundary_tolerance
    
    def get_points(self):
        return self.polygon.exterior.coords

    def is_inside(self, p):
        """Check if point is inside polygon (including boundary)"""
        point = Point(p)
        return self.polygon.contains(point) or self.polygon.touches(point)

    def handle_boundary_crossing(self, p_start, p_end):
        if self.is_inside(p_end):
            return p_end
        
        intersections = self.boundary.intersection(LineString([p_start, p_end]))
        if intersections.is_empty:
            return p_start
        
        _, closest = nearest_points(Point(p_start), intersections)
        intersection_coords = np.array(closest.coords[0])
        
        # Project back if needed due to numerical errors
        if not self.is_inside(intersection_coords):
            boundary_point = self.polygon.boundary.interpolate(
                self.polygon.boundary.project(Point(intersection_coords))
            )
            intersection_coords = np.array(boundary_point.coords[0])
        
        return intersection_coords

class WalkStrategy(ABC):
    @abstractmethod
    def compute_step(self, p, boundary: Boundary):
        pass

    def __str__(self):
        return f'{self.__class__.__name__}'

class SimpleRandomWalkStrategy(WalkStrategy):
    def __init__(self, max_step_size, max_attempts=100):
        self.max_step_size = max_step_size
        self.max_attempts = max_attempts

    def compute_step(self, p, boundary: Boundary):
        """Compute step with resampling if stuck on boundary"""
        for attempt in range(self.max_attempts):
            dp = np.random.uniform(-1, 1, p.shape) * self.max_step_size
            new_p = p + dp
            result = boundary.handle_boundary_crossing(p, new_p)
            if result is not None:
                return result
            
        raise RuntimeError(f"Can't find a valid step after {self.max_attempts} attempts")


class LevyFlightStrategy(WalkStrategy):
    def __init__(self, dim=2, alpha=1, step_size=1, step_size_bias=0, momentum=0.9, 
                 momentum_bias=0, bias_direction=None, max_attempts=100):
        self.dim = dim
        self.alpha = alpha
        self.step_size = step_size
        self.step_size_bias = step_size_bias
        self.momentum = momentum
        self.momentum_bias = momentum_bias
        self.max_attempts = max_attempts
        
        if bias_direction is None:
            self.bias_direction = np.zeros(dim)
        else:
            direction = np.array(bias_direction)
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.bias_direction = direction / norm
            else:
                self.bias_direction = np.zeros(dim)
        
        self.bias_vector = self.momentum_bias * self.bias_direction
        self.v_prev = np.zeros(dim)
    
    def compute_step(self, p, boundary: Boundary):
        """Compute step with resampling if stuck on boundary"""
        for attempt in range(self.max_attempts):
            # Generate step length from power-law distribution
            dp = (self.step_size * np.random.pareto(self.alpha, self.dim) + self.step_size_bias) * \
                 np.random.choice([-1, 1], size=p.shape)
            
            # Apply momentum only if not stuck (first attempt uses previous velocity)
            if attempt == 0:
                dp = self.momentum * self.v_prev + (1 - self.momentum) * dp + self.bias_vector
            else:
                # On resamples, don't use old momentum (we're stuck)
                dp = dp + self.bias_vector
            
            new_p = p + dp
            result = boundary.handle_boundary_crossing(p, new_p)
            
            if result is not None:
                self.v_prev = result - p
                return result
        
        # If we couldn't find a valid move, stay in place and reset velocity
        self.v_prev = np.zeros(self.dim)
        return p

def random_walk(dim: int, steps: int, strategy: WalkStrategy, boundary: Boundary, origin=None):
    agent_p = np.zeros(dim) if origin is None else origin
    if not boundary.is_inside(agent_p):
        raise RuntimeError(f'Illegal start position according to {boundary}: {agent_p}')
    positions = []
    # positions.append((agent_p))
    for _ in range(steps):
        agent_p = strategy.compute_step(agent_p, boundary)
        positions.append(agent_p.copy())
        
    return np.array(positions)

def positions_to_p_v_pairs(positions: np.array):
    velocities = np.diff(positions, axis=0, prepend=np.zeros((1, positions.shape[1])))
    return positions, velocities 

def generate_polygon_points(n, radius, rotation=0, center=None, decimals=10):
    """
    Generate points representing a regular polygon shape with n sides, the given radius, and rotation.
    The polygon will be centered around the specified center point.
    :param n: Number of sides of the polygon
    :param radius: Radius of the polygon
    :param rotation: Rotation of the polygon in radians
    :param center: Center point of the polygon, defaults to (0, 0)
    :return: Numpy array of points representing the vertices of the polygon
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
    return np.round(points, decimals)

def stretch_polygon(polygon_boundary: PolygonBoundary, stretch_x: float, stretch_y: float):
    """
    Stretch the polygon points along the x and y axes.
    """
    stretched_points = [(x * stretch_x, y * stretch_y) for x, y in polygon_boundary.get_points()]
    polygon_boundary.points = stretched_points
    polygon_boundary.polygon = Polygon(stretched_points)
    return polygon_boundary


if __name__ == '__main__':

    # dim = 1
    # # -------------------------------------------------------------
    # # boundary = NoBoundary()
    # boundary = IntervalBoundary([-.5, .5]) 
    # strategy = LevyFlightStrategy(dim=dim, alpha=3, momentum=0.9)

    dim = 2
    # -------------------------------------------------------------
    # boundary = NoBoundary()
    square = generate_polygon_points(4, 1, rotation=np.pi/4) 
    boundary = PolygonBoundary(points=square)
    # strategy = SimpleRandomWalkStrategy(1)
    strategy = LevyFlightStrategy(dim=dim, alpha=3, momentum=0.9)

    # Simulate the walk
    steps = 5
    positions = random_walk(dim, steps, strategy, boundary)
    plot_random_walk('/out/', positions, strategy, boundary)