from benchmarks.utils.parameters import DatasetParameters
from typing import Dict, Any, Optional, Union, List, Literal
from pydantic import BaseModel, Field, model_validator, PrivateAttr, field_validator
from pathlib import Path
import math
import hashlib
import json
from benchmarks.path_integration.constrained_foraging_path import (
    LevyFlightStrategy, SimpleRandomWalkStrategy, 
    PolygonBoundary, IntervalBoundary, NoBoundary, 
    generate_polygon_points, stretch_polygon
)

'''
Note that complex objects are accessed as properties.
These are designed to be lazy and private to avoid dumping them.
All lists as usual are expanded for many instances with the values in the list.
'''

# Strategy parameter classes
class LevyFlightStrategyParams(BaseModel):
    """Parameters specific to Levy Flight Strategy"""
    type: Literal["LevyFlightStrategy"] = "LevyFlightStrategy"
    alpha: Union[float, List[float]] = Field(1.5, description="Alpha parameter for pareto distribtion (Levy flight)")
    momentum: Union[float, List[float]] = Field(0.0, description="Momentum parameter")
    step_size: Union[float, List[float]] = Field(1.0, description="Scaling of stepsize from distribution")

class SimpleRandomWalkStrategyParams(BaseModel):
    """Parameters specific to Simple Random Walk Strategy"""
    type: Literal["SimpleRandomWalkStrategy"] = "SimpleRandomWalkStrategy"
    step_size: Union[float, List[float]] = Field(1.0, description="Step size for random walk")

# Boundary parameter classes
class PolygonBoundaryParams(BaseModel):
    """Parameters specific to Polygon Boundary"""
    type: Literal["PolygonBoundary"] = "PolygonBoundary"
    n_sides: Union[int, List[int]] = Field(4, description="Number of polygon sides")
    radius: Union[float, List[float]] = Field(1.0, description="Boundary radius")
    center: Optional[Union[List[float], List[List[float]]]] = Field(None, description="Boundary center(s)")
    rotation: Union[float, List[float]] = Field(math.pi/4, description="Rotation angle radians")
    stretch_x: Union[float, List[float]] = Field(1.0, description="Scale in x axis")
    stretch_y: Union[float, List[float]] = Field(1.0, description="Scale in y axis")
    
    @classmethod
    def parse_math_expression(cls, expr):
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, str):
            try:
                return float(eval(expr, {'pi': math.pi}))
            except:
                raise ValueError(f"Invalid mathematical expression: {expr}")
        return expr
    
    @classmethod
    def parse_math_field(cls, v):
        if isinstance(v, list):
            return [cls.parse_math_expression(item) for item in v]
        else:
            return cls.parse_math_expression(v)
    
    @field_validator('rotation', 'radius', 'n_sides', 'stretch_x', 'stretch_y', mode='before')
    @classmethod
    def parse_numeric_fields(cls, v):
        # Skip parsing if we have lists (will be expanded combinatorially)
        if isinstance(v, list):
            return v
        return cls.parse_math_field(v)

class IntervalBoundaryParams(BaseModel):
    """Parameters specific to Interval Boundary"""
    type: Literal["IntervalBoundary"] = "IntervalBoundary"
    radius: Union[float, List[float]] = Field(1.0, description="Boundary radius")
    center: Optional[Union[float, List[float]]] = Field(0.0, description="Boundary center")

class NoBoundaryParams(BaseModel):
    """Parameters for no boundary"""
    type: Literal["NoBoundary"] = "NoBoundary"

# Union types for strategy and boundary parameters
StrategyParams = Union[LevyFlightStrategyParams, SimpleRandomWalkStrategyParams]
BoundaryParams = Union[PolygonBoundaryParams, IntervalBoundaryParams, NoBoundaryParams]

class PathIntegrationDatasetParams(DatasetParameters):
    # Basic parameters
    dimensions: Union[int, List[int]] = Field(2, description="Number of dimensions")
    steps: Union[int, List[int]] = Field(10, description="Number of steps")
    
    # Strategy and boundary parameters (flattened)
    strategy: StrategyParams = Field(
        default_factory=LevyFlightStrategyParams,
        description="Strategy parameters",
        discriminator='type'
    )
    
    boundary: BoundaryParams = Field(
        default_factory=PolygonBoundaryParams,
        description="Boundary parameters",
        discriminator='type'
    )
    
    # Private attributes for caching lazy-loaded objects
    _strategy_obj: Optional[Any] = PrivateAttr(default=None)
    _boundary_obj: Optional[Any] = PrivateAttr(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
   
    @model_validator(mode='after')
    def update_path_after_init(self):
        """Update path after initialization - only if not using list parameters"""
        # Skip if any parameter is a list (will be expanded combinatorially)
        if (isinstance(self.dimensions, list) or 
            isinstance(self.steps, list) or
            self._has_list_in_strategy() or
            self._has_list_in_boundary()):
            return self
        
        self.path = self._generate_path()
        return self
    
    def _has_list_in_strategy(self) -> bool:
        """Check if strategy has any list parameters"""
        for field_name, field_value in self.strategy.__dict__.items():
            if isinstance(field_value, list):
                return True
        return False
    
    def _has_list_in_boundary(self) -> bool:
        """Check if boundary has any list parameters"""
        for field_name, field_value in self.boundary.__dict__.items():
            if isinstance(field_value, list):
                return True
        return False
    
    @property
    def strategy_obj(self):
        """Lazy property that creates and caches the strategy object"""
        if self._strategy_obj is None:
            if self.strategy.type == "LevyFlightStrategy":
                self._strategy_obj = LevyFlightStrategy(
                    dim=self.dimensions,
                    alpha=self.strategy.alpha,
                    momentum=self.strategy.momentum,
                    step_size=self.strategy.step_size,
                )
            elif self.strategy.type == "SimpleRandomWalkStrategy":
                self._strategy_obj = SimpleRandomWalkStrategy(
                    step_size=self.strategy.step_size
                )
            else:
                raise ValueError(f"Unknown strategy type: {self.strategy.type}")
        
        return self._strategy_obj
    
    @property
    def boundary_obj(self):
        """Lazy property that creates and caches the boundary object"""
        if self._boundary_obj is None:
            if self.boundary.type == "PolygonBoundary":
                points = generate_polygon_points(
                    self.boundary.n_sides,
                    self.boundary.radius,
                    rotation=self.boundary.rotation,
                    center=self.boundary.center,
                )
                points = PolygonBoundary(points=points)
                self._boundary_obj = stretch_polygon(
                    points, 
                    self.boundary.stretch_x, 
                    self.boundary.stretch_y
                )
            
            elif self.boundary.type == "IntervalBoundary":
                self._boundary_obj = IntervalBoundary(
                    (-self.boundary.radius / 2, self.boundary.radius / 2),
                    self.boundary.center,
                )
            
            elif self.boundary.type == "NoBoundary":
                self._boundary_obj = NoBoundary()
            
            else:
                raise ValueError(f"Unknown boundary type: {self.boundary.type}")
        
        return self._boundary_obj
    
    @staticmethod
    def _hash_basemodel(base_model, n_chars=5):
        model_dict = base_model.model_dump()
        json_str = json.dumps(model_dict, sort_keys=True)
        hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        return hash[:n_chars]
        
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        strategy_str = ''.join(filter(str.isupper, self.strategy.type))
        strategy_hash = self._hash_basemodel(self.strategy) 
        boundary_str = ''.join(filter(str.isupper, self.boundary.type))
        boundary_hash = self._hash_basemodel(self.boundary)
        
        return Path(
            f'data/path_integration/'
            f'd-{self.dimensions}/'
            f's-{self.steps}/'
            f'{strategy_str}/'
            f'{strategy_hash}/'
            f'{boundary_str}/'
            f'{boundary_hash}/'
            f'm-{self.samples}/'
            f'r-{self.seed}/'
            f'dataset.pt'
        )
    
    def update_path(self):
        """Update the path based on current parameters"""
        self.path = self._generate_path()
    
    def clear_cache(self):
        """Clear cached lazy properties (useful after deserialization)"""
        self._strategy_obj = None
        self._boundary_obj = None


if __name__ == '__main__':
    # Example: Many combinations from YAML
    import yaml
    
    yaml_content = """
    dimensions: [2, 3]
    steps: 10
    strategy:
      type: LevyFlightStrategy
      alpha: 3.0
      momentum: [0.8, 0.9]
    boundary:
      type: PolygonBoundary
      n_sides: [4, 6]
      radius: 0.2
      rotation: 1.57
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams(**config)
    
    print(f"Strategy: {p.strategy}")
    print(f"Boundary: {p.boundary}")
    
    # Test serialization - no warnings!
    import pickle
    serialized = pickle.dumps(p)
    deserialized = pickle.loads(serialized)
    
    # After deserialization, clear cache to ensure fresh objects
    deserialized.clear_cache()
    
    # Test with generate_param_combinations
    from projects.boolean_reservoir.code.parameters import generate_param_combinations 
    p_list = generate_param_combinations(p)
    
    print(f"Number of parameter combinations: {len(p_list)}")
    for i, p in enumerate(p_list[:2]):  # Show first 2
        print(f"\n--- Combination {i+1} ---")
        print(f"Strategy: {p.strategy}")
        print(f"Boundary: {p.boundary}")
        print(f"Strategy object: {p.strategy_obj}")  # Will be created lazily
        print(f"Boundary object: {p.boundary_obj}")  # Will be created lazily