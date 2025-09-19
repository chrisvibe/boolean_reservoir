from benchmarks.utils.parameters import DatasetParameters
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, model_validator, PrivateAttr, field_validator
from pathlib import Path
import math
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


# Strategy-specific parameter classes
class LevyFlightStrategyParams(BaseModel):
    """Parameters specific to Levy Flight Strategy"""
    alpha: Union[float, List[float]] = Field(1.5, description="Alpha parameter for pareto distribtion (Levy flight)")
    momentum: Union[float, List[float]] = Field(0.0, description="Momentum parameter")
    step_size: Union[float, List[float]] = Field(1.0, description="Scaling of stepsize from distribution")

class SimpleRandomWalkStrategyParams(BaseModel):
    """Parameters specific to Simple Random Walk Strategy"""
    step_size: Union[float, List[float]] = Field(1.0, description="Step size for random walk")


# Boundary-specific parameter classes
class PolygonBoundaryParams(BaseModel):
    """Parameters specific to Polygon Boundary"""
    n_sides: Union[int, List[int]] = Field(4, description="Number of polygon sides")
    radius: Union[float, List[float]] = Field(1, description="Boundary radius")
    center: Optional[Union[List[float], List[List[float]]]] = Field(None, description="Boundary center(s)")
    rotation: Union[float, List[float]] = Field(math.pi/4, description="Rotation angle radians")
    stretch_x: Union[float, List[float]] = Field(1, description="Scale in x axis")
    stretch_y: Union[float, List[float]] = Field(1, description="Scale in y axis")
    
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
    radius: Union[float, List[float]] = Field(1, description="Boundary radius")
    center: Optional[Union[float, List[float]]] = Field(0, description="Boundary center")


class NoBoundaryParams(BaseModel):
    """Parameters for no boundary (empty params)"""
    pass


class ParamsDict(BaseModel):
    """Generic parameters container that allows any fields"""
    model_config = {
        "extra": "allow"  # Allow any fields to be added dynamically
    }


class StrategyParams(BaseModel):
    """Strategy configuration with type and parameters"""
    type: Union[str, List[str]] = Field("LevyFlightStrategy", description="Strategy type(s)")
    params: ParamsDict = Field(default_factory=ParamsDict, description="Strategy parameters")
    
    _typed_params: Optional[BaseModel] = PrivateAttr(default=None)
    
    @property
    def typed_params(self):
        """Lazy property that returns the typed params object"""
        if self._typed_params is None:
            strategy_type = self.type
            params_map = {
                'LevyFlightStrategy': LevyFlightStrategyParams,
                'SimpleRandomWalkStrategy': SimpleRandomWalkStrategyParams,
            }
            params_class = params_map.get(strategy_type, LevyFlightStrategyParams)
            self._typed_params = params_class(**self.params.__dict__)
        return self._typed_params
    

class BoundaryParams(BaseModel):
    """Boundary configuration with type and parameters"""
    type: Union[str, List[str]] = Field("PolygonBoundary", description="Boundary type(s)")
    params: ParamsDict = Field(default_factory=ParamsDict, description="Boundary parameters")
    
    _typed_params: Optional[BaseModel] = PrivateAttr(default=None)
    
    @property
    def typed_params(self):
        """Lazy property that returns the typed params object"""
        if self._typed_params is None:
            boundary_type = self.type
            params_map = {
                'PolygonBoundary': PolygonBoundaryParams,
                'IntervalBoundary': IntervalBoundaryParams,
                'NoBoundary': NoBoundaryParams,
            }
            params_class = params_map.get(boundary_type, PolygonBoundaryParams)
            self._typed_params = params_class(**self.params.__dict__)
        return self._typed_params
    

class PathIntegrationDatasetParams(DatasetParameters):
    # Basic parameters
    dimensions: Union[int, List[int]] = Field(2, description="Number of dimensions")
    steps: Union[int, List[int]] = Field(10, description="Number of steps")
    
    # Strategy and boundary configurations (field names match YAML keys)
    strategy_config: StrategyParams = Field(
        default_factory=StrategyParams,
        description="Strategy configuration"
    )
    
    boundary_config: BoundaryParams = Field(
        default_factory=BoundaryParams,
        description="Boundary configuration"
    )
    
    # Private attributes for caching lazy-loaded objects
    _strategy: Optional[Any] = PrivateAttr(default=None)
    _boundary: Optional[Any] = PrivateAttr(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
   
    @model_validator(mode='after')
    def update_path_after_init(self):
        """Update path after initialization - only if not using list parameters"""
        # Skip if any parameter is a list (will be expanded combinatorially)
        if (isinstance(self.dimensions, list) or 
            isinstance(self.steps, list) or
            isinstance(self.strategy_config.type, list) or
            isinstance(self.boundary_config.type, list)):
            return self
        
        self.path = self._generate_path()
        return self
    
    @property
    def strategy(self):
        """Lazy property that creates and caches the strategy object"""
        if self._strategy is None:
            strategy_map = {
                "LevyFlightStrategy": lambda: LevyFlightStrategy(
                    dim=self.dimensions,
                    alpha=self.strategy_config.typed_params.alpha,
                    momentum=self.strategy_config.typed_params.momentum,
                    step_size=self.strategy_config.typed_params.step_size,
                ),
                "SimpleRandomWalkStrategy": lambda: SimpleRandomWalkStrategy(
                    step_size=self.strategy_config.typed_params.step_size
                ),
            }
            
            strategy_type = self.strategy_config.type
            if strategy_type not in strategy_map:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            self._strategy = strategy_map[strategy_type]()
        
        return self._strategy
    
    @property
    def boundary(self):
        """Lazy property that creates and caches the boundary object"""
        if self._boundary is None:
            boundary_type = self.boundary_config.type
            
            if boundary_type == "PolygonBoundary":
                typed_params = self.boundary_config.typed_params
                points = generate_polygon_points(
                    typed_params.n_sides,
                    typed_params.radius,
                    rotation=typed_params.rotation,
                    center=typed_params.center,
                )
                points = PolygonBoundary(points=points)
                self._boundary = stretch_polygon(
                    points, 
                    typed_params.stretch_x, 
                    typed_params.stretch_y
                )
            
            elif boundary_type == "IntervalBoundary":
                typed_params = self.boundary_config.typed_params
                self._boundary = IntervalBoundary(
                    (-typed_params.radius / 2, typed_params.radius / 2),
                    typed_params.center,
                )
            
            elif boundary_type == "NoBoundary":
                self._boundary = NoBoundary()
            
            else:
                raise ValueError(f"Unknown boundary type: {boundary_type}")
        
        return self._boundary
    
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        # Create shorter hash for strategy and boundary
        strategy_type = self.strategy_config.type
        if isinstance(strategy_type, list):
            strategy_type = strategy_type[0]
        boundary_type = self.boundary_config.type
        if isinstance(boundary_type, list):
            boundary_type = boundary_type[0]
            
        strategy_str = ''.join(filter(str.isupper, strategy_type))
        boundary_str = ''.join(filter(str.isupper, boundary_type))
        
        return Path(
            f'data/path_integration/'
            f'd-{self.dimensions}/'
            f's-{self.steps}/'
            f'{strategy_str}/'
            f'{boundary_str}/'
            f'm-{self.samples}/'
            f'r-{self.seed}/'
            f'dataset.pt'
        )
    
    def update_path(self):
        """Update the path based on current parameters"""
        self.path = self._generate_path()
    
    def clear_cache(self):
        """Clear cached lazy properties (useful after deserialization)"""
        self._strategy = None
        self._boundary = None
        if hasattr(self.strategy_config, '_typed_params'):
            self.strategy_config._typed_params = None
        if hasattr(self.boundary_config, '_typed_params'):
            self.boundary_config._typed_params = None


if __name__ == '__main__':
    # Example: Many combinations from YAML
    import yaml
    
    yaml_content = """
    dimensions: [2, 3]
    steps: 10
    strategy_config:
      type: LevyFlightStrategy
      params:
        alpha: 3.0
        momentum: [0.8, 0.9]
    boundary_config:
      type: PolygonBoundary
      params:
        n_sides: [4, 6]
        radius: 0.2
        rotation: 1.57
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams(**config)
    
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
        print(f"Strategy config: {p.strategy_config.params}")
        print(f"Boundary config: {p.boundary_config.params}")
        print(f"Strategy object: {p.strategy}")  # Will be created lazily
        print(f"Boundary object: {p.boundary}")  # Will be created lazily