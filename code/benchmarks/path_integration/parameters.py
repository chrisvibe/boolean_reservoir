from benchmarks.utils.parameters import DatasetParameters
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, model_validator, PrivateAttr, field_validator
from pathlib import Path
import math
from benchmarks.path_integration.constrained_foraging_path import LevyFlightStrategy, SimpleRandomWalkStrategy, PolygonBoundary, IntervalBoundary, NoBoundary, generate_polygon_points, stretch_polygon


# Strategy-specific parameter classes
class LevyFlightStrategyParams(BaseModel):
    """Parameters specific to Levy Flight Strategy"""
    alpha: Union[float, List[float]] = Field(1.5, description="Alpha parameter for Levy flight")
    momentum: Union[float, List[float]] = Field(0.0, description="Momentum parameter")


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
        return cls.parse_math_field(v)


class IntervalBoundaryParams(BaseModel):
    """Parameters specific to Interval Boundary"""
    radius: Union[float, List[float]] = Field(1, description="Boundary radius")
    center: Optional[Union[float, List[float]]] = Field(0, description="Boundary center")


class NoBoundaryParams(BaseModel):
    """Parameters for no boundary (empty params)"""
    pass


# Union types for params
StrategyParamsType = Union[LevyFlightStrategyParams, SimpleRandomWalkStrategyParams]
BoundaryParamsType = Union[PolygonBoundaryParams, IntervalBoundaryParams, NoBoundaryParams]


class StrategyParams(BaseModel):
    """Strategy configuration with type and parameters"""
    type: Union[str, List[str]] = Field("LevyFlightStrategy", description="Strategy type(s)")
    params: StrategyParamsType = Field(default_factory=LevyFlightStrategyParams, description="Strategy parameters")
    
    @field_validator('params', mode='before')
    @classmethod
    def create_typed_params(cls, v, info):
        """Create the appropriate params class based on type"""
        if isinstance(v, BaseModel):
            return v
        
        # Get the type from the data
        data = info.data
        strategy_type = data.get('type', 'LevyFlightStrategy')
        
        # Map type to params class
        params_map = {
            'LevyFlightStrategy': LevyFlightStrategyParams,
            'SimpleRandomWalkStrategy': SimpleRandomWalkStrategyParams,
        }
        
        params_class = params_map.get(strategy_type, LevyFlightStrategyParams)
        
        if isinstance(v, dict):
            return params_class(**v)
        return params_class()
    

class BoundaryParams(BaseModel):
    """Boundary configuration with type and parameters"""
    type: Union[str, List[str]] = Field("PolygonBoundary", description="Boundary type(s)")
    params: BoundaryParamsType = Field(default_factory=PolygonBoundaryParams, description="Boundary parameters")
    
    @field_validator('params', mode='before')
    @classmethod
    def create_typed_params(cls, v, info):
        """Create the appropriate params class based on type"""
        if isinstance(v, BaseModel):
            return v
        
        # Get the type from the data
        data = info.data
        boundary_type = data.get('type', 'PolygonBoundary')
        
        # If type is a list, use the first one for now
        if isinstance(boundary_type, list):
            boundary_type = boundary_type[0]
        
        # Map type to params class
        params_map = {
            'PolygonBoundary': PolygonBoundaryParams,
            'IntervalBoundary': IntervalBoundaryParams,
            'NoBoundary': NoBoundaryParams,
        }
        
        params_class = params_map.get(boundary_type, PolygonBoundaryParams)
        
        if isinstance(v, dict):
            return params_class(**v)
        return params_class()
    

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
    
    model_config = {
        "arbitrary_types_allowed": True
    }
   
    @model_validator(mode='after')
    def update_path_after_init(self):
        """Update path after initialization"""
        self.path = self._generate_path()
        return self
    
    @property
    def strategy(self):
        """Get the actual strategy object built from strategy_config"""
        strategy_map = {
            "LevyFlightStrategy": lambda: LevyFlightStrategy(
                dim=self.dimensions,
                alpha=self.strategy_config.params.alpha,
                momentum=self.strategy_config.params.momentum,
            ),
            "SimpleRandomWalkStrategy": lambda: SimpleRandomWalkStrategy(
                step_size=self.strategy_config.params.step_size
            ),
        }
        
        strategy_type = self.strategy_config.type
        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_map[strategy_type]()
    
    @property
    def boundary(self):
        """Get the actual boundary object built from boundary_config"""
        boundary_type = self.boundary_config.type
        
        if boundary_type == "PolygonBoundary":
            points = generate_polygon_points(
                self.boundary_config.params.n_sides,
                self.boundary_config.params.radius,
                rotation=self.boundary_config.params.rotation,
                center=self.boundary_config.params.center,
            )
            points = PolygonBoundary(points=points)
            return stretch_polygon(points, self.boundary_config.params.stretch_x, self.boundary_config.params.stretch_y)
        
        elif boundary_type == "IntervalBoundary":
            return IntervalBoundary(
                (-self.boundary_config.params.radius / 2, self.boundary_config.params.radius / 2),
                self.boundary_config.params.center,
            )
        
        elif boundary_type == "NoBoundary":
            return NoBoundary()
        
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        # Create shorter hash for strategy and boundary
        strategy_str = ''.join(filter(str.isupper, self.strategy_config.type))
        boundary_str = ''.join(filter(str.isupper, self.boundary_config.type))
        
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


if __name__ == '__main__':
    # Note all lists are combinations...
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
    from projects.boolean_reservoir.code.parameters import generate_param_combinations 
    p_list = generate_param_combinations(p)
    
    print(f"Number of parameter combinations: {len(p_list)}")
    for i, p in enumerate(p_list):
        print(f"\n--- Combination {i+1} ---")
        print(f"Strategy: {p.strategy_config}")
        print(f"Boundary: {p.boundary_config}")