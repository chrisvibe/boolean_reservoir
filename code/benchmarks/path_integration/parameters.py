from benchmarks.utils.parameters import DatasetParameters
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, model_validator, PrivateAttr, field_validator
from pathlib import Path
import math
from benchmarks.path_integration.constrained_foraging_path import LevyFlightStrategy, SimpleRandomWalkStrategy, PolygonBoundary, IntervalBoundary, NoBoundary, generate_polygon_points, stretch_polygon


class StrategyParams(BaseModel):
    """Parameters for walk strategies"""
    type: Union[str, List[str]] = Field("LevyFlightStrategy", description="Strategy type(s)")
    alpha: Union[float, List[float]] = Field(3.0, description="Alpha parameter for Levy flight")
    momentum: Union[float, List[float]] = Field(0.9, description="Momentum parameter")
    step_size: Union[float, List[float]] = Field(1.0, description="Step size for random walk")


class BoundaryParams(BaseModel):
    """Parameters for boundaries"""
    type: Union[str, List[str]] = Field("PolygonBoundary", description="Boundary type(s)")
    n_sides: Union[int, List[int]] = Field(4, description="Number of polygon sides")
    radius: Union[float, List[float]] = Field(0.1, description="Boundary radius")
    dimensions: int = Field(2, description="Number of dimensions")
    center: Optional[Union[List[float], List[List[float]]]] = Field(None, description="Boundary center(s)")
    rotation: Union[float, List[float]] = Field(0, description="Rotation angle radians (pi symbol is ok)")
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


class PathIntegrationDatasetParams(DatasetParameters):
    # Basic parameters
    dimensions: Union[int, List[int]] = Field(2, description="Number of dimensions")
    steps: Union[int, List[int]] = Field(10, description="Number of steps")
    
    # Strategy and boundary parameters as sub-models
    strategy_params: StrategyParams = Field(
        default_factory=StrategyParams,
        description="Strategy parameters",
        alias="strategy_config",
    )
    
    boundary_params: BoundaryParams = Field(
        default_factory=BoundaryParams,
        description="Boundary parameters",
        alias="boundary_config",
    )
    
    # Private attributes for actual objects
    _strategy: Optional[Any] = PrivateAttr(default=None)
    _boundary: Optional[Any] = PrivateAttr(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @model_validator(mode='after')
    def create_complex_objects(self):
        """Create strategy and boundary objects from configurations"""
        # Create objects (they should have single values after generate_param_combinations)
        self._strategy = self._create_strategy()
        self._boundary = self._create_boundary()
        
        # Update path
        self.path = self._generate_path()
        
        return self
    
    def _create_strategy(self):
        """Factory method to create strategy instances"""
        strategy_map = {
            "LevyFlightStrategy": lambda: LevyFlightStrategy(
                dim=self.dimensions,
                alpha=self.strategy_params.alpha,
                momentum=self.strategy_params.momentum,
            ),
            "SimpleRandomWalkStrategy": lambda: SimpleRandomWalkStrategy(
                step_size=self.strategy_params.step_size
            ),
        }
        
        strategy_type = self.strategy_params.type
        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_map[strategy_type]()
    
    def _create_boundary(self):
        """Factory method to create boundary instances"""
        boundary_type = self.boundary_params.type
        
        if boundary_type == "PolygonBoundary":
            points = generate_polygon_points(
                self.boundary_params.n_sides,
                self.boundary_params.radius,
                rotation=self.boundary_params.rotation,
                center=self.boundary_params.center,
            )
            points = PolygonBoundary(points=points)
            return stretch_polygon(points, self.boundary_params.stretch_x, self.boundary_params.stretch_y)
        
        elif boundary_type == "IntervalBoundary":
            return IntervalBoundary(
                (-self.boundary_params.radius / 2, self.boundary_params.radius / 2),
                self.boundary_params.center,
            )
        
        elif boundary_type == "NoBoundary":
            return NoBoundary()
        
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        # Create shorter hash for strategy and boundary
        strategy_str = ''.join(filter(str.isupper, self.strategy_params.type))
        boundary_str = ''.join(filter(str.isupper, self.boundary_params.type))
        
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
    
    @property
    def strategy(self):
        """Get the actual strategy object"""
        if self._strategy is None and not self._has_any_lists():
            self._strategy = self._create_strategy()
        return self._strategy
    
    @property
    def boundary(self):
        """Get the actual boundary object"""
        if self._boundary is None and not self._has_any_lists():
            self._boundary = self._create_boundary()
        return self._boundary
    
    def update_path(self):
        """Update the path based on current parameters"""
        self.path = self._generate_path()
    
    @classmethod
    def from_yaml(cls, yaml_dict: dict):
        """Create instance from YAML configuration"""
        # Parse the YAML structure
        params = {}
        
        # Direct parameters
        for key in ['dimensions', 'steps', 'samples', 'seed', 'path']:
            if key in yaml_dict:
                params[key] = yaml_dict[key]
        
        # Strategy config
        if 'strategy_config' in yaml_dict:
            sc = yaml_dict['strategy_config']
            strategy_params = {
                'type': sc.get('type', 'LevyFlightStrategy')
            }
            if 'params' in sc:
                strategy_params.update(sc['params'])
            params['strategy_params'] = StrategyParams(**strategy_params)
        
        # Boundary config
        if 'boundary_config' in yaml_dict:
            bc = yaml_dict['boundary_config']
            boundary_params = {
                'type': bc.get('type', 'PolygonBoundary')
            }
            if 'params' in bc:
                boundary_params.update(bc['params'])
            params['boundary_params'] = BoundaryParams(**boundary_params)
        
        return cls(**params)
    
    def to_yaml_dict(self) -> dict:
        """Convert to YAML-serializable dictionary"""
        strategy_dict = self.strategy_params.model_dump()
        strategy_type = strategy_dict.pop('type')
        
        boundary_dict = self.boundary_params.model_dump()
        boundary_type = boundary_dict.pop('type')
        
        return {
            "dimensions": self.dimensions,
            "steps": self.steps,
            "samples": self.samples,
            "seed": self.seed,
            "strategy_config": {
                "type": strategy_type,
                "params": strategy_dict
            },
            "boundary_config": {
                "type": boundary_type,
                "params": boundary_dict
            }
        }


if __name__ == '__main__':
    # Note all lists are combinations...

    # Example 1: Mixed strategy types from class
    params_mixed = PathIntegrationDatasetParams(
        dimensions=2,
        steps=10,
        strategy_params=StrategyParams(
            type=["LevyFlightStrategy", "SimpleRandomWalkStrategy"],  # List of strategy types
            alpha=3.0,
            momentum=0.9
        )
    )
    
    # Example 2: Many combinations from YAML
    import yaml
    
    yaml_content = """
    dimensions: [2, 3]
    steps: 10
    strategy_config:
      type: "LevyFlightStrategy"
      params:
        alpha: [2.5, 3.0, 3.5]
        momentum: [0.8, 0.9]
    boundary_config:
      type: "PolygonBoundary"
      params:
        n_sides: [4, 6]
        radius: 0.2
        rotation: 1.57
    """

    # Generate combinations
    from projects.boolean_reservoir.code.parameters import generate_param_combinations 
    p_list = generate_param_combinations(params_mixed)
    
    print(f"Number of parameter combinations: {len(p_list)}")
    print(f"Expected: 2 dims × 2 steps × 3 alphas × 2 momentums × 3 n_sides × 2 rotations = {2*2*3*2*3*2}")
    
    for i, p in enumerate(p_list[:5]):  # Show first 5 combinations
        print(f"\n--- Combination {i+1} ---")
        print(f"Dimensions: {p.dimensions}")
        print(f"Steps: {p.steps}")
        print(f"Strategy type: {p.strategy_params.type}")
        print(f"  alpha: {p.strategy_params.alpha}")
        print(f"  momentum: {p.strategy_params.momentum}")
        print(f"Boundary type: {p.boundary_params.type}")
        print(f"  n_sides: {p.boundary_params.n_sides}")
        print(f"  rotation: {p.boundary_params.rotation}")