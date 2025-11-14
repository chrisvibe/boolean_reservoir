from typing import Union, List
from pydantic import Field, model_validator
from pathlib import Path
import math
import hashlib
import json
from benchmarks.utils.parameters import DatasetParameters
from benchmarks.path_integration.constrained_foraging_path import (
    LevyFlightStrategy, SimpleRandomWalkStrategy, 
    PolygonBoundary, IntervalBoundary, NoBoundary, 
    generate_polygon_points, stretch_polygon
)
from projects.boolean_reservoir.code.utils.param_utils import DynamicParams, ExpressionEvaluator

def strategy_factory(p: DynamicParams, dimensions: int):
    """Factory function to create strategy objects"""
    strategy_map = {
        'LevyFlightStrategy': LevyFlightStrategy,
        'SimpleRandomWalkStrategy': SimpleRandomWalkStrategy,
    }
    
    if p.name not in strategy_map:
        raise ValueError(f"Unsupported strategy type: {p.name}. Available: {list(strategy_map.keys())}")

    if p.name == 'SimpleRandomWalkStrategy':
        return p.call(strategy_map[p.name])
    else:
        return p.call(strategy_map[p.name], dim=dimensions)
    
def boundary_factory(p: DynamicParams):
    """Factory function to create boundary objects"""
    evaluator = ExpressionEvaluator(symbols={'pi': math.pi})
    
    if p.name == 'PolygonBoundary':
        points = p.call(generate_polygon_points, evaluator=evaluator)
        boundary = PolygonBoundary(points=points)
        return p.call(stretch_polygon, evaluator=evaluator, boundary=boundary)
    
    elif p.name == 'IntervalBoundary':
        return p.call(IntervalBoundary)
    
    elif p.name == 'NoBoundary':
        return p.call(NoBoundary)
    
    else:
        raise ValueError(f"Unsupported boundary type: {p.name}. Available: ['PolygonBoundary', 'IntervalBoundary', 'NoBoundary']")


class PathIntegrationDatasetParams(DatasetParameters):
    dimensions: Union[int, List[int]] = Field(2, description="Number of dimensions")
    steps: Union[int, List[int]] = Field(10, description="Number of steps")

    strategy: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(
            name='LevyFlightStrategy',
            params={'alpha': 1.5, 'momentum': 0.0, 'step_size': 1.0}
        ),
        description="Strategy configuration"
    )

    boundary: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(
            name='PolygonBoundary',
            params={'n_sides': 4, 'radius': 1.0, 'rotation': math.pi/4, 'stretch_x': 1.0, 'stretch_y': 1.0}
        ),
        description="Boundary configuration"
    )

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    @property
    def strategy_obj(self):
        """Property that constructs the strategy object using factory"""
        return strategy_factory(self.strategy, self.dimensions)
    
    @property
    def boundary_obj(self):
        """Property that constructs the boundary object using factory"""
        return boundary_factory(self.boundary)
    
    @staticmethod
    def _hash_dict(params_dict, n_chars=5):
        """Hash a dictionary for path generation"""
        json_str = json.dumps(params_dict, sort_keys=True)
        hash_obj = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        return hash_obj[:n_chars]
        
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        strategy_str = self.strategy.name[:3].upper()  # First 3 chars of name
        strategy_hash = self._hash_dict(self.strategy.params)
        
        boundary_str = self.boundary.name[:3].upper()  # First 3 chars of name
        boundary_hash = self._hash_dict(self.boundary.params)
        
        return (
            Path('data/path_integration')
            / f'd-{self.dimensions}'
            / f's-{self.steps}'
            / strategy_str
            / strategy_hash
            / boundary_str
            / boundary_hash
            / f'm-{self.samples}'
            / f'r-{self.seed}'
            / 'dataset.pt'
        )

    def update_path(self):
        """Update the path based on current parameters"""
        self.path = self._generate_path()



if __name__ == '__main__':
    import yaml
    
    yaml_content = """
    dimensions: [2, 3]
    steps: 10
    strategy:
      name: LevyFlightStrategy
      params:
        alpha: 3.0
        momentum: [0.8, 0.9]
        step_size: 1.0
    boundary:
      name: PolygonBoundary
      params:
        n_sides: [4, 6]
        radius: 0.2
        rotation: pi/4
    samples: 64
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams(**config)
    
    print(f"Strategy: {p.strategy}")
    print(f"Boundary: {p.boundary}")
    print(f"\nMath expression 'pi/4' will be evaluated in factory")
    
    # Test with generate_param_combinations
    try:
        from projects.boolean_reservoir.code.utils.param_utils import generate_param_combinations
        p_list = generate_param_combinations(p)
        
        print(f"\nGenerated {len(p_list)} combinations")
        print(f"Expected: dimensions[2,3] × momentum[0.8,0.9] × n_sides[4,6] = 2×2×2 = 8")
        
        for i, combo in enumerate(p_list[:2]):
            print(f"\n--- Combination {i+1} ---")
            print(f"  dimensions: {combo.dimensions}")
            print(f"  strategy.params: {combo.strategy.params}")
            print(f"  boundary.params: {combo.boundary.params}")
            print(f"  strategy_obj: {combo.strategy_obj}")
            print(f"  boundary_obj: {combo.boundary_obj}")
    except ImportError as e:
        print(f"\nCould not test with generate_param_combinations: {e}")
