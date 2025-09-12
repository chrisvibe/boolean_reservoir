import torch
from torch.utils.data import DataLoader
from projects.boolean_reservoir.code.utils import set_seed
from projects.boolean_reservoir.code.encoding import float_array_to_boolean, min_max_normalization
from projects.boolean_reservoir.code.parameters import InputParams
from benchmarks.path_integration.constrained_foraging_path import random_walk, positions_to_p_v_pairs
from benchmarks.path_integration.visualizations import plot_random_walk
from benchmarks.path_integration.parameters import PathIntegrationDatasetParams
from benchmarks.utils.base_dataset import BaseDataset
import yaml

class ConstrainedForagingPathDataset(BaseDataset):
    def __init__(self, D: PathIntegrationDatasetParams):
        self.D = D
        set_seed(D.seed)
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None
        
        if D.path.exists() and not D.generate_data:
            self.load_data()
        else:
            self.data = self.generate_data(D.dimensions, D.samples, D.steps, D.strategy, D.boundary)
            self.save_data()

    @staticmethod
    def generate_data(dimensions, samples, n_steps, strategy, boundary):
        data_x = []
        data_y = []
        
        # Generate the data
        # Note velocity is zero if the next step would be outside the boundary, as the position is unchanged
        for _ in range(samples):
            p = random_walk(dimensions, n_steps, strategy, boundary)
            p, v = positions_to_p_v_pairs(p)
            data_x.append(torch.tensor(v, dtype=torch.float))
            data_y.append(torch.tensor(p[-1], dtype=torch.float))

        return {
            'x': torch.stack(data_x),
            'y': torch.stack(data_y),
        }


if __name__ == '__main__':
    # demo of dataset init 
    yaml_content = """
    dimensions: 2
    steps: 500
    strategy_config:
        type: LevyFlightStrategy
        params:
            alpha: 3
            momentum: 0.9
    boundary_config:
        type: PolygonBoundary
        params:
            n_sides: 4
            radius: 1
            rotation: pi/4 
            stretch_x: 2 
            stretch_y: 1/2
        split:
            train: 0.4
            dev: 0.3
            test: 0.3
    samples: 10000
    seed: 0
    """
    config = yaml.safe_load(yaml_content)
    D = PathIntegrationDatasetParams(**config)
    positions = random_walk(D.dimensions, D.steps, D.strategy, D.boundary)
    plot_random_walk('/out', positions, D.strategy, D.boundary, file_prepend='demo_path')


    from projects.boolean_reservoir.code.parameters import load_yaml_config 
    P = load_yaml_config('config/path_integration/test/1D/verification_model.yaml')
    D = P.D
    positions = random_walk(D.dimensions, D.steps, D.strategy, D.boundary)
    plot_random_walk('/out', positions, D.strategy, D.boundary, file_prepend='test_verification_model')
    
    from projects.boolean_reservoir.code.parameters import load_yaml_config 
    P = load_yaml_config('config/path_integration/test/2D/verification_model.yaml')
    D = P.D
    positions = random_walk(D.dimensions, D.steps, D.strategy, D.boundary)
    plot_random_walk('/out', positions, D.strategy, D.boundary, file_prepend='test_verification_model')