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
    p = PathIntegrationDatasetParams.from_yaml(config)
    positions = random_walk(p.dimensions, p.steps, p.strategy, p.boundary)
    plot_random_walk('/out', positions, p.strategy, p.boundary, file_prepend='demo_path')

    dataset = ConstrainedForagingPathDataset(p)
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize()
    
    # Parameters: Input Layer
    encoding = 'base2'
    bits_per_feature = 3  # Number of bits per dimension
    n_features = 2  # Number of dimensions
    I = InputParams(bits_per_feature=bits_per_feature, encoding=encoding, features=n_features)

    encoder = lambda x: float_array_to_boolean(x, I)
    dataset.set_encoder_x(encoder)
    dataset.encode_x()
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for x, y in data_loader:
        print('-'*50, x.shape)
        print(x.to(torch.int))
        print('-'*50, y.shape)
        print(y)
        break

    ################################################

    # generate_dataset_2D_levy_square
    yaml_content = """
    dimensions: 2
    steps: 5
    strategy_config:
        type: LevyFlightStrategy
        params:
            alpha: 3
            momentum: 0.9
    boundary_config:
        type: PolygonBoundary
        params:
            n_sides: 4
            radius: 0.1
            rotation: pi/4 
        split:
            train: 0.4
            dev: 0.3
            test: 0.3
    samples: 10000
    seed: 0
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams.from_yaml(config)
    positions = random_walk(p.dimensions, p.steps, p.strategy, p.boundary)
    plot_random_walk('/out', positions, p.strategy, p.boundary, file_prepend='generate_dataset_2D_levy_square')
    

    # generate_dataset_1D_levy_interval
    yaml_content = """
    dimensions: 1
    steps: 5
    strategy_config:
        type: LevyFlightStrategy
        params:
            alpha: 3
            momentum: 0.9
    boundary_config:
        type: IntervalBoundary
        params:
            radius: .2
    split:
        train: 0.4
        dev: 0.3
        test: 0.3
    samples: 10000
    seed: 0
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams.from_yaml(config)
    positions = random_walk(p.dimensions, p.steps, p.strategy, p.boundary)
    plot_random_walk('/out', positions, p.strategy, p.boundary, file_prepend='generate_dataset_1D_levy_interval')

    # test verification model 2D 
    yaml_content = """
    dimensions: 2
    steps: 5
    strategy_config:
        type: LevyFlightStrategy
        params:
            alpha: 3
            momentum: 0.9
    boundary_config:
        type: PolygonBoundary
        params:
            n_sides: 4
            radius: 0.5
            rotation: pi/4 
    split:
        train: 0.4
        dev: 0.3
        test: 0.3
    samples: 10000
    seed: 0
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams.from_yaml(config)
    positions = random_walk(p.dimensions, p.steps, p.strategy, p.boundary)
    plot_random_walk('/out', positions, p.strategy, p.boundary, file_prepend='test_verification_model')
    


