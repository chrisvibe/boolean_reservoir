import torch
from projects.boolean_reservoir.code.utils.utils import set_seed
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
            self.data = self.generate_data(D.dimensions, D.samples, D.steps, D.strategy_obj, D.boundary_obj)
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
    strategy:
        name: LevyFlightStrategy
        params:
            alpha: 3
            momentum: 0.9
    boundary_config:
        name: PolygonBoundary
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
    # config = yaml.safe_load(yaml_content)
    # D = PathIntegrationDatasetParams(**config)
    # positions = random_walk(D.dimensions, D.steps, D.strategy_obj, D.boundary_obj)
    # plot_random_walk('/out', positions, D.strategy, D.boundary, file_prepend='demo_path')

    from projects.boolean_reservoir.code.parameters import load_yaml_config 
    from projects.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
    P = load_yaml_config('projects/path_integration/test/config/1D/single_run/test_model.yaml')
    P = generate_param_combinations(P)[0]
    D = P.D
    positions = random_walk(D.dimensions, D.steps, D.strategy_obj, D.boundary_obj)
    plot_random_walk('/out', positions, D.strategy_obj, D.boundary_obj, file_prepend=P.L.out_path.name)