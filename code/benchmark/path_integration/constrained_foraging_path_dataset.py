import torch
import numpy as np
from project.boolean_reservoir.code.utils.utils import set_seed
from benchmark.path_integration.constrained_foraging_path import generate_dual_trajectory, random_walk, to_polar, to_cartesian
from benchmark.path_integration.visualization import plot_random_walk
from benchmark.path_integration.parameter import PathIntegrationDatasetParams
from benchmark.utils.base_dataset import BaseDataset
import yaml

class ConstrainedForagingPathDataset(BaseDataset):
    def __init__(self, D: PathIntegrationDatasetParams):
        super().__init__(D)
        self.D = D
        set_seed(D.seed)
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None
        
        if D.path.exists() and not D.generate_data:
            self.load_data()
        else:
            raw_data = self.generate_data(D.dimensions, D.samples, D.steps, D.strategy_obj, D.boundary_obj, D.coordinate, D.reset, D.mode, D.output_coordinate)
            self.set_data(raw_data)
            self.save_data()
        if D.shuffle:
            self.shuffle_data()

    @staticmethod
    def generate_data(dimensions, samples, n_steps, strategy, boundary, coordinate_system='cartesian', reset=True, mode='displacement', output_coordinate='cartesian'):
        data_x = []
        data_y = []
        origin = None

        for _ in range(samples):
            d = random_walk(dimensions, n_steps, strategy, boundary, origin=origin)
            positions = d['positions']
            p_final = positions[-1]

            if not reset:
                raise NotImplementedError("reset=False is not implemented yet. y label is coupled with boundary and this would cause drift")
                origin = p_final.copy()  # cartesian, before any coord conversion

            if mode == 'acceleration':
                x_data = d['a_net']
            elif mode == 'velocity':
                x_data = d['velocities']
            elif mode == 'displacement':
              # sum(x) = p_final - origin always
              x_data = np.diff(positions, axis=0)
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be one of 'acceleration', 'velocity', or 'displacement'.")

            if coordinate_system == 'polar':
                x_data = to_polar(x_data)

            # TODO: add flexibility to predict non-cartesian output?
            if output_coordinate == 'polar':
                p_final = to_polar(p_final)

            data_x.append(torch.tensor(x_data, dtype=torch.float))
            data_y.append(torch.tensor(p_final, dtype=torch.float))
        
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

    from project.boolean_reservoir.code.parameter import load_yaml_config 
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
    P = load_yaml_config('project/path_integration/test/config/1D/single_run/test_model.yaml')
    P = generate_param_combinations(P)[0]
    D = P.D
    d = generate_dual_trajectory(D.dimensions, D.steps, D.strategy_obj, D.boundary_obj)
    plot_random_walk('/out', d['positions_actual'], D.strategy_obj, D.boundary_obj, dual=d, file_prepend=P.L.out_path.name)

    dataset = ConstrainedForagingPathDataset(D)