from benchmarks.path_integration.constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from benchmarks.path_integration.constrained_foraging_path import generate_polygon_points, stretch_polygon, LevyFlightStrategy, PolygonBoundary, IntervalBoundary
from numpy import pi
from projects.boolean_reservoir.code.utils import set_seed

def generate_dataset_2D_levy_square():
    seed = 0
    set_seed(seed)

    # Parameters: Path Integration
    n_dimensions = 2
    n_steps = 5
    strategy = LevyFlightStrategy(dim=n_dimensions, alpha=3, momentum=0.9, bias=0)
    square = generate_polygon_points(4, .1, rotation=pi/4) 
    boundary = PolygonBoundary(points=square)

    # Parameters: Dataset
    samples = 10000
    data_path = f'data/path_integration/d-{n_dimensions}/s-{strategy}/b-{boundary}/n-{n_steps}/m-{samples}/r-{seed}/dataset.pt'

    dataset = ConstrainedForagingPathDataset(samples=samples, n_steps=n_steps, n_dimensions=n_dimensions, strategy=strategy, boundary=boundary, data_path=data_path, generate_data=True)
    dataset.save_data()

def generate_dataset_1D_levy_interval():
    seed = 0
    set_seed(seed)

    # Parameters: Path Integration
    n_dimensions = 1
    n_steps = 5
    strategy = LevyFlightStrategy(dim=n_dimensions, alpha=3, momentum=0.9, bias=0)
    boundary = IntervalBoundary([-.1, .1]) 

    # Parameters: Dataset
    samples = 10000
    data_path = f'data/path_integration/d-{n_dimensions}/s-{strategy}/b-{boundary}/n-{n_steps}/m-{samples}/r-{seed}/dataset.pt'

    dataset = ConstrainedForagingPathDataset(samples=samples, n_steps=n_steps, n_dimensions=n_dimensions, strategy=strategy, boundary=boundary, data_path=data_path, generate_data=True)
    dataset.save_data()


if __name__ == '__main__':
    generate_dataset_1D_levy_interval()
    generate_dataset_2D_levy_square()
   

   