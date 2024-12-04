from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from constrained_foraging_path import generate_polygon_points, stretch_polygon, LevyFlightStrategy, PolygonBoundary
from numpy import pi

def generate_dataset_levy_square_no_momentum():
    # Parameters: Dataset
    data_path = '/data/levy_walk/25_steps/square_boundary/dataset.pt'
    samples = 10000

    # Parameters: Path Integration
    n_dimensions = 2
    n_steps = 8 
    strategy = LevyFlightStrategy(momentum=0, bias=0)
    square = generate_polygon_points(4, 10, rotation=pi/4) 
    boundary = PolygonBoundary(points=square)

    dataset = ConstrainedForagingPathDataset(samples=samples, n_steps=n_steps, n_dimensions=n_dimensions, strategy=strategy, boundary=boundary, data_path=data_path, generate_data=True)
    dataset.save_data()

def generate_dataset_levy_square():
    # Parameters: Dataset
    data_path = '/data/levy_walk/25_steps/square_boundary/dataset.pt'
    samples = 10000

    # Parameters: Path Integration
    n_dimensions = 2
    n_steps = 25
    strategy = LevyFlightStrategy(momentum=0.9, bias=0)
    square = generate_polygon_points(4, 10, rotation=pi/4) 
    boundary = PolygonBoundary(points=square)

    dataset = ConstrainedForagingPathDataset(samples=samples, n_steps=n_steps, n_dimensions=n_dimensions, strategy=strategy, boundary=boundary, data_path=data_path, generate_data=True)
    dataset.save_data()

if __name__ == '__main__':
    # generate_dataset_levy_square()
    generate_dataset_levy_square_no_momentum()
   