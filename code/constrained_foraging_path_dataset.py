from utils import set_seed
set_seed(42)

import torch
from torch.utils.data import Dataset, DataLoader
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path import random_walk, Boundary, NoBoundary, positions_to_p_v_pairs, generate_polygon_points, stretch_polygon, PolygonBoundary, LevyFlightStrategy 
from numpy import pi
from pathlib import Path
from math import floor

class ConstrainedForagingPathDataset(Dataset):
    def __init__(self, samples=100, n_steps=25, n_dimensions=2, strategy=random_walk, boundary: Boundary=NoBoundary, data_path='/data/test/dataset.pt', generate_data=False):
        self.data_path = Path(data_path)
        self.n_dimensions = n_dimensions
        self.boundary = boundary
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None
        
        if generate_data:
            self.data = self.generate_data(n_dimensions, samples, n_steps, strategy, boundary)
        elif self.data_path.exists():
            self.load_data()
        else:
            print(f'Cant find data at: {self.data_path} (or set generate_data flag)')

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

    def split_dataset(self, split=(0.6, 0.3, 0.1)):
        assert sum(split) == 1, "Split ratios must sum to 1."

        x, y = self.data['x'], self.data['y']
        idx = torch.randperm(x.size(0))

        train_end, dev_end = floor(split[0] * x.size(0)), floor((split[0] + split[1]) * x.size(0))

        self.data = {
            'x': x[idx[:train_end]],
            'y': y[idx[:train_end]],
            'x_dev': x[idx[train_end:dev_end]],
            'y_dev': y[idx[train_end:dev_end]],
            'x_test': x[idx[dev_end:]],
            'y_test': y[idx[dev_end:]],
        }

    def save_data(self):
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, self.data_path)
        print(f"Data saved to {self.data_path}")

    def load_data(self):
        self.data = torch.load(self.data_path, weights_only=True)
        print(f"Data loaded from {self.data_path}")
    
    def set_normalizer_x(self, normalizer_x):
        self.normalizer_x = normalizer_x

    def set_normalizer_y(self, normalizer_y):
        self.normalizer_y = normalizer_y

    def set_encoder_x(self, encoder_x):
        self.encoder_x = encoder_x

    def __len__(self):
        return self.data['x'].size(0)
    
    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        return x, y

    def normalize(self):
        self.data['x'] = self.normalizer_x(self.data['x'])
        self.data['y'] = self.normalizer_y(self.data['y'])
    
    def encode_x(self):
        self.data['x'] = self.encoder_x(self.data['x'])
    


if __name__ == '__main__':
    # Parameters: Path Integration
    n_dimensions = 2
    n_steps = 5
    strategy = LevyFlightStrategy(momentum=0.9, bias=0)

    # Make bounary of path traversal
    square = generate_polygon_points(4, 10, rotation=pi/4) 
    rectangle = stretch_polygon(square, 2, 1/2) 
    boundary = PolygonBoundary(points=rectangle)
    
    # Parameters: Input Layer
    encoding = 'binary'
    bits_per_feature = 3  # Number of bits per dimension
    n_features = 2  # Number of dimensions

    # Parameters: Output Layer
    output_size = 2  # Number of dimensions

    # Parameters: Other
    samples = 6
    batch_size = 2

    dataset = ConstrainedForagingPathDataset(samples=samples, n_steps=n_steps, n_dimensions=n_dimensions, strategy=strategy, boundary=boundary, generate_data=True)
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize()
    encoder = lambda x: float_array_to_boolean(x, bits=bits_per_feature, encoding_type=encoding)
    dataset.set_encoder_x(encoder)
    dataset.encode_x()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for x, y in data_loader:
        print('-'*50, x.shape)
        print(x.to(torch.int))
        print('-'*50, y.shape)
        print(y)
        break