import torch
from torch.utils.data import Dataset
from pathlib import Path
from math import floor
import numpy as np
from boolean_reservoir.utils import set_seed

class TemporalDatasetBase(Dataset):
    def __init__(self, task, samples=100, stream_length=10, window_size=4, tao=1, data_path='/data/test/dataset.pt', generate_data=False):
        self.data_path = Path(data_path)
        
        if self.data_path.exists() and not generate_data:
            self.load_data()
        else:
            self.data = self.generate_data(task, samples, stream_length, window_size, tao)

    def generate_data(self, task, samples, stream_length, tao, window_size):
        data_x = []
        data_y = []
        
        # Generate the data
        for _ in range(samples):
            arr = self.gen_boolean_array(stream_length)
            label = task(arr, tao, window_size)
            data_x.append(torch.tensor(arr, dtype=torch.uint8).unsqueeze(0).unsqueeze(0))
            data_y.append(torch.tensor(label, dtype=torch.float).unsqueeze(0))

        return {
            'x': torch.stack(data_x),
            'y': torch.stack(data_y),
        }

    def split_dataset(self, split=(0.8, 0.1, 0.1)):
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
    
    def __len__(self):
        return self.data['x'].size(0)
    
    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        return x, y

class BooleanDataGenerator():
    @staticmethod
    def gen_boolean_array(n):
        return np.random.randint(0, 2, size=n, dtype=bool)

class TemporalDensity:
    @staticmethod
    def temporal_density(u, tau, n):
        """
        Determines whether an odd number of bits τ + n to τ time steps in the past have more "1" values than "0."
        Args:
            u (str Input stream (string of "0" and "1").
            tau (int): Delay.
            n (int): Window size (must be odd).
        Returns:
            tuple: (Input stream, density label)
        """
        T = len(u)
        if tau + n > T:
            raise ValueError("tau + n must be less than or equal to the length of the input stream")

        if n % 2 == 0:
            raise ValueError("Window size n must be odd")

        # Extract the relevant window of bits
        window = u[T-tau-n:T-tau]

        # Count the number of "1" values
        count_ones = window.sum()

        # Determine if there are more "1" values than "0"
        density = 2 * count_ones > n

        return density


class TemporalParity:
    @staticmethod
    def temporal_parity(u, tau, n):
        """
        Determines if n bits τ + n to τ time steps in the past have an odd number of "1" values.
        Args:
            u (str): Input stream (string of "0" and "1").
            tau (int): Delay.
            n (int): Window size.
        Returns:
            tuple: (Input stream, parity label)
        """
        T = len(u)
        if tau + n > T:
            raise ValueError("tau + n must be less than or equal to the length of the input stream")

        # Extract the relevant window of bits
        window = u[T-tau-n:T-tau]

        # Count the number of "1" values
        count_ones = window.sum()

        # Determine if the number of "1" values is odd
        parity = count_ones % 2 != 0

        return parity


class TemporalDensityDataset(TemporalDatasetBase, TemporalDensity, BooleanDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(self.temporal_density, **kwargs)


class TemporalParityDataset(TemporalDatasetBase, TemporalParity, BooleanDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(self.temporal_parity, **kwargs)


def generate_dataset_temporal_density():
    set_seed(0)

    # Parameters: Dataset
    data_path = '/data/temporal/density/dataset.pt'
    samples = 6000

    # Parameters: Temporal density 
    stream_length=10
    window_size=4
    tao=1

    dataset = TemporalDensityDataset(samples=samples, stream_length=stream_length, window_size=window_size, tao=tao, data_path=data_path, generate_data=True)
    dataset.save_data()

def generate_dataset_temporal_parity():
    set_seed(0)

    # Parameters: Dataset
    data_path = '/data/temporal/parity/dataset.pt'
    samples = 6000

    # Parameters: Temporal parity 
    stream_length=10
    window_size=4
    tao=1

    dataset = TemporalParityDataset(samples=samples, stream_length=stream_length, window_size=window_size, tao=tao, data_path=data_path, generate_data=True)
    dataset.save_data()
   

if __name__ == '__main__':
    input_stream = BooleanDataGenerator.gen_boolean_array(10) 
    tau_value = 1
    n_value = 3
    parity = TemporalParity.temporal_parity(input_stream, tau_value, n_value)
    print(f"Parity  Task - Stream: {input_stream.astype(np.uint8)}, Parity : {parity}")

    density = TemporalDensity.temporal_density(input_stream, tau_value, n_value)
    print(f"Density Task - Stream: {input_stream.astype(np.uint8)}, Density: {density}")

    generate_dataset_temporal_density()
    generate_dataset_temporal_parity()
