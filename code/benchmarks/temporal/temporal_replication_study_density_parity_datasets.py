import torch
from torch.utils.data import Dataset
from pathlib import Path
from math import floor
import numpy as np
from projects.boolean_reservoir.code.utils import set_seed
from benchmarks.temporal.parameters import TemporalDatasetParams

class TemporalDatasetBase(Dataset):
    def __init__(self, task, p: TemporalDatasetParams):
        # bit_stream_length i.e. 12: 101010111010
        # window size (in the bit stream) i.e. 4: 10101011[1010]
        # tao is the delay (for the window) i.e. 1: 1010101[1101]0
        set_seed(p.seed)
        self.data_path = Path(p.path)
        
        if self.data_path.exists() and not p.generate_data:
            self.load_data()
        else:
            if p.bit_stream_length < p.window_size + p.tao:
                print('Warning: bit_stream_length < tao + window_size, overriding bit_stream_length...')
                p.bit_stream_length = p.window_size + p.tao
            self.data = self.generate_data(task, p.samples, p.bit_stream_length, p.tao, p.window_size)
            self.save_data()
        self.split = p.split

    def generate_data(self, task, samples, stream_length, tao, window_size):
        data_x = []
        data_y = []
        
        # Generate the data
        for _ in range(samples):
            arr = self.gen_boolean_array(stream_length)
            label = task(arr, tao, window_size)
            data_x.append(torch.tensor(arr, dtype=torch.uint8).unsqueeze(-1).unsqueeze(-1)) # s, f, b
            data_y.append(torch.tensor(label, dtype=torch.float).unsqueeze(0))

        return {
            'x': torch.stack(data_x),
            'y': torch.stack(data_y),
        }

    def split_dataset(self, split=[0.8, 0.1, 0.1]):
        split_train = split[0] if self.split is None else self.split.train
        split_dev = split[1] if self.split is None else self.split.dev
        split_test = split[2] if self.split is None else self.split.test
        assert float(sum((split_train, split_dev, split_test))) == 1.0, "Split ratios must sum to 1."
        x, y = self.data['x'], self.data['y']
        idx = torch.randperm(x.size(0))

        train_end, dev_end = floor(split_train * x.size(0)), floor((split_train + split_dev) * x.size(0))

        self.data = {
            'x': x[idx[:train_end]],
            'y': y[idx[:train_end]],
            'x_dev': x[idx[train_end:dev_end]],
            'y_dev': y[idx[train_end:dev_end]],
            'x_test': x[idx[dev_end:]],
            'y_test': y[idx[dev_end:]],
        }

    def to(self, device):
        for key in self.data.keys():
            self.data[key] = self.data[key].to(device)
        return self
    
    def save_data(self):
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, self.data_path)

    def load_data(self):
        self.data = torch.load(self.data_path, weights_only=True)
    
    def __len__(self):
        return self.data['x'].size(0)
    
    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        return x, y
    
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


class TemporalDensityDataset(TemporalDatasetBase, TemporalDensity):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, self.temporal_density, p)


class TemporalParityDataset(TemporalDatasetBase, TemporalParity):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, self.temporal_parity, p)


def color_the_stream(input_stream, tau, n):
    # Prepare the stream for printing with the window in red
    start_index = len(input_stream) - n - tau
    end_index = len(input_stream) - tau
    colored_stream = ""
    for i in range(len(input_stream)):
        if start_index <= i < end_index:
            colored_stream += f"\033[91m{input_stream[i].astype(np.uint8)}\033[0m"  # Red color
        else:
            colored_stream += f"{input_stream[i].astype(np.uint8)}"
    return colored_stream


if __name__ == '__main__':
    input_stream = TemporalDatasetBase.gen_boolean_array(10) 
    tau_value = 1
    n_value = 5
    colored_stream = color_the_stream(input_stream, tau_value, n_value)

    density = TemporalDensity.temporal_density(input_stream, tau_value, n_value)
    print(f"Density Task - Stream: {colored_stream}: {"more 1's than 0's" if density else "more 0's than 1's"}")

    parity = TemporalParity.temporal_parity(input_stream, tau_value, n_value)
    print(f"Parity  Task - Stream: {colored_stream}: {"odd number of 1's" if parity else "even number of 1's"}")
