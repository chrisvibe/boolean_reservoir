import torch
import numpy as np
from benchmarks.temporal.parameters import TemporalDatasetParams
from projects.boolean_reservoir.code.utils.utils import set_seed
from benchmarks.utils.base_dataset import BaseDataset

class TemporalDatasetBase(BaseDataset):
    def __init__(self, task, D: TemporalDatasetParams):
        # bit_stream_length i.e. 12: 101010111010
        # window size (in the bit stream) i.e. 4: 10101011[1010]
        # tao is the delay (for the window) i.e. 1: 1010101[1101]0
        self.D = D
        set_seed(D.seed)
        self.task = task
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None
        
        if self.D.path.exists() and not D.generate_data:
            self.load_data()
        else:
            if D.bit_stream_length < D.window_size + D.tao:
                print('Warning: bit_stream_length < tao + window_size, overriding bit_stream_length...')
                D.bit_stream_length = D.window_size + D.tao
            self.data = self.generate_data(D.samples, D.bit_stream_length, D.tao, D.window_size)
            self.save_data()

    def generate_data(self, samples, stream_length, tao, window_size):
        data_x = []
        data_y = []

        # Generate the data
        for _ in range(samples):
            arr = self.gen_boolean_array(stream_length)
            label = self.task(arr, tao, window_size)
            data_x.append(torch.tensor(arr, dtype=torch.uint8).unsqueeze(-1).unsqueeze(-1)) # s, f, b
            data_y.append(torch.tensor(label, dtype=torch.float).unsqueeze(0))

        return {
            'x': torch.stack(data_x),
            'y': torch.stack(data_y),
        }
   
    @staticmethod
    def gen_boolean_array(n):
        return np.random.randint(0, 2, size=n, dtype=bool)
    
    def demo(self, iterations=5):
        print(f"\n=== {self.D.task.upper()} TASK DEMO ===")
        for i in range(iterations):
            input_stream = self.gen_boolean_array(10)
            colored_stream = color_the_stream(input_stream, self.D.tao, self.D.window_size)
            result = self.task(input_stream, self.D.tao, self.D.window_size)

            if self.D.task.lower() == "density":
                result_text = "more 1's than 0's" if result else "more 0's than 1's"
            elif self.D.task.lower() == "parity":
                result_text = "odd number of 1's" if result else "even number of 1's"
            else:
                print(f"Unknown task: {self.D.task}. Available tasks: 'density', 'parity'")
                return
            print(f"Run {i+1} - Stream: {colored_stream} -> {result_text}")


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
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_density",
        task="density",
        bit_stream_length=10,
        window_size=5,
        tao=2,
    )
    dataset = TemporalDensityDataset(D)
    dataset.demo()

    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_parity",
        task="parity",
        bit_stream_length=10,
        window_size=5,
        tao=2,
    )
    dataset = TemporalParityDataset(D)
    dataset.demo()


