import torch
import numpy as np
from benchmark.temporal.parameters import TemporalDatasetParams
from project.boolean_reservoir.code.utils.utils import set_seed
from benchmark.utils.base_dataset import BaseDataset
from project.boolean_reservoir.code.encoding import dec2bin

class TemporalDatasetBase(BaseDataset):
    def __init__(self, D: TemporalDatasetParams, task):
        # bits i.e. 12: 101010111010
        # window (in the bit stream) i.e. 4: 10101011[1010]
        # delay (for the window) i.e. 1: 1010101[1101]0
        super().__init__(D)
        self.D = D
        set_seed(D.seed)
        self.task = task
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None
        
        if self.D.path.exists() and not D.generate_data:
            self.load_data()
        else:
            if D.bits < D.window + D.delay:
                print('Warning: bits < delay + window, overriding bits...')
                D.bits = D.window + D.delay
            raw_data = self.generate_data(D.samples, D.bits, D.delay, D.window)
            self.set_data(raw_data)
            self.save_data()
    
    @staticmethod
    def gen_integer_samples(samples, stream_length, sampling_mode):
        """Generate integer samples based on sampling mode."""
        if sampling_mode == 'exhaustive':
            # Guard against memory explosion
            max_possible = 2**stream_length
            if stream_length > 20:  # 2^20 = ~1M samples, reasonable limit
                raise ValueError(
                    f"Exhaustive mode with stream_length={stream_length} would generate "
                    f"{max_possible} samples, which is too large. Consider random mode or smaller stream_length."
                )
            
            # Generate all possible bit patterns (truth table)
            int_samples = np.array(list(range(max_possible)))
            # If we need more samples than exist, repeat
            if samples > len(int_samples):
                int_samples = np.tile(int_samples, samples // len(int_samples) + 1)
            int_samples = int_samples[:samples]
        else:  # 'random'
            # np.random.randint max value is np.iinfo(np.int64).max
            if stream_length > 63:
                raise ValueError(
                    f"stream_length={stream_length} is too large for random integer generation. "
                    f"Maximum is 63 bits."
                )
            # Generate random integers (repetition allowed)
            int_samples = np.random.randint(0, 2**stream_length, size=samples)
        
        return int_samples
    
    def generate_data(self, samples, stream_length, delay, window):
        """Generate dataset from integer samples."""
        # Generate all integers at once: samples * dimensions
        total_streams = samples * self.D.dimensions
        int_samples = self.gen_integer_samples(total_streams, stream_length, self.D.sampling_mode)
        
        # Convert to torch for dec2bin
        int_samples = torch.from_numpy(int_samples)
        
        # Convert to binary arrays
        arrays = [dec2bin(int_val, stream_length) for int_val in int_samples]
        
        # Compute labels (arrays are already torch tensors)
        labels = [self.task(arr, window, delay) for arr in arrays]
        
        # Stack and reshape
        x = torch.stack(arrays).reshape(samples, self.D.dimensions, stream_length)
        labels = torch.tensor(labels).reshape(samples, self.D.dimensions)
        
        # Final shape: m x 1 x d x b
        x = x.unsqueeze(1)  # m x 1 x d x b
        y = labels.float()  # m x d
        
        return {'x': x, 'y': y}
    
    @staticmethod
    def gen_boolean_array(n):
        return np.random.randint(0, 2, size=n, dtype=bool)
    

class TemporalDensityDataset(TemporalDatasetBase):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, p, self.density_task)

    @staticmethod
    def density_task(bits, window, delay):
        """
        Determines whether an sub string of bits at position bits[-window-delay:-delay] in the past have more "1" values than "0."
        Args:
            bits (str Input stream (string of "0" and "1").
            window (int): selected window size.
            delay (int): shift window right to left.
        Returns:
            tuple: (Input stream, density label)
        """
        b = len(bits)
        if delay + window > b:
            raise ValueError("delay + window must be less than or equal to the bits in the input stream")

        # Extract the relevant window of bits
        window_bits = bits[b-window-delay:b-delay] 

        # Count the number of "1" values
        count_ones = window_bits.sum()

        # Determine if there are more "1" values than "0"
        density = 2 * count_ones > window

        return density


class TemporalParityDataset(TemporalDatasetBase):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, p, self.parity_task)

    @staticmethod
    def parity_task(bits, window, delay):
        """
        Determines whether an sub string of bits at position bits[-window-delay:-delay] in the past have a odd number "1" values
        Args:
            bits (str Input stream (string of "0" and "1").
            window (int): selected window size.
            delay (int): shift window right to left.
        Returns:
            tuple: (Input stream, density label)
        """
        b = len(bits)
        if delay + window > b:
            raise ValueError("delay + window must be less than or equal to the bits in the input stream")

        # Extract the relevant window of bits
        window_bits = bits[b-window-delay:b-delay] 

        # Count the number of "1" values
        count_ones = window_bits.sum()

        # Determine if the number of "1" values is odd
        parity = count_ones % 2 != 0

        return parity