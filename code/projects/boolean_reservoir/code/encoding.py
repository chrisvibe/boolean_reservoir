import torch
from projects.boolean_reservoir.code.parameters import InputParams


def float_array_to_boolean(values, I:InputParams):
    '''
    accepts a point tensor with dimensions as columns and points are rows
    returns a boolean encoding of this

    1. assume input is normalized [0, 1] as float vals
    2. rescaling based on bit representation
    3. convert to bits
    '''
    bin_values = None
    assert torch.is_floating_point(values)
    assert torch.max(values) <= 1
    assert torch.min(values) >= 0
    if I.encoding == 'base2':
        bin_values = dec2bin(values, I.resolution)
        bin_values = bin_values.repeat(1, 1, 1, I.redundancy)
    elif I.encoding == 'tally':
        bin_values = dec2tally(values, I.resolution)
        bin_values = bin_values.repeat(1, 1, 1, I.redundancy)
    elif I.encoding == 'binary_embedding':
        encoder = BinaryEmbedding(b=I.resolution, n=I.redundancy)
        bin_values = encoder.encode(values)
    else:
        raise ValueError(f"encoding {I.encoding} is not an option!")
    if I.interleaving:
        bin_values = interleave_features(bin_values, group_size=I.interleaving) # Note: no effect at 1D with grouping=1
    return bin_values.to(torch.uint8)

def dec2bin(x, bits):
    '''
    Convert decimal to boolean array representation with a fixed number of bits
    Assume input is normalized [0, 1] as float vals
    '''
    x = (x * (2 ** bits - 1)).to(torch.int64)
    mask = (2 ** torch.arange(bits - 1, -1, -1, device=x.device)).to(torch.int64)
    x = (x.unsqueeze(-1).bitwise_and(mask)).ne(0)
    return x

def dec2tally(x, bits):
    '''
    Convert decimal to boolean array representation with a fixed number of bits
    Assume input is normalized [0, 1] as float vals
    '''
    d = (bits * x).round().unsqueeze(-1)
    bit_range = torch.arange(bits, device=x.device).float()
    return bit_range.lt(d)

def bin2dec(x, bits, small_endian=False):
    if small_endian:
        mask = 2 ** torch.arange(bits, device=x.device)
    else:
        mask = 2 ** torch.arange(bits - 1, -1, -1, device=x.device)
    mask.to(x.device, x.dtype)
    vals = torch.sum(mask * x, -1).long()
    return vals / (2**bits - 1)

def interleave_features(x, group_size=1):
    # interleave between dimension d (dim 2) mxsxdxb
    # group_size is grouping when interleaving f.ex N=2 → [012, 345] → [(01)(34)(25)]
    shape = x.shape
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, group_size)
    x = torch.transpose(x, 2, 3).reshape(shape)
    return x


def min_max_normalization(data):
    data = data.to(torch.float)
    min_ = data.amin(dim=(0, 1), keepdim=True)  # Minimum along the samples and steps dimensions
    max_ = data.amax(dim=(0, 1), keepdim=True)  # Maximum along the samples and steps dimensions
    return (data - min_) / (max_ - min_)

def standard_normalization(data):
    data = data.to(torch.float)
    means = data.mean(dim=(0, 1), keepdim=True)  # Mean along the samples and steps dimensions
    stds = data.std(dim=(0, 1), keepdim=True)    # Standard deviation along the samples and steps dimensions
    return (data - means) / stds


class BinaryEmbedding:
    def __init__(self, b, n):
        """
        Initializes the BinaryEmbeddingEncoder with a fixed set of random boolean key vectors.

        Args:
            b (int): Bit resolution of data (2^b values).
            n (int): The expansion factor.
        """
        self.b = None
        self.n = None 
        self.random_boolean_keys = None
        self.set_random_boolean_keys(b, n)

    def encode(self, data):
        """Encode float data using the Binary Embedding method with the pre-generated random key vectors.
        
        Args:
            data: A normalized [0, 1] float tensor of shape (batch_size, sequence_length, n_inputs, bit_resolution).
                
        Returns:
            A binary encoded tensor with self-similar properties of shape (batch_size, sequence_length, n_inputs, n*bit_resolution).
        """
        batch_size, seq_length, n_inputs = data.size()
        
        # Convert normalized data to binary representation
        binary_tensors = dec2bin(data, self.b).to(torch.uint8)
        
        # Expand dimensions for broadcasting
        binary_tensors = binary_tensors.unsqueeze(3).expand(-1, -1, -1, self.n, -1)  # Output shape: (batch_size, seq_length, n_inputs, self.n, self.b)
    
        # Perform XOR operations
        encoded_tensors = binary_tensors ^ self.random_boolean_keys

        # Flatten encoded tensors per value
        encoded_tensors = encoded_tensors.view(batch_size, seq_length, n_inputs, -1)

        return encoded_tensors

    def set_random_boolean_keys(self, new_n=None, new_b=None):
        """Set a new set of random vectors. If new_n or new_b is None, current values are used.
        
        Args:
            new_n (int, optional): New expansion factor. Defaults to None.
            new_b (int, optional): New bit resolution. Defaults to None.
        """
        self.n = new_n if new_n is not None else self.n
        self.b = new_b if new_b is not None else self.b
        self.random_boolean_keys = torch.randint(0, 2, (1, 1, 1, self.n, self.b)).to(torch.uint8)


 
if __name__ == '__main__':
    bits = 3
    p = torch.randint(0, 2**bits, (3, 2))
    print("Original array:")
    print(p.numpy())

    print("Normalized array:")
    p = min_max_normalization(p)
    print(p.numpy())

    print("Normalized array times boolean max:")
    print((p * (2**bits - 1)).numpy())

    print("Boolean representation:")
    I = InputParams(bits_per_feature=bits, encoding='base2', n_inputs=p.shape[-1])
    boolean_representation = float_array_to_boolean(p, I)
    print(boolean_representation.to(torch.int).numpy())

    ##########################################################
    encoder = BinaryEmbedding(b=4, n=3)
    batch_size = 2
    s = 3
    n_inputs = 2
    x = torch.randint(0, 10, (batch_size, s, n_inputs,), dtype=torch.float) / 10

    encoded_tensors = encoder.encode(x)
    print("Input Tensor:\n", x.numpy())
    print("Encoded Tensor Shape:", encoded_tensors.shape)
    print("Encoded Tensor:\n", encoded_tensors.to(torch.int).numpy())