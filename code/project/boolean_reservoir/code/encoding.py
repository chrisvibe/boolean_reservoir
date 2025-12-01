import torch
from project.boolean_reservoir.code.parameter import InputParams


class BooleanEncoder:
    def __init__(self, I):
        self.I = I

    def __call__(self, x):
        return float_array_to_boolean(x, self.I)


def float_array_to_boolean(values, I: InputParams):
    '''
    accepts a point tensor with dimensions as columns and points are rows
    returns a boolean encoding of this

    1. assume input is normalized [0, 1] as float vals
    2. rescaling based on bit representation
    3. convert to bits
    4. optionally: add redundancy
    5. optionally: perform interleaving
    '''
    bin_values = None
    values = values.float()
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
        # Note: no effect at 1D with grouping=1
        bin_values = interleave_features(bin_values, group_size=I.interleaving)
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


def interleave_features_old(x, group_size=1):  # cant handle uneven b/group
    # for testing x = torch.arange(0, 10).reshape(2, -1).unsqueeze(0).unsqueeze(0)
    # interleave between dimension d mxsxdxb
    # group_size is grouping when interleaving f.ex N=1 → [[0, 1, 2], [3, 4, 5]] → [[0, 3, 1], [4, 2, 5)]
    # group_size is grouping when interleaving f.ex N=2 → [[0, 1, 2], [3, 4, 5]] → [[0, 1, 3], [4, 2, 5)]
    shape = x.shape
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, group_size)
    x = torch.transpose(x, 2, 3).reshape(shape)
    return x


def interleave_features(x, group_size=1):  # can handle uneven b/group
    # for testing x = torch.arange(0, 10).reshape(2, -1).unsqueeze(0).unsqueeze(0)
    shape = x.shape
    n, m, d, b = shape
    gs = group_size
    num_full = b // gs
    rem = b % gs
    parts = []
    for i in range(num_full):
        for j in range(d):
            part = x[:, :, j:j+1, i*gs: (i+1)*gs]
            parts.append(part)
    if rem > 0:
        for j in range(d):
            part = x[:, :, j:j+1, -rem:]
            parts.append(part)
    interleaved = torch.cat(parts, dim=-1)
    interleaved = interleaved.reshape(shape)
    return interleaved


def min_max_normalization(data):
    data = data.to(torch.float)
    # Minimum along the samples and steps dimensions
    min_ = data.amin(dim=(0, 1), keepdim=True)
    # Maximum along the samples and steps dimensions
    max_ = data.amax(dim=(0, 1), keepdim=True)
    return (data - min_) / (max_ - min_)


def standard_normalization(data):
    data = data.to(torch.float)
    # Mean along the samples and steps dimensions
    means = data.mean(dim=(0, 1), keepdim=True)
    # Standard deviation along the samples and steps dimensions
    stds = data.std(dim=(0, 1), keepdim=True)
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
            data: A normalized [0, 1] float tensor of shape (batch_size, sequence_length, features).

        Returns:
            A binary encoded tensor with self-similar properties of shape (batch_size, sequence_length, features, n*bit_resolution).
        """
        batch_size, seq_length, features = data.size()

        # Convert normalized data to binary representation
        binary_tensors = dec2bin(data, self.b).to(torch.uint8)

        # Expand dimensions for broadcasting
        # Output shape: (batch_size, seq_length, features, self.n, self.b)
        binary_tensors = binary_tensors.unsqueeze(
            3).expand(-1, -1, -1, self.n, -1)

        # Perform XOR operations
        encoded_tensors = binary_tensors ^ self.random_boolean_keys

        # Flatten encoded tensors per value
        encoded_tensors = encoded_tensors.view(
            batch_size, seq_length, features, -1)

        return encoded_tensors

    def set_random_boolean_keys(self, new_b=None, new_n=None):
        """Set a new set of random vectors. If new_n or new_b is None, current values are used.

        Args:
            new_b (int, optional): New bit resolution. Defaults to None.
            new_n (int, optional): New expansion factor. Defaults to None.
        """
        self.n = new_n if new_n is not None else self.n
        self.b = new_b if new_b is not None else self.b
        self.random_boolean_keys = torch.randint(
            0, 2, (1, 1, 1, self.n, self.b)).to(torch.uint8)


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
    I = InputParams(chunk_size=bits, encoding='base2', features=p.shape[-1])
    boolean_representation = float_array_to_boolean(p, I)
    print(boolean_representation.to(torch.int).numpy())

    ##########################################################
    encoder = BinaryEmbedding(b=4, n=3)
    batch_size = 2
    s = 3
    features = 2
    x = torch.randint(0, 10, (batch_size, s, features,),
                      dtype=torch.float) / 10

    encoded_tensors = encoder.encode(x)
    print("Input Tensor:\n", x.numpy())
    print("Encoded Tensor Shape:", encoded_tensors.shape)
    print("Encoded Tensor:\n", encoded_tensors.to(torch.int).numpy())
