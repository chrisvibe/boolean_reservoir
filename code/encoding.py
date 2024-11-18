import torch


def float_array_to_boolean(float_values, encoding_type='binary', bits=8):
    '''
    accepts a point tensor with dimensions as columns and points are rows
    returns a boolean encoding of this

    1. normalization
    2. rescaling based on bit representation
    3. convert to bits
    '''
    normalized_values = (float_values - torch.min(float_values)) / (torch.max(float_values) - torch.min(float_values))
    if encoding_type == 'binary':
        bin_values = dec2bin(normalized_values, bits)
    elif encoding_type == 'tally':
        bin_values = dec2tally(normalized_values, bits)
    else:
        raise ValueError(f"encoding {encoding_type} is not an option!")

    return bin_values

def dec2bin(x, bits):
    '''
    Convert decimal to boolean array representation with a fixed number of bits
    '''
    x = (x * (2 ** bits - 1)).to(torch.int64)
    mask = (2 ** torch.arange(bits - 1, -1, -1, device=x.device)).to(torch.int64)
    x = (x.unsqueeze(-1).bitwise_and(mask)).ne(0)
    return x

def dec2tally(x, bits):
    '''
    Convert decimal to boolean array representation with a fixed number of bits
    '''
    d = (bits * x).round().unsqueeze(-1)
    bit_range = torch.arange(bits, device=x.device).float()
    return bit_range.lt(d)

if __name__ == '__main__':
    p = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]], dtype=torch.float32)
    print("Original array:")
    print(p)

    # Convert coordinates to PyTorch tensor
    p_tensor = torch.tensor(p, dtype=torch.float32)

    # Convert coordinates to boolean
    boolean_representation = float_array_to_boolean(p_tensor, bits=3)
    print("Boolean representation:")
    print(boolean_representation.to(torch.int).numpy())