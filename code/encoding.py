import torch


def float_array_to_boolean(values, encoding_type='binary', bits=8):
    '''
    accepts a point tensor with dimensions as columns and points are rows
    returns a boolean encoding of this

    1. assume input is normalized [0, 1] as float vals
    2. rescaling based on bit representation
    3. convert to bits
    '''
    assert torch.is_floating_point(values)
    assert torch.max(values) <= 1
    assert torch.min(values) >= 0
    if encoding_type == 'binary':
        bin_values = dec2bin(values, bits)
    elif encoding_type == 'tally':
        bin_values = dec2tally(values, bits)
    else:
        raise ValueError(f"encoding {encoding_type} is not an option!")

    return bin_values.to(torch.bool)

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
    return torch.sum(mask * x, -1).long()

# def min_max_normalization(data): # TODO problem here!!!!!!!!!!!!! Bug
#     data = data.to(torch.float)
#     min_ = data.min(axis=0).values
#     max_ = data.max(axis=0).values
    # return (data - min_) / (max_ - min_)

# def min_max_normalization(data): # TODO try just doing a random normalization here to see if problems persist...
#     data = data.to(torch.float)
#     min_ = data[10:11, 0:1] - 0.3
#     max_ = data[13:14, 0:1] + 0.3
#     return (data - min_) / (max_ - min_)

def min_max_normalization(data):
    data = data.to(torch.float)
    min_ = data.amin(dim=(0, 1), keepdim=True)  # Minimum along the samples and steps dimensions
    max_ = data.amax(dim=(0, 1), keepdim=True)  # Maximum along the samples and steps dimensions
    return (data - min_) / (max_ - min_)

# def standard_normalization(data): # TODO this probably has a bug, see bug above...
#     data = data.to(torch.float)
#     means = data.mean(axis=0)
#     stds = data.std(axis=0)
#     return (data - means) / stds

def standard_normalization(data):
    data = data.to(torch.float)
    means = data.mean(dim=(0, 1), keepdim=True)  # Mean along the samples and steps dimensions
    stds = data.std(dim=(0, 1), keepdim=True)    # Standard deviation along the samples and steps dimensions
    return (data - means) / stds

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
    boolean_representation = float_array_to_boolean(p, bits=bits, encoding_type='binary')
    print(boolean_representation.to(torch.int).numpy())
