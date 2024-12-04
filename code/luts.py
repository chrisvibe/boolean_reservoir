import torch

def lut_random(reservoir_size, lut_length):
    lut_list = [torch.randint(0, 2, (2 ** lut_length,), dtype=torch.bool) for _ in range(reservoir_size)]
    lut = torch.stack(lut_list)
    return lut