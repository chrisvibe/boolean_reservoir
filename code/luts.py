import torch

def lut_random(reservoir_size, lut_length, p=0.5):
    # Step 1: Create a random float array
    lut = torch.rand(reservoir_size, lut_length)

    # Step 2: Update the last column based on the probability p
    lut[:, -1] = torch.bernoulli(torch.full((reservoir_size,), p))

    # Step 3: Convert the whole array to boolean
    lut_bool = lut > 0.5

    return lut_bool