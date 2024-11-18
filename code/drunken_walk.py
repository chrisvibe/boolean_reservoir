import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.utils.data import Dataset, DataLoader
from encoding import float_array_to_boolean
matplotlib.use('Agg')

def naive_drunken_walk_with_momentum(num_steps=1000, momentum=0.9, step_size=1.0):
    x, y = np.zeros(num_steps), np.zeros(num_steps) 
    angles = np.random.uniform(0, 2*np.pi, num_steps)
    angle_w_momentum = 0
    
    for i in range(num_steps-1):
        angle_w_momentum = (1 - momentum) * angles[i] + momentum * angle_w_momentum
        x[i+1] = x[i] + step_size * np.cos(angle_w_momentum) 
        y[i+1] = y[i] + step_size * np.sin(angle_w_momentum) 

    return np.column_stack([x, y]), angles


def drunken_walk_with_momentum_decomposed_velocity(num_steps=1000, num_dimensions=2, momentum=0.9, step_size=1.0):
    '''
    Performs a drunken walk with momentum in a given number of dimensions.
    By decomposing the velocity vector and avoiding angles we can simplify calculations
    The simple momentum calculation also avoids the circular function problems associated with angles
    Time is assumed to be 1 in the equation P(T) = P0 + V(T)*T

    Args:
        num_steps (int): Number of steps in the walk.
        num_dimensions (int): Number of dimensions of the walk.
        momentum (float): Momentum factor, between 0 and 1.
        step_size (float): Maximum size of each step.

    Returns:
        p (ndarray): Positions of the drunken walk.
        v (ndarray): Velocities of the drunken walk.
    '''
    # Initialize the position array
    p = np.zeros((num_steps, num_dimensions))

    # Random walk by random velocities
    v = np.random.uniform(-step_size, step_size, (num_steps, num_dimensions))
    v_w_momentum = np.zeros(num_dimensions)
    
    for i in range(num_steps - 1):
        # Apply simple momentum to velocity 
        v_w_momentum = momentum * v_w_momentum + (1 - momentum) * v[i]

        # Normalize the vector for fixed step-size
        norm = np.linalg.norm(v_w_momentum)
        if norm != 0:
            v_w_momentum /= norm

        # keep if we want to preserve step size
        # v_w_momentum *= np.linalg.norm(v[i])
        
        # Take one step
        v[i] = v_w_momentum
        p[i + 1] = p[i] + v[i]
    
    return p, v


def drunken_walk_with_momentum(n_steps=1000, n_dimensions=2, momentum=0.9, step_size=1.0):
    return drunken_walk_with_momentum_decomposed_velocity(num_steps=n_steps, num_dimensions=n_dimensions, momentum=momentum, step_size=step_size)


class DrunkenWalkDataset(Dataset):
    def __init__(self, batch_size, bits_per_dimension, n_steps, n_dimensions, momentum, step_size, encoding):
        data_x = []
        data_y = []
        self.bits_per_dimension = bits_per_dimension
        self.n_steps = n_steps
        self.encoding = encoding

        # Generate the data
        for _ in range(batch_size):
            p, v = drunken_walk_with_momentum(n_steps=1000, n_dimensions=n_dimensions, momentum=momentum, step_size=step_size)
            data_x.append(torch.tensor(v, dtype=torch.float))
            data_y.append(torch.tensor(p[-1], dtype=torch.float))

        self.data = {
            'x': torch.stack(data_x),
            'y': torch.stack(data_y)
        }
    
    def __len__(self):
        return self.data['x'].size(0)
    
    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        return x, y

    def normalize(self):
        self.data['x'] = self.standard_normalization(self.data['x']).to(torch.float)
        self.data['x'] = float_array_to_boolean(self.data['x'], bits=self.bits_per_dimension, encoding_type=self.encoding).to(torch.bool)
        self.data['y'] = self.standard_normalization(self.data['y']).to(torch.float)

    @staticmethod
    def min_max_normalization(data):
        min_ = data.max(axis=0)
        max_ = data.min(axis=0)
        return (data - min_) / (max_ - min_)

    @staticmethod
    def standard_normalization(data):
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        return (data - means) / stds


if __name__ == '__main__':
    # Parameters
    num_steps = 1000
    momentum = 0.8
    step_size = 1.0

    # p, _ = naive_drunken_walk_with_momentum(num_steps=num_steps, momentum=momentum, step_size=step_size)
    p, _ = drunken_walk_with_momentum(n_steps=num_steps, momentum=momentum, step_size=step_size)

    # Plot the walk
    plt.figure(figsize=(8, 8))
    plt.plot(p[:, 0], p[:, 1], marker='o', markersize=2, alpha=0.6)
    plt.title("Drunken Walk with Momentum")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.savefig("/out/drunken_walk.png")