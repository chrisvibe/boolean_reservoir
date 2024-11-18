import torch
import numpy as np
import random

# Ensure reproducibility by setting seeds globally
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call it before any random operations
seed = 42
set_seed(seed)

from reservoir import BooleanReservoir, ThreeLayerPerceptron
from primes import primes
from drunken_walk import DrunkenWalkDataset
from torch.utils.data import DataLoader
import torch.nn as nn

'''
TODO
- check if bool can be used instead of int32 etc...
- lup needs a max connectivity (or else it becomes super long)
- how to enforce and measure a target connectivity
- take a spectral radius measurement of reservoir apriori
- good threshold of output data
- how to encode input velocities as binary data
- decide on how many nodes are used for input (input redundancy)
- input nodes override select reservoir nodes, is this good?
- scaling issue if we want many connections with current lut
- should each node have a lut? or use a master lut?
- is the path integration good? Are important properties lost upon scaling?
'''

# Parameters: Path Integration
n_dimensions = 2
n_steps = 1000
momentum = 0.9
step_size = 1.0

# Parameters: Reservoir 
reservoir_size = 100  # Number of nodes in the reservoir
lut_length = 16  # max LUT length (implies max connectivity)
n_features = n_dimensions  # Number of dimensions
bits_per_feature = 5  # Number of bits per dimension
output_size = n_dimensions  # Number of dimensions
encoding = 'tally'

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = BooleanReservoir(bits_per_feature, n_features, reservoir_size, output_size, lut_length, device, primes, seed=seed).to(device)
# uncomment for alternative model
# model = ThreeLayerPerceptron(bits_per_feature, n_features, n_steps)
# model.flush_history = lambda: None

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
batch_size = 50
epochs = 25
radius_threshold = 0.5

dataset = DrunkenWalkDataset(batch_size=batch_size, bits_per_dimension=bits_per_feature, n_steps=n_steps, n_dimensions=n_dimensions, momentum=momentum, step_size=step_size,  encoding=encoding)
dataset.normalize() # TODO note that it is more natural to do this outside of dataset given parameter grouping above...
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    dataset.data['x'].to(device)
    dataset.data['y'].to(device)
    for x, y in data_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

    # Calculate accuracy after each epoch
    with torch.no_grad():
        # Compute Euclidean distance between predicted and true coordinates
        distances = torch.sqrt(torch.sum((y_hat - y) ** 2, dim=1))
        correct_predictions = distances < radius_threshold
        accuracy = correct_predictions.sum().item() / batch_size

    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    model.flush_history()
    model.record = False # only need history from first epoch if the process is deterministic...