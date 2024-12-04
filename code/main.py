import torch

from utils import set_seed, balance_dataset
set_seed(42)

from reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBinaryEncoding
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from visualisations import plot_predictions_and_labels
from luts import lut_random 
from graphs import graph_average_k_income_edges_w_self_loops 

'''
TODO
- check if bool can be used instead of int32 etc...
- lup needs a max connectivity (or else it becomes super long)
- how to enforce and measure a target connectivity
- take a spectral radius measurement of reservoir apriori
- decide on how many nodes are used for input (input redundancy)
- input nodes override select reservoir nodes, is this good?
- scaling issue if we want many connections with current lut
'''

def dataset_init(batch_size, bits_per_feature, encoding):
    dataset = ConstrainedForagingPathDataset(data_path='/data/levy_walk/25_steps/square_boundary/dataset.pt')
    bins = 100
    balance_dataset(dataset, num_bins=bins) # Note that data range play a role here (outliers dangerous)
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize()
    encoder = lambda x: float_array_to_boolean(x, bits=bits_per_feature, encoding_type=encoding)
    dataset.set_encoder_x(encoder)
    dataset.encode_x()
    dataset.split_dataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Parameters: Input Layer
encoding = 'binary'
bits_per_feature = 10  # Number of bits per dimension
input_dim = 2  # Number of dimensions

# Parameters: Reservoir Layer
reservoir_size = 100  # Number of nodes in the reservoir
lut_length = 16  # max LUT length (implies max connectivity)

# Parameters: Output Layer
output_dim = 2  # Number of dimensions

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# lut = lut_random(reservoir_size, lut_length)
# graph = graph_average_k_income_edges_w_self_loops(reservoir_size, avg_k=2)
# model = BooleanReservoir(graph, lut, bits_per_feature, input_dim, reservoir_size, output_dim, device, record=False).to(device)

# uncomment for alternative model
# model = PathIntegrationVerificationModel(bits_per_feature, input_dim).to(device)
model = PathIntegrationVerificationModelBinaryEncoding(input_dim).to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
batch_size = 100
epochs = 250
radius_threshold = 0.05
data_loader = dataset_init(batch_size, bits_per_feature, encoding)


x_test, y_test = data_loader.dataset.data['x_test'].to(device), data_loader.dataset.data['y_test'].to(device)
for epoch in range(epochs):
    data_loader.dataset.data['x'].to(device)
    data_loader.dataset.data['y'].to(device)
    for x, y in data_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

    # Calculate accuracy after each epoch
    with torch.no_grad():
        data_loader.dataset.data['x_test'].to(device)
        data_loader.dataset.data['y_test'].to(device)
        y_hat_test = model(x_test)
        # Compute Euclidean distance between predicted and true coordinates
        distances = torch.sqrt(torch.sum((y_hat_test - y_test) ** 2, dim=1))
        correct_predictions = distances < radius_threshold
        accuracy_test = correct_predictions.sum().item() / len(y_test)

        print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy_test:.4f}")

    model.flush_history()
    model.record = False # only need history from first epoch if the process is deterministic...
    if epoch == epochs - 1:
        plot_predictions_and_labels(y_hat, y, tolerance=radius_threshold, scale=[0, 1])
