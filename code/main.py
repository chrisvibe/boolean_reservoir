import torch

from utils import set_seed
seed = 42 # TODO is this global now???!!!
set_seed(42)

from reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBinaryEncoding
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from visualisations import plot_predictions_and_labels

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
'''

def dataset_init(batch_size, bits_per_feature, encoding):
    dataset = ConstrainedForagingPathDataset(data_path='/data/levy_walk/25_steps/square_boundary/dataset.pt')
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize() # TODO uncomment!!!
    encoder = lambda x: float_array_to_boolean(x, bits=bits_per_feature, encoding_type=encoding)
    dataset.set_encoder_x(encoder)
    # dataset.encode_x() # TODO uncomment!!!
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Parameters: Input Layer
encoding = 'binary'
bits_per_feature = 5  # Number of bits per dimension
input_dim = 2  # Number of dimensions

# Parameters: Reservoir Layer
reservoir_size = 100  # Number of nodes in the reservoir
lut_length = 16  # max LUT length (implies max connectivity)

# Parameters: Output Layer
output_dim = 2  # Number of dimensions

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BooleanReservoir(bits_per_feature, input_dim, reservoir_size, output_dim, lut_length, device, primes, seed=seed).to(device)

# uncomment for alternative model
# model = PathIntegrationVerificationModel(bits_per_feature).to(device)
model = PathIntegrationVerificationModelBinaryEncoding().to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
batch_size = 100
epochs = 25
radius_threshold = 0.1
data_loader = dataset_init(batch_size, bits_per_feature, encoding)


model.record = False # TODO uncomment
for epoch in range(epochs):
    data_loader.dataset.data['x'].to(device)
    data_loader.dataset.data['y'].to(device)
    for x, y in data_loader:
        # ------------------------------------------------------------------
        # TODO DEBUG DELETE

        # sum without normalization or encoding:
        # m = 0
        # print(torch.sum(x[m:m+1], dim=1), y[m]) # works

        # sum with normalization but without encoding wont work since they are normalized differently, but then you can scale by a fixed constant!
        m = 0
        k = y[m:m+1] / torch.sum(x[m:m+1], dim=1)
        m = 1
        print(torch.sum(x[m:m+1], dim=1) * k, y[m]) # doesnt work

        # testing comutative property: sum with manual normalization (no normalization or encoding)
        # m = 0
        # k = x[m:m+1, 0]  # select random k
        # m = 1
        # print(torch.sum(x[m:m+1], dim=1) * k, y[m] * k) # works
        # print(torch.sum(x[m:m+1] * k, dim=1), y[m] * k) # works
        # ------------------------------------------------------------------

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
    if epoch == epochs - 1:
        plot_predictions_and_labels(y_hat, y)
