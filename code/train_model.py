from utils import set_seed, balance_dataset, euclidean_distance_accuracy
set_seed(42)

import torch
from reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBinaryEncoding
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from visualisations import plot_predictions_and_labels, plot_train_history
from graph_visualizations import plot_graph_with_weight_coloring_1D
from luts import lut_random 
from graphs import graph_average_k_incoming_edges_w_self_loops

'''
TODO
- take a spectral radius measurement of reservoir apriori
- decide on how many nodes are used for input (input redundancy)
'''

def dataset_init(batch_size, n_inputs, bits_per_feature, encoding):
    if n_inputs == 1:
        dataset = ConstrainedForagingPathDataset(data_path='/data/1D/levy_walk/n_steps/interval_boundary/dataset.pt')
    elif n_inputs == 2:
        dataset = ConstrainedForagingPathDataset(data_path='/data/2D/levy_walk/n_steps/square_boundary/dataset.pt')
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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters: Input Layer
    encoding = 'tally'
    n_inputs = 1
    bits_per_feature = 48
    redundancy = 6

    # Parameters: Reservoir Layer
    n_nodes = 1000
    max_connectivity = 5 
    avg_k = 3
    p = 0.5

    # Parameters: Output Layer
    n_outputs = n_inputs

    # Training
    batch_size = 100
    epochs = 100
    radius_threshold = 0.05
    learning_rate = 0.001

    # Create model
    graph = graph_average_k_incoming_edges_w_self_loops(n_nodes, avg_k)
    lut = lut_random(n_nodes, 2**max_connectivity, p=p)
    model = BooleanReservoir(graph, lut, batch_size, max_connectivity, n_inputs, bits_per_feature, n_outputs)

    # uncomment for verification models
    # model = PathIntegrationVerificationModel(bits_per_feature, n_inputs).to(device)
    # model = PathIntegrationVerificationModelBinaryEncoding(n_inputs).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Init data
    data_loader = dataset_init(batch_size, n_inputs, bits_per_feature, encoding)
    x_test, y_test = data_loader.dataset.data['x_test'], data_loader.dataset.data['y_test']
    data_loader.dataset.data['x_test'].to(device)
    data_loader.dataset.data['y_test'].to(device)
    data_loader.dataset.data['x'].to(device)
    data_loader.dataset.data['y'].to(device)
    y_hat_test = None
    history = list()
    checkpoint_stats = dict()

    for epoch in range(epochs):
        stats = dict()
        epoch_train_loss = 0
        epoch_correct_train_predictions = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_correct_train_predictions += euclidean_distance_accuracy(y_hat, y, radius_threshold, normalize=False) 

        stats['loss_train'] = epoch_train_loss / len(data_loader)
        stats['accuracy_train'] = epoch_correct_train_predictions / (len(data_loader) * data_loader.batch_size)

        with torch.no_grad():
            y_hat_test = model(x_test)
            stats['epoch'] = epoch + 1
            stats['loss_test'] = loss.item()
            stats['accuracy_test'] = euclidean_distance_accuracy(y_hat_test, y_test, radius_threshold) 
            history.append(stats)
            print(f"Epoch: {stats['epoch']:0{len(str(epochs))}d}/{epochs}, Loss: {stats['loss_test']:.4f}, Test Accuracy: {stats['accuracy_test']:.4f}")

        model.flush_history()
        model.record = False # only need history from first epoch if the process is deterministic...

    plot_predictions_and_labels(y_hat_test[:500], y_test[:500], tolerance=radius_threshold, axis_limits=[0, 1])
    plot_train_history(history)
    # plot_graph_with_weight_coloring_1D(model, layout='dot')