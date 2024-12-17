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
from graphs import generate_graph_w_k_avg_incoming_edges
from parameters import InputParams, ReservoirParams, OutputParams, TrainingParams, ModelParams, load_yaml_config, generate_param_combinations

'''
TODO
- take a spectral radius measurement of reservoir apriori
- decide on how many nodes are used for input (input redundancy)
'''

def dataset_init(encoding, n_inputs, bits_per_feature, redundancy):
    if n_inputs == 1:
        dataset = ConstrainedForagingPathDataset(data_path='/data/1D/levy_walk/n_steps/interval_boundary/dataset.pt')
    elif n_inputs == 2:
        dataset = ConstrainedForagingPathDataset(data_path='/data/2D/levy_walk/n_steps/square_boundary/dataset.pt')
    bins = 100
    balance_dataset(dataset, num_bins=bins) # Note that data range play a role here (outliers dangerous)
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize()
    encoder = lambda x: float_array_to_boolean(x, bits=bits_per_feature, encoding_type=encoding, redundancy=redundancy)
    dataset.set_encoder_x(encoder)
    dataset.encode_x()
    dataset.split_dataset()
    return dataset

def train_single_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    I = InputParams(
        encoding='binary_embedding', 
        n_inputs=1, 
        bits_per_feature=48, 
        redundancy=6
        )

    R = ReservoirParams(
        n_nodes=1000, 
        k_avg=2, 
        k_max=10, 
        p=0.5, 
        self_loops=None
        ) 

    O = OutputParams(
        n_outputs=I.n_inputs
        ) 

    T = TrainingParams(
        batch_size=100, 
        epochs=100, 
        radius_threshold=0.05,
        learning_rate=0.001
        )

    P = ModelParams(
    input_layer=I, 
    reservoir_layer=R, 
    output_layer=O, 
    training=T
    )
   
    # Create model
    graph = generate_graph_w_k_avg_incoming_edges(R.n_nodes, R.k_avg, k_max=R.k_max, self_loops=R.self_loops)
    lut = lut_random(R.n_nodes, R.k_max, p=R.p)
    model = BooleanReservoir(graph, lut, T.batch_size, R.k_max, I.n_inputs, I.bits_per_feature, O.n_outputs)

    # uncomment for verification models
    # model = PathIntegrationVerificationModel(I.bits_per_feature, I.n_inputs).to(device)
    # model = PathIntegrationVerificationModelBinaryEncoding(I.n_inputs).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)

    # Init data
    dataset = dataset_init(I.encoding, I.n_inputs, I.bits_per_feature, I.redundancy)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)
    x_test, y_test = data_loader.dataset.data['x_test'], data_loader.dataset.data['y_test']
    data_loader.dataset.data['x_test'].to(device)
    data_loader.dataset.data['y_test'].to(device)
    data_loader.dataset.data['x'].to(device)
    data_loader.dataset.data['y'].to(device)
    y_hat_test = None
    history = list()

    for epoch in range(T.epochs):
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
            epoch_correct_train_predictions += euclidean_distance_accuracy(y_hat, y, T.radius_threshold, normalize=False) 

        stats['loss_train'] = epoch_train_loss / len(data_loader)
        stats['accuracy_train'] = epoch_correct_train_predictions / (len(data_loader) * data_loader.batch_size)

        with torch.no_grad():
            y_hat_test = model(x_test)
            stats['epoch'] = epoch + 1
            stats['loss_test'] = loss.item()
            stats['accuracy_test'] = euclidean_distance_accuracy(y_hat_test, y_test, T.radius_threshold) 
            history.append(stats)
            print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_test']:.4f}, Test Accuracy: {stats['accuracy_test']:.4f}")

        model.flush_history()
        model.record = False # only need history from first epoch if the process is deterministic...

    plot_predictions_and_labels(y_hat_test[:500], y_test[:500], tolerance=T.radius_threshold, axis_limits=[0, 1])
    plot_train_history(history)
    # plot_graph_with_weight_coloring_1D(model, layout='dot')


def train_and_evaluate(params: ModelParams, device, dataset):
    I = params.input_layer
    R = params.reservoir_layer
    O = params.output_layer
    T = params.training

    graph = generate_graph_w_k_avg_incoming_edges(
        R.n_nodes, R.k_avg, 
        k_max=R.k_max, self_loops=R.self_loops
    )
    lut = lut_random(R.n_nodes, R.k_max, p=R.p)
    model = BooleanReservoir(graph, lut, T.batch_size, R.k_max, 
                             I.n_inputs, I.bits_per_feature, O.n_outputs).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    if dataset is None:
        dataset = dataset_init(
            I.encoding[0], 
            I.n_inputs[0],
            I.bits_per_feature[0], 
            I.redundancy[0], 
        )
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)
    x_dev, y_dev = dataset.data['x_dev'].to(device), dataset.data['y_dev'].to(device)
    
    best_accuracy = 0
    best_loss = float('inf')
    best_epoch = 0 
    
    for epoch in range(T.epochs):
        model.train()
        epoch_train_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            y_hat = model(x.to(device))
            loss = criterion(y_hat, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            y_hat_dev = model(x_dev)
            dev_loss = criterion(y_hat_dev, y_dev).item()
            dev_accuracy = euclidean_distance_accuracy(y_hat_dev, y_dev, T.radius_threshold)

            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                best_loss = dev_loss
                best_epoch = epoch

    return best_accuracy, best_loss, best_epoch

def grid_search(yaml_path):
    P = load_yaml_config(yaml_path)
    param_combinations = generate_param_combinations(P.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_accuracy = 0
    best_loss = float('inf')
    best_params = None
    i = 0
    n = len(param_combinations)

    # avoid dataset load if it doesnt change 
    dataset = None
    I = P.model.input_layer
    def dataset_changes(input_params: InputParams) -> bool:
        for value in input_params._asdict().values():
            if isinstance(value, list) and len(value) > 1:
                return True
        return False
    if not dataset_changes(I):
        dataset = dataset_init(
            I.encoding[0], 
            I.n_inputs[0],
            I.bits_per_feature[0], 
            I.redundancy[0], 
        )
    for params in param_combinations:
        print('#'*60)
        accuracy, loss, epoch = train_and_evaluate(params, device, dataset)
        print(f"Iteration: {i:0{len(str(n))}d}/{n}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Epoch: {epoch}")
        print(params)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_loss = loss
            best_epoch = epoch
            best_params = params
        i += 1

    return best_params, best_accuracy, best_loss, best_epoch

if __name__ == '__main__':
    # train_single_model()

    # Grid stuff 
    ######################################
    best_params, best_accuracy, best_loss, best_epoch = grid_search('config.yaml')
    print('#'*60)
    print(f'Best accuracy: {best_accuracy}, Best loss: {best_loss}, Best epoch: {best_epoch}')
    print(f'Best parameters: {best_params}')

