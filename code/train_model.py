from utils import set_seed, balance_dataset, euclidean_distance_accuracy
set_seed(42)

import torch
from reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBinaryEncoding
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from visualisations import plot_predictions_and_labels, plot_train_history, plot_grid_search
from graph_visualizations import plot_graph_with_weight_coloring_1D
from parameters import * 
import pandas as pd
from tqdm import tqdm

'''
TODO
- take a spectral radius measurement of reservoir apriori
- decide on how many nodes are used for input (input redundancy)
'''

def dataset_init(I: InputParams):
    if I.n_inputs == 1:
        dataset = ConstrainedForagingPathDataset(data_path='/data/1D/levy_walk/n_steps/interval_boundary/dataset.pt')
    elif I.n_inputs == 2:
        dataset = ConstrainedForagingPathDataset(data_path='/data/2D/levy_walk/n_steps/square_boundary/dataset.pt')
    bins = 100
    balance_dataset(dataset, num_bins=bins) # Note that data range play a role here (outliers dangerous)
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize()
    encoder = lambda x: float_array_to_boolean(x, bits=I.bits_per_feature, encoding_type=I.encoding, redundancy=I.redundancy)
    dataset.set_encoder_x(encoder)
    dataset.encode_x()
    dataset.split_dataset()
    return dataset

def train_single_model(yaml_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = load_yaml_config(yaml_path)
    I = P.model.input_layer
    T = P.model.training
    set_seed(T.seed)

    # Create model
    model = BooleanReservoir(P).to(device)

    # uncomment for verification models
    # model = PathIntegrationVerificationModel(I.bits_per_feature, I.n_inputs).to(device)
    # model = PathIntegrationVerificationModelBinaryEncoding(I.n_inputs).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)

    # Init data
    dataset = dataset_init(I)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)
    x_test, y_test = data_loader.dataset.data['x_test'], data_loader.dataset.data['y_test']
    data_loader.dataset.data['x_test'].to(device)
    data_loader.dataset.data['y_test'].to(device)
    data_loader.dataset.data['x'].to(device)
    data_loader.dataset.data['y'].to(device)
    y_hat_test = None
    history = list()

    for epoch in range(T.epochs):
        model.train()
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

        model.eval()
        with torch.no_grad():
            y_hat_test = model(x_test)
            stats['epoch'] = epoch + 1
            stats['loss_test'] = loss.item()
            stats['accuracy_test'] = euclidean_distance_accuracy(y_hat_test, y_test, T.radius_threshold) 
            model.P.logging.train_log.accuracy = stats['accuracy_test']
            model.P.logging.train_log.loss = stats['loss_test']
            model.P.logging.train_log.epoch = stats['epoch']
            history.append(stats)
            print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_test']:.4f}, Test Accuracy: {stats['accuracy_test']:.4f}")

        model.flush_history()
        model.record_history = False # only need history from first epoch if the process is deterministic...

    plot_predictions_and_labels(y_hat_test[:500], y_test[:500], tolerance=T.radius_threshold, axis_limits=[0, 1])
    plot_train_history(history)
    # plot_graph_with_weight_coloring_1D(model, layout='dot')
    return P, model, dataset

def train_and_evaluate(params: Params, device, dataset):
    I = params.model.input_layer
    T = params.model.training
    if multi_config_in_parameters(I):
        dataset = dataset_init(I)
    model = BooleanReservoir(params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    if not dataset:
        dataset = dataset_init(I)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)
    x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    data_loader.dataset.data['x_dev'].to(device)
    data_loader.dataset.data['y_dev'].to(device)
    data_loader.dataset.data['x'].to(device)
    data_loader.dataset.data['y'].to(device)
    best_accuracy = 0
    best_loss = float('inf')
    best_epoch = 0 
    for epoch in range(T.epochs):
        model.train()
        epoch_train_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
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

    return best_accuracy, best_loss, best_epoch, model

def grid_search(yaml_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    I = P.model.input_layer
    L = P.logging
    set_seed(L.grid_search.seed)
    assert not P.logging.out_path.exists()
    param_combinations = generate_param_combinations(P.model)
    history = list() 
    best_accuracy = 0
    best_loss = float('inf')
    best_params = None
    n_config = len(param_combinations)
    n_sample = L.grid_search.n_samples
    dataset = dataset_init(I)

    pbar = tqdm(total=n_config*n_sample, desc="Grid Search Progress")
    for i in range(n_config):
        if i % 100 == 0:
            torch.cuda.empty_cache()
        model_params = param_combinations[i]
        params = Params(model=model_params, logging=P.logging)
        for j in range(n_sample):
            print('#'*60)
            accuracy, loss, epoch, model = train_and_evaluate(params, device, dataset)
            print(f"{model.timestamp_utc}: Config: {i+1:0{len(str(n_config))}d}/{n_config}, Sample: {j+1:0{len(str(n_sample))}d}/{n_sample}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Epoch: {epoch}")
            print(model_params)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_loss = loss
                best_epoch = epoch
                best_params = model_params
            log_data = dict()
            model.P.logging.train_log.accuracy = accuracy
            model.P.logging.train_log.loss = loss
            model.P.logging.train_log.epoch = epoch
            log_data['timestamp_utc'] = model.timestamp_utc 
            log_data['config'] = i+1
            log_data['sample'] = j+1 
            log_data['accuracy'] = accuracy 
            log_data['loss'] = loss 
            log_data['epoch'] = epoch 
            log_data['model_params'] = model_params.model_dump() 
            log_data['model_save_paths'] = model.save()
            history.append(log_data)
            pbar.update(1)
    pbar.close()
    history_df = pd.DataFrame(history)
    file_path = L.out_path / 'log.h5'
    history_df.to_hdf(file_path, key='df', mode='w')
    plot_grid_search(file_path)
    return best_params, best_accuracy, best_loss, best_epoch


if __name__ == '__main__':

    # # Test saving and loading models  
    # p, model, dataset = train_single_model('config/1D/test_single_run.yaml')
    # paths = model.save()
    # load_path = paths['parameters'].parent 
    # model2 = BooleanReservoir(load_path=load_path)
    # p2 = model2.P
    # model2.initial_states = model.initial_states.clone()
    # assert p.model == p2.model
    # assert (model.state_dict()['readout.bias'] == model2.state_dict()['readout.bias']).all()
    # assert (model.state_dict()['readout.weight'] == model2.state_dict()['readout.weight']).all()
    # assert (model.lut == model2.lut).all()
    # assert (model.input_nodes == model2.input_nodes).all()
    # assert (model.initial_states == model2.initial_states).all()
    # assert (list(model.graph.edges(data=True)) == list(model2.graph.edges(data=True)))
    # x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    # model.eval()
    # model2.eval()
    # with torch.no_grad():
    #     y_hat_dev = model(x_dev)
    #     y_hat_dev2 = model2(x_dev)
    #     dev_accuracy = euclidean_distance_accuracy(y_hat_dev, y_dev, p.model.training.radius_threshold)
    #     dev_accuracy2 = euclidean_distance_accuracy(y_hat_dev2, y_dev, p2.model.training.radius_threshold)
    #     assert dev_accuracy == dev_accuracy2

    # # Grid stuff 
    # #####################################
    # best_params, best_accuracy, best_loss, best_epoch = grid_search('config/1D/test_sweep.yaml')
    # best_params, best_accuracy, best_loss, best_epoch = grid_search('config/1D/initial_sweep.yaml')

    best_params, best_accuracy, best_loss, best_epoch = grid_search('config/2D/initial_sweep.yaml')
    print('#'*60)
    print(f'Best accuracy: {best_accuracy}, Best loss: {best_loss}, Best epoch: {best_epoch}')
    print(f'Best parameters: {best_params}')

