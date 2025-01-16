from utils import set_seed, balance_dataset, euclidean_distance_accuracy
import torch
from reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBaseTwoEncoding
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from visualisations import plot_predictions_and_labels, plot_train_history, plot_grid_search
from graph_visualizations import plot_graph_with_weight_coloring_1D
from parameters import * 
import pandas as pd
from tqdm import tqdm


def dataset_init(I: InputParams):
    set_seed(I.seed) # Note that model is sensitive to this init (new training needed per seed)
    if I.n_inputs == 1:
        dataset = ConstrainedForagingPathDataset(data_path='/data/1D/levy_walk/n_steps/interval_boundary/dataset.pt')
    elif I.n_inputs == 2:
        dataset = ConstrainedForagingPathDataset(data_path='/data/2D/levy_walk/n_steps/square_boundary/dataset.pt')
    bins = 100
    balance_dataset(dataset, num_bins=bins) # Note that data range affects bin assignment (outliers dangerous)
    dataset.set_normalizer_x(min_max_normalization)
    dataset.set_normalizer_y(min_max_normalization)
    dataset.normalize()
    encoder = lambda x: float_array_to_boolean(x, I)
    dataset.set_encoder_x(encoder)
    dataset.encode_x()
    dataset.split_dataset()
    return dataset

def train_single_model(yaml_or_checkpoint_path='', parameter_override:Params=None, model=None):
    if model is None:
        model = BooleanReservoir(params=parameter_override, load_path=yaml_or_checkpoint_path)
    P = model.P
    I = P.model.input_layer
    T = P.model.training
    set_seed(T.seed)

    # Training setup
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)

    # Init data
    dataset = dataset_init(I)
    dataset.data['x'] = dataset.data['x'].to(device)
    dataset.data['y'] = dataset.data['y'].to(device)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)
    dataset.data['x_test'] = dataset.data['x_test'].to(device)
    dataset.data['y_test'] = dataset.data['y_test'].to(device)
    y_hat_test = None
    history = list()
    best_accuracy, best_loss, best_epoch = 0, float('inf'), 0

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
            y_hat_test = model(dataset.data['x_test'])
            stats['epoch'] = epoch + 1
            stats['loss_test'] = loss.item()
            stats['accuracy_test'] = euclidean_distance_accuracy(y_hat_test, dataset.data['y_test'], T.radius_threshold) 
            if stats['accuracy_test'] > best_accuracy:
                best_accuracy = stats['accuracy_test']
                best_loss = stats['loss_test']
                best_epoch = stats['epoch']
            history.append(stats)
            print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_test']:.4f}, Test Accuracy: {stats['accuracy_test']:.4f}")

        model.P.logging.train_log.accuracy = best_accuracy 
        model.P.logging.train_log.loss = best_loss 
        model.P.logging.train_log.epoch = best_epoch 
        model.flush_history()
        model.record_history = False # only need history from first epoch if the process is deterministic...

    plot_predictions_and_labels(y_hat_test[:500], dataset.data['y_test'][:500], tolerance=T.radius_threshold, axis_limits=[0, 1])
    plot_train_history(history)
    # plot_graph_with_weight_coloring_1D(model, layout='dot')
    return P, model, dataset

def train_and_evaluate(p:Params, model: BooleanReservoir, dataset):
    T = p.model.training
    set_seed(T.seed)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)

    best_accuracy, best_loss, best_epoch = 0, float('inf'), 0
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
            y_hat_dev = model(dataset.data['x_dev'])
            dev_loss = criterion(y_hat_dev, dataset.data['y_dev']).item()
            dev_accuracy = euclidean_distance_accuracy(y_hat_dev, dataset.data['y_dev'], T.radius_threshold)
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                best_loss = dev_loss
                best_epoch = epoch
    return best_accuracy, best_loss, best_epoch, model

def grid_search(yaml_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    L = P.logging
    set_seed(P.logging.grid_search.seed)
    assert not P.logging.out_path.exists(), 'Grid search already exists (path taken)'
    param_combinations = generate_param_combinations(P.model)
    history = list() 
    best_accuracy = 0
    best_loss = float('inf')
    best_params = None
    n_config = len(param_combinations)
    n_sample = L.grid_search.n_samples
    last_input_layer_params = dataset = None
    torch.cuda.empty_cache()

    pbar = tqdm(total=n_config*n_sample, desc="Grid Search Progress")
    for i in range(n_config):
        p = Params(model=param_combinations[i], logging=P.logging)
        for j in range(n_sample):
            print('#'*60)
            p.model.training.seed = L.grid_search.seed + j
            p.model.reservoir_layer.seed = L.grid_search.seed + j
            p.model.output_layer.seed = L.grid_search.seed + j
            if last_input_layer_params != p.model.input_layer:
                if dataset:
                    dataset.data['x'].to(cpu_device)
                    dataset.data['y'].to(cpu_device)
                    dataset.data['x_dev'].to(cpu_device)
                    dataset.data['y_dev'].to(cpu_device)
                dataset = dataset_init(p.model.input_layer)
                dataset.data['x'] = dataset.data['x'].to(device)
                dataset.data['y'] = dataset.data['y'].to(device)
                dataset.data['x_dev'] = dataset.data['x_dev'].to(device)
                dataset.data['y_dev'] = dataset.data['y_dev'].to(device)
                last_input_layer_params = p.model.input_layer
            model = BooleanReservoir(p).to(device)
            accuracy, loss, epoch, model = train_and_evaluate(p, model, dataset)
            model.to(cpu_device)
            print(f"{model.timestamp_utc}: Config: {i+1:0{len(str(n_config))}d}/{n_config}, Sample: {j+1:0{len(str(n_sample))}d}/{n_sample}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Epoch: {epoch}")
            print(p.model)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_loss = loss
                best_epoch = epoch
                best_params = p.model
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
            log_data['model_params'] = p.model.model_dump() 
            log_data['model_save_paths'] = model.save()
            history.append(log_data)
            pbar.update(1)
    pbar.close()
    save_yaml_config(P, L.out_path / 'parameters.yaml')
    history_df = pd.DataFrame(history)
    file_path = L.out_path / 'log.h5'
    history_df.to_hdf(file_path, key='df', mode='w')
    plot_grid_search(file_path)
    print('#'*60)
    print(f'Best accuracy: {best_accuracy}, Best loss: {best_loss}, Best epoch: {best_epoch}')
    print(f'Best parameters: {best_params}')

def test_saving_and_loading_models():
    p, model, dataset = train_single_model('config/2D/test_single_run.yaml')
    paths = model.save()
    load_path = paths['parameters'].parent 
    model2 = BooleanReservoir(load_path)
    p2 = model2.P
    model2.initial_states = model.initial_states.clone()
    assert p.model == p2.model
    assert (model.state_dict()['readout.bias'] == model2.state_dict()['readout.bias']).all()
    assert (model.state_dict()['readout.weight'] == model2.state_dict()['readout.weight']).all()
    assert (model.lut == model2.lut).all()
    assert (model.input_nodes == model2.input_nodes).all()
    assert (model.initial_states == model2.initial_states).all()
    assert (list(model.graph.edges(data=True)) == list(model2.graph.edges(data=True)))
    x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    model.eval()
    model2.eval()
    with torch.no_grad():
        y_hat_dev = model(x_dev)
        y_hat_dev2 = model2(x_dev)
        dev_accuracy = euclidean_distance_accuracy(y_hat_dev, y_dev, p.model.training.radius_threshold)
        dev_accuracy2 = euclidean_distance_accuracy(y_hat_dev2, y_dev, p2.model.training.radius_threshold)
        assert dev_accuracy == dev_accuracy2


if __name__ == '__main__':
    # # Test
    # #####################################
    # test_saving_and_loading_models()

    # # Verification models
    #############################################################
    # P = load_yaml_config('config/1D/verification_model.yaml')
    # P = load_yaml_config('config/2D/verification_model.yaml')
    # I = P.model.input_layer
    # model = PathIntegrationVerificationModelBaseTwoEncoding(n_dims=I.n_inputs)
    # model = PathIntegrationVerificationModel(I.bits_per_feature, I.n_inputs)
    # model.P = P
    # p, model, dataset = train_single_model(model=model)

    # # Simple run
    # #####################################
    # p, model, dataset = train_single_model('config/1D/test_single_run.yaml')
    # p, model, dataset = train_single_model('config/2D/test_single_run.yaml')

    # # Grid search stuff 
    # #####################################
    # grid_search('config/1D/test_sweep.yaml')
    # grid_search('config/1D/initial_sweep.yaml')
    # grid_search('config/2D/initial_sweep.yaml')
    #python -u train_model.py | tee /out/logging/1d_and_2d_2025-01-16.log

    # # Load checkpoint, override stuff, and continue training
    #############################################################
    checkpoint_path = Path('/out/grid_search/2D/initial_sweep/models/2025_01_16_091842')
    p = load_yaml_config(checkpoint_path / 'parameters.yaml')
    p.model.training.epochs = 75
    # model = BooleanReservoir(params=p, load_path=checkpoint_path)
    # p, model, dataset = train_single_model(model=model)
    p, model, dataset = train_single_model(parameter_override=p)

    # delete
    # p = load_yaml_config('config/2D/test_single_run.yaml')
    # p.model.training.epochs = 40
    # dataset = dataset_init(p.model.input_layer)
    # for i in range(1000):
    #     p.model.training.seed = i
    #     a, _, _, model = train_and_evaluate(p, model=BooleanReservoir(params=p), dataset=dataset)
    #     if a > 0.3:
    #         save_yaml_config(p, f'/out/test_single_run_{int(a*100):03d}_{i}.yaml')

