from utils import set_seed, balance_dataset, euclidean_distance_accuracy
import torch
from reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBaseTwoEncoding
from encoding import float_array_to_boolean, min_max_normalization
from constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from visualisations import plot_predictions_and_labels, plot_train_history, plot_grid_search
from graph_visualizations import plot_graph_with_weight_coloring_1D
from parameters import * 
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
from copy import deepcopy

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

    # Init data
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset_init(I)
    dataset.data['x'] = dataset.data['x'].to(device)
    dataset.data['y'] = dataset.data['y'].to(device)
    dataset.data['x_test'] = dataset.data['x_test'].to(device)
    dataset.data['y_test'] = dataset.data['y_test'].to(device)

    # TODO DELETE
    dataset.data['x_test'] = dataset.data['x_dev'].to(device)
    dataset.data['y_test'] = dataset.data['y_dev'].to(device)
    # model.state_dict()
# OrderedDict([('readout.weight', tensor([[-0.0222, -0.0316, -0.0093,  ..., -0.0105,  0.0191, -0.0138],
        # [ 0.021...0.0176,  ...,  0.0183,  0.0115, -0.0097]])), ('readout.bias', tensor([0.0193, 0.0226]))])

    _, model, history = train_and_evaluate(P, model, dataset, evaluation='test', verbose=True)

    y_test = dataset.data['y_test'][:500]
    y_hat_test = model(dataset.data['x_test'][:500])
    plot_predictions_and_labels(y_hat_test, y_test, tolerance=T.radius_threshold, axis_limits=[0, 1])
    plot_train_history(history)
    # plot_graph_with_weight_coloring_1D(model, layout='dot')
    return P, model, dataset

def train_and_evaluate(p:Params, model: BooleanReservoir, dataset: Dataset, evaluation='test', verbose=False):
    T = p.model.training
    set_seed(T.seed)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True)
    x_eval = 'x_' + evaluation
    y_eval = 'y_' + evaluation
    if verbose:
        print(f"Evaluation: {evaluation}")

    best = {'eval': evaluation, 'epoch': 0, 'accuracy':0, 'loss': float('inf')}
    history = list()
    for epoch in range(T.epochs):
        epoch_correct_train_predictions = epoch_train_loss = 0
        model.train()
        for x, y in data_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            stats = dict()
            stats['loss_train'] = epoch_train_loss / len(data_loader)
            stats['accuracy_train'] = epoch_correct_train_predictions / (len(data_loader) * data_loader.batch_size)
            y_hat_eval = model(dataset.data[x_eval])
            eval_loss = criterion(y_hat_eval, dataset.data[y_eval]).item()
            eval_accuracy = euclidean_distance_accuracy(y_hat_eval, dataset.data[y_eval], T.radius_threshold)
            stats['epoch'] = epoch + 1
            stats['loss_' + evaluation] = eval_loss 
            stats['accuracy_' + evaluation] = eval_accuracy 
            if eval_accuracy > best['accuracy']:
                best['epoch'] = epoch + 1
                best['accuracy'] = eval_accuracy
                best['loss'] = eval_loss
            history.append(stats)
            if verbose:
                print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_' + evaluation]:.4f}, Accuracy: {stats['accuracy_' + evaluation]:.4f}")
        # only need reservoir dynamics history from first epoch if the process is deterministic...
        model.flush_history()
        model.record_history = False
    if verbose:
        print(f'Evaluation: {evaluation}, Best accuracy: {best['accuracy']}, Best loss: {best['loss']}, Best epoch: {best['epoch']}')
    model.P.logging.train_log.accuracy = best['accuracy']
    model.P.logging.train_log.loss = best['loss']
    model.P.logging.train_log.epoch = best['epoch']
    return best, model, history

def grid_search(yaml_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    L = P.logging
    set_seed(L.grid_search.seed)
    assert not L.out_path.exists(), 'Grid search already exists (path taken)'
    L.out_path.mkdir(parents=True, exist_ok=True)
    save_yaml_config(P, L.out_path / 'parameters.yaml')
    param_combinations = generate_param_combinations(P.model)
    history = list() 
    n_config = len(param_combinations)
    n_sample = L.grid_search.n_samples
    last_input_layer_params = dataset = None
    torch.cuda.empty_cache()

    evaluation = 'dev'
    best_model = {'eval': evaluation, 'epoch': 0, 'accuracy':0, 'loss': float('inf'), 'params': None}
    pbar = tqdm(total=n_config*n_sample, desc="Grid Search Progress")
    for i in range(n_config):
        p = Params(model=param_combinations[i], logging=deepcopy(P.logging))
        for j in range(n_sample):
            print('#'*60)
            k = L.grid_search.seed*4 + i*2 + j
            p.model.training.seed = k 
            p.model.reservoir_layer.seed = k 
            p.model.output_layer.seed = k 
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
            best_epoch, model, _ = train_and_evaluate(p, model, dataset, evaluation=evaluation, verbose=False)
            model.to(cpu_device)
            print(f"{model.timestamp_utc}: Config: {i+1:0{len(str(n_config))}d}/{n_config}, Sample: {j+1:0{len(str(n_sample))}d}/{n_sample}, Loss: {best_epoch['loss']:.4f}, Accuracy: {best_epoch['accuracy']:.4f}, Epoch: {best_epoch['epoch']}")
            print(p.model)
            if best_epoch['accuracy'] > best_model['accuracy']:
                best_model.update(best_epoch)
                best_model['params'] = p
            log_data = dict()
            log_data['timestamp_utc'] = model.timestamp_utc 
            log_data['config'] = i+1
            log_data['sample'] = j+1 
            log_data['accuracy'] = best_epoch['accuracy'] 
            log_data['loss'] = best_epoch['loss'] 
            log_data['epoch'] = best_epoch['epoch'] 
            log_data['params'] = p.model_dump() 
            model.save()
            history.append(log_data)
            pbar.update(1)
    pbar.close()
    print('saving history...')
    history_df = pd.DataFrame(history)
    file_path = L.out_path / 'log.h5'
    history_df.to_hdf(file_path, key='df', mode='w')
    print('making plots...')
    plot_grid_search(file_path)
    print('#'*60)
    print(f'Best accuracy: {best_model['accuracy']}, Best loss: {best_model['loss']}, Best epoch: {best_model['epoch']}')
    print(f'Best parameters: {best_model['params']}')
    return P, best_model['params']

def test_saving_and_loading_models():
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    p, model, dataset = train_single_model('config/2D/test_model.yaml')
    paths = model.save()
    #################################################################
    model2 = BooleanReservoir(load_path=paths['parameters'].parent)
    test_model_likeness(model, model2, dataset)

def test_reproducibility_of_loaded_grid_search_checkpoint():
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    _, P = grid_search('config/2D/test_sweep.yaml')
    print('-'*10, '\n', P, '\n', '-'*10)
    model = BooleanReservoir(load_path=P.logging.checkpoint_path)
    #################################################################
    p2 = deepcopy(model.P)
    p2, model2, dataset2 = train_single_model(parameter_override=p2)
    test_model_likeness(model, model2, dataset2)
    assert model.P.logging.train_log.accuracy == model2.P.logging.train_log.accuracy, 'log accuracies'

def test_model_likeness(model, model2, dataset):
    assert model.P.model == model.P.model, 'models'
    assert (model.state_dict()['readout.bias'] == model2.state_dict()['readout.bias']).all(), 'bias'
    assert (model.state_dict()['readout.weight'] == model2.state_dict()['readout.weight']).all(), 'weights'
    assert (model.lut == model2.lut).all(), 'lut'
    assert (model.input_nodes == model2.input_nodes).all(), 'input nodes'
    assert (model.initial_states == model2.initial_states).all(), 'initial_states'
    assert (list(model.graph.edges(data=True)) == list(model2.graph.edges(data=True))), 'graph'
    x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    model.eval()
    model2.eval()
    with torch.no_grad():
        y_hat_dev = model(x_dev)
        y_hat_dev2 = model2(x_dev)
        dev_accuracy = euclidean_distance_accuracy(y_hat_dev, y_dev, model.P.model.training.radius_threshold)
        dev_accuracy2 = euclidean_distance_accuracy(y_hat_dev2, y_dev, model2.P.model.training.radius_threshold)
        assert dev_accuracy == dev_accuracy2, 'accuracies'


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
    # p, model, dataset = train_single_model('config/1D/good_model.yaml')
    # p, model, dataset = train_single_model('config/2D/good_model.yaml') # TODO not good

    # # Grid search stuff 
    # #####################################
    # grid_search('config/1D/test_sweep.yaml')
    # grid_search('config/1D/initial_sweep.yaml')
    # grid_search('config/2D/initial_sweep.yaml')
    #python -u train_model.py | tee /out/logging/1d_and_2d_2025-01-16.log

    # Test
    #####################################
    test_saving_and_loading_models()
    test_reproducibility_of_loaded_grid_search_checkpoint()

    # TODO test this:
    # checkpoint_path = Path('/out/grid_search/2D/initial_sweep/models/2025_01_16_091842')