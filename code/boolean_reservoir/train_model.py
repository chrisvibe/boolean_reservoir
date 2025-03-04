from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
from copy import deepcopy
from boolean_reservoir.utils import set_seed
from boolean_reservoir.reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBaseTwoEncoding
from boolean_reservoir.graph_visualizations_dash import *
from boolean_reservoir.parameters import * 
from boolean_reservoir.visualizations import *

class AccuracyFunction(ABC):
    @abstractmethod
    def accuracy(self, y_hat, y, threshold, normalize=True):
        pass

class EuclideanDistanceAccuracy(AccuracyFunction):
    def accuracy(self, y_hat, y, threshold, normalize=True):
        distances = torch.sqrt(torch.sum((y_hat - y) ** 2, dim=1))
        correct_predictions = (distances < threshold).sum().item()
        if normalize:
            return correct_predictions / len(y)
        else:
            return correct_predictions

class BooleanAccuracy(AccuracyFunction):
    def accuracy(self, y_hat, y, threshold, normalize=True):
        y_hat_rounded = y_hat > threshold
        correct = (y_hat_rounded == y.to(torch.bool)).sum().item()
        if normalize:
            return correct / len(y)
        else:
            return correct

class DatasetInit(ABC):
    @abstractmethod
    def dataset_init(self, I: InputParams):
        pass

def criterion_strategy(strategy):
    if strategy == 'MSE':
        return nn.MSELoss()
    elif strategy == 'BCE':
        return nn.BCELoss()
    else:
        raise ValueError

def train_single_model(yaml_or_checkpoint_path='', parameter_override:Params=None, model=None, save_model=True, dataset_init: DatasetInit=None, accuracy: AccuracyFunction=None):
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
    dataset.data['x_' + T.evaluation] = dataset.data['x_' + T.evaluation].to(device)
    dataset.data['y_' + T.evaluation] = dataset.data['y_' + T.evaluation].to(device)
    _, model, history = train_and_evaluate(P, model, dataset, record_stats=True, verbose=True, accuracy=accuracy)
    if save_model:
        model.save()
    return P, model, dataset, history

def train_and_evaluate(p:Params, model: BooleanReservoir, dataset: Dataset, record_stats=False, verbose=False, accuracy: AccuracyFunction=EuclideanDistanceAccuracy()):
    T = p.model.training
    set_seed(T.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    criterion = criterion_strategy(T.criterion)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True, drop_last=True)
    x_eval = 'x_' + T.evaluation
    y_eval = 'y_' + T.evaluation
    best_stats = {'eval': T.evaluation, 'epoch': 0, 'accuracy':0, 'loss': float('inf')}
    train_history = list()

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
            epoch_correct_train_predictions += accuracy(y_hat, y, T.accuracy_threshold, normalize=False)
        model.eval()
        with torch.no_grad():
            y_hat_eval = model(dataset.data[x_eval])
            eval_loss = criterion(y_hat_eval, dataset.data[y_eval]).item()
            eval_accuracy = accuracy(y_hat_eval, dataset.data[y_eval], T.accuracy_threshold)
            if eval_loss < best_stats['loss']:
                best_stats['epoch'] = epoch + 1
                best_stats['loss'] = eval_loss
                best_stats['accuracy'] = eval_accuracy
            if record_stats:
                stats = dict()
                stats['epoch'] = epoch + 1
                stats['loss_train'] = epoch_train_loss / len(data_loader)
                stats['accuracy_train'] = epoch_correct_train_predictions / (len(data_loader) * data_loader.batch_size)
                stats['loss_' + T.evaluation] = eval_loss 
                stats['accuracy_' + T.evaluation] = eval_accuracy 
                train_history.append(stats)
                if verbose:
                    print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_' + T.evaluation]:.4f}, Accuracy: {stats['accuracy_' + T.evaluation]:.4f}")
        # deterministic reservoirs only need history from the first epoch
        if hasattr(model, 'flush_history'):
            model.flush_history()
            model.record_history = False
    if verbose:
        print(f'Best loss: {best_stats}')
    model.P.logging.train_log.accuracy = best_stats['accuracy']
    model.P.logging.train_log.loss = best_stats['loss']
    model.P.logging.train_log.epoch = best_stats['epoch']
    return best_stats, model, train_history

def grid_search(yaml_path, dataset_init:DatasetInit=None, accuracy:AccuracyFunction=None, param_combinations=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    L = P.logging
    set_seed(L.grid_search.seed)
    assert not L.out_path.exists(), 'Grid search already exists (path taken)'
    L.out_path.mkdir(parents=True, exist_ok=True)
    save_yaml_config(P, L.out_path / 'parameters.yaml')
    if param_combinations is None:
        param_combinations = generate_param_combinations(P.model)
    history = list() 
    n_config = len(param_combinations)
    n_sample = L.grid_search.n_samples
    last_input_layer_params = dataset = best_params = None
    torch.cuda.empty_cache()

    pbar = tqdm(total=n_config*n_sample, desc="Grid Search Progress")
    for i in range(n_config):
        for j in range(n_sample):
            p = Params(model=deepcopy(param_combinations[i]), logging=deepcopy(P.logging))
            t = p.model.training
            print('#'*60)
            k = L.grid_search.seed*4 + i*2 + j
            t.seed = k 
            p.model.reservoir_layer.seed = k 
            p.model.output_layer.seed = k 
            if last_input_layer_params != p.model.input_layer:
                dataset = dataset_init(p.model.input_layer)
                dataset.data['x'] = dataset.data['x'].to(device)
                dataset.data['y'] = dataset.data['y'].to(device)
                dataset.data['x_' + t.evaluation] = dataset.data['x_' + t.evaluation].to(device)
                dataset.data['y_' + t.evaluation] = dataset.data['y_' + t.evaluation].to(device)
                last_input_layer_params = p.model.input_layer
            model = best_epoch = None
            try:
                model = BooleanReservoir(p).to(device)
                best_epoch, model, _ = train_and_evaluate(p, model, dataset, record_stats=False, verbose=False, accuracy=accuracy)
                model.to(cpu_device)
            except Exception as error:
                print("Error:", error)
                model = BooleanReservoir
                model.P = p
                model.timestamp_utc = model.get_timestamp_utc()
                model.save = lambda *args: None
                best_epoch = {'eval': t.evaluation, 'epoch': 0, 'accuracy':0, 'loss': float('inf')}
            print(f"{model.timestamp_utc}: Config: {i+1:0{len(str(n_config))}d}/{n_config}, Sample: {j+1:0{len(str(n_sample))}d}/{n_sample}, Loss: {best_epoch['loss']:.4f}, Accuracy: {best_epoch['accuracy']:.4f}, Epoch: {best_epoch['epoch']}")
            print(p.model)
            if best_params is None or best_params.logging.train_log.loss is None or (best_epoch['loss'] < best_params.logging.train_log.loss):
                best_params = model.P
            log_data = dict()
            log_data['timestamp_utc'] = model.timestamp_utc 
            log_data['config'] = i+1
            log_data['sample'] = j+1 
            log_data.update(best_epoch)
            log_data['params'] = p
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
    print(f'Best parameters:\n{best_params}')
    return P, best_params

def test_saving_and_loading_models():
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    p, model, dataset, history = train_single_model('config/path_integration/2D/test_model.yaml')
    paths = model.save()
    #################################################################
    model2 = BooleanReservoir(load_path=paths['parameters'].parent)
    test_model_likeness(model, model2, dataset)

def test_reproducibility_of_loaded_grid_search_checkpoint():
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    _, p = grid_search('config/path_integration/2D/test_sweep.yaml')
    print('-'*10, '\n', p, '\n', '-'*10)
    model = BooleanReservoir(load_path=p.logging.last_checkpoint)
    #################################################################
    p2 = deepcopy(model.P)
    p2, model2, dataset2, history2 = train_single_model(parameter_override=p2)
    test_model_likeness(model, model2, dataset2)
    assert model.P.logging.train_log.accuracy == model2.P.logging.train_log.accuracy, 'log accuracies'

def test_model_likeness(model, model2, dataset, accuracy=EuclideanDistanceAccuracy()):
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
        dev_accuracy = accuracy(y_hat_dev, y_dev, model.P.model.training.accuracy_threshold)
        dev_accuracy2 = accuracy(y_hat_dev2, y_dev, model2.P.model.training.accuracy_threshold)
        assert dev_accuracy == dev_accuracy2, 'accuracies'