from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import pandas as pd
from tqdm import tqdm
from projects.boolean_reservoir.code.utils import set_seed, generate_unique_seed
from projects.boolean_reservoir.code.reservoir import BooleanReservoir
from projects.boolean_reservoir.code.graph_visualizations_dash import *
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.visualizations import *

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
    def dataset_init(self, P: Params=None) -> Dataset:
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
    T = P.model.training

    # Init data
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset_init(P)
    dataset.data['x'] = dataset.data['x'].to(device)
    dataset.data['y'] = dataset.data['y'].to(device)
    dataset.data['x_' + T.evaluation] = dataset.data['x_' + T.evaluation].to(device)
    dataset.data['y_' + T.evaluation] = dataset.data['y_' + T.evaluation].to(device)
    _, model, train_history = train_and_evaluate(model, dataset, record_stats=True, verbose=True, accuracy=accuracy)
    if save_model:
        model.save()
    return P, model, dataset, train_history

def train_and_evaluate(model: BooleanReservoir, dataset: Dataset, record_stats=False, verbose=False, accuracy: AccuracyFunction=EuclideanDistanceAccuracy()):
    T = model.P.model.training
    set_seed(T.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    criterion = criterion_strategy(T.criterion)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=T.shuffle, drop_last=T.drop_last)
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
                stats['loss_train'] = epoch_train_loss / len(data_loader.dataset)
                stats['accuracy_train'] = epoch_correct_train_predictions / len(data_loader.dataset)
                stats['loss_' + T.evaluation] = eval_loss 
                stats['accuracy_' + T.evaluation] = eval_accuracy 
                train_history.append(stats)
                if verbose:
                    print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_' + T.evaluation]:.4f}, Accuracy: {stats['accuracy_' + T.evaluation]:.4f}")
        # deterministic reservoirs only need history from the first epoch
        if hasattr(model, 'flush_history') and model.record_history:
            model.flush_history()
            model.record_history = False
    if verbose:
        print(f'Best loss: {best_stats}')
    model.P.logging.train_log.accuracy = best_stats['accuracy']
    model.P.logging.train_log.loss = best_stats['loss']
    model.P.logging.train_log.epoch = best_stats['epoch']
    return best_stats, model, train_history

def grid_search(yaml_path: str, dataset_init:DatasetInit=None, accuracy:AccuracyFunction=None, param_combinations: list[Params]=None):
    # dataset is re-initialized when parameters from input_layer or dataset change
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
        param_combinations = generate_param_combinations(P)
    n_config = len(param_combinations)
    n_sample = L.grid_search.n_samples
    last_dataset_params = last_input_params = dataset = best_params = None
    history = list() 
    torch.cuda.empty_cache()

    # TODO split dataset and input layer init, atm this assumes dataset_init is invariant to I.seed
    def is_equal_except_seed(this, that):
        if isinstance(that, Params):
            return all(
                getattr(this, attr) == getattr(that, attr)
                for attr in this.dict() if attr != 'seed'
            )
        return False

    pbar = tqdm(total=n_config*n_sample, desc="Grid Search Progress")
    for i, p in enumerate(param_combinations):
        for j in range(n_sample):
            t = p.model.training
            print('#'*60)
            k = generate_unique_seed(L.grid_search.seed, i, j)
            p.model.input_layer.seed = k
            p.model.reservoir_layer.seed = k 
            p.model.output_layer.seed = k 
            t.seed = k 
            if last_dataset_params != p.dataset or not is_equal_except_seed(last_input_params, p.model.input_layer):
                dataset = dataset_init(P=p)
                dataset.data['x'] = dataset.data['x'].to(device)
                dataset.data['y'] = dataset.data['y'].to(device)
                dataset.data['x_' + t.evaluation] = dataset.data['x_' + t.evaluation].to(device)
                dataset.data['y_' + t.evaluation] = dataset.data['y_' + t.evaluation].to(device)
                last_dataset_params = p.dataset
                last_input_params = p.model.input_layer
            model = best_epoch = None
            try:
                model = BooleanReservoir(p).to(device)
                best_epoch, model, _ = train_and_evaluate(model, dataset, record_stats=False, verbose=False, accuracy=accuracy)
                model.to(cpu_device)
            except (ValueError, AttributeError, TypeError) as error:
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

