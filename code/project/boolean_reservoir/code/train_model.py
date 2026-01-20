from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.graph_visualizations_dash import *
from project.boolean_reservoir.code.parameter import * 
from project.boolean_reservoir.code.visualization import *

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
    critertion_map = {
        'MSE': nn.MSELoss,
        'BCE': nn.BCELoss
    }
    if strategy not in critertion_map:
        raise ValueError(f"Unsupported criterion type: {strategy}. Available: {list(critertion_map.keys())}")
    return critertion_map[strategy]()

# TODO handle compiling of model in train_and_evaluate?
def train_single_model(yaml_or_checkpoint_path='', parameter_override:Params=None, model=None, save_model=True, dataset_init: DatasetInit=None, accuracy: AccuracyFunction=None, ignore_gpu=False, compile_model=False, reset_dynamo=True):
    if model is None:
        model = BooleanReservoir(params=parameter_override, load_path=yaml_or_checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not ignore_gpu else "cpu")
    model.to(device)
    P = model.P

    # Init data
    dataset = dataset_init(P).to(device)
    _, model, train_history = train_and_evaluate(model, dataset, record_stats=True, verbose=True, accuracy=accuracy)
    if save_model:
        model.save()
    return P, model, dataset, train_history

def optimizer_strategy(p: DynamicParams, model: nn.Module):
    """Factory function to create optim objects"""
    opt_map = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }
    if p.name not in opt_map:
        raise ValueError(f"Unsupported optimizer type: {p.name}. Available: {list(opt_map.keys())}")
 
    return p.call(opt_map[p.name], params=model.parameters())

# Assumption: the .to(device) call is made outside this function since the data is small
def train_and_evaluate(model: BooleanReservoir, dataset: Dataset, record_stats=False, verbose=False, accuracy: AccuracyFunction=EuclideanDistanceAccuracy()):
    T = model.P.M.T
    set_seed(T.seed)
    optimizer = optimizer_strategy(T.optim, model)
    criterion = criterion_strategy(T.criterion)
    data_loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=T.shuffle, drop_last=T.drop_last)
    m = len(data_loader) * T.batch_size if T.drop_last else len(dataset)
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
            epoch_train_loss += loss.item() * len(x)
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
                stats['loss_train'] = epoch_train_loss / m # average loss per sample
                stats['accuracy_train'] = epoch_correct_train_predictions / m # average accuracy per sample 
                stats['loss_' + T.evaluation] = eval_loss 
                stats['accuracy_' + T.evaluation] = eval_accuracy 
                train_history.append(stats)
                if verbose:
                    print(f"Epoch: {stats['epoch']:0{len(str(T.epochs))}d}/{T.epochs}, Loss: {stats['loss_' + T.evaluation]:.4f}, Accuracy: {stats['accuracy_' + T.evaluation]:.4f}")
        # deterministic reservoirs only need history from the first epoch
        if hasattr(model, 'flush_history') and model.record:
            model.flush_history()
            model.record = False
    if verbose:
        print(f'Best loss: {best_stats}')
    model.P.L.train = TrainLog(**best_stats)
    return best_stats, model, train_history
