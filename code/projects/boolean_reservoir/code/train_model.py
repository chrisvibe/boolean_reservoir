from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import pandas as pd
from tqdm import tqdm
from projects.boolean_reservoir.code.utils import set_seed, generate_unique_seed, CudaMemoryManager
from projects.boolean_reservoir.code.reservoir import BooleanReservoir
from projects.boolean_reservoir.code.graph_visualizations_dash import *
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.visualizations import *
from time import sleep

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
    mem = CudaMemoryManager()
    mem.manage_memory()
    if model is None:
        model = BooleanReservoir(params=parameter_override, load_path=yaml_or_checkpoint_path).to(mem.device)
    P = model.P

    # Init data
    dataset = dataset_init(P).to(mem.device)
    _, model, train_history = train_and_evaluate(model, dataset, record_stats=True, verbose=True, accuracy=accuracy)
    if save_model:
        model.save()
    return P, model, dataset, train_history

def train_and_evaluate(model: BooleanReservoir, dataset: Dataset, record_stats=False, verbose=False, accuracy: AccuracyFunction=EuclideanDistanceAccuracy()):
    T = model.P.model.training
    set_seed(T.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=T.learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=T.learning_rate, eps=1e-4, weight_decay=1e-3) # TODO consider upgrade of optimizer + options for stability liks epislon and weight_decay. optimizer for boolean readout?
    optimizer = torch.optim.AdamW(model.parameters(), lr=T.learning_rate, weight_decay=1e-1, eps=1e-3)
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
        if hasattr(model, 'flush_history') and model.record_history:
            model.flush_history()
            model.record_history = False
    if verbose:
        print(f'Best loss: {best_stats}')
    model.P.L.train_log.accuracy = best_stats['accuracy']
    model.P.L.train_log.loss = best_stats['loss']
    model.P.L.train_log.epoch = best_stats['epoch']
    return best_stats, model, train_history
