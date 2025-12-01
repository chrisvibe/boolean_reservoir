import torch
from torch import nn
from torch.utils.data import Dataset
from benchmark.utils.parameters import DatasetParameters
from math import floor

class BaseDataset(nn.Module, Dataset):
    def __init__(self, D: DatasetParameters):
        super().__init__()
        self.D = D
        self.data = {k: None for k in ['x', 'y', 'x_dev', 'y_dev', 'x_test', 'y_test']}
    
    def to(self, device):
        super().to(device)
        self._sync_buffers_to_dict()
        return self
    
    def set_data(self, data_dict):
        """Set initial data and register as buffers"""
        for key, tensor in data_dict.items():
            if hasattr(self, key):
                delattr(self, key)
            self.register_buffer(key, tensor)
        self._sync_buffers_to_dict()
    
    def _sync_buffers_to_dict(self):
        """Internal: sync registered buffers back to data dict"""
        for key in self.data.keys():
            if hasattr(self, key):
                self.data[key] = getattr(self, key)
    
    def split_dataset(self, split=[0.8, 0.1, 0.1]):
        split_train = split[0] if self.D.split is None else self.D.split.train
        split_dev = split[1] if self.D.split is None else self.D.split.dev
        split_test = split[2] if self.D.split is None else self.D.split.test
        assert float(sum((split_train, split_dev, split_test))) == 1.0, "Split ratios must sum to 1."
        
        x, y = self.data['x'], self.data['y']
        idx = torch.randperm(x.size(0))
        train_end, dev_end = floor(split_train * x.size(0)), floor((split_train + split_dev) * x.size(0))
        
        split_data = {
            'x': x[idx[:train_end]],
            'y': y[idx[:train_end]],
            'x_dev': x[idx[train_end:dev_end]],
            'y_dev': y[idx[train_end:dev_end]],
            'x_test': x[idx[dev_end:]],
            'y_test': y[idx[dev_end:]],
        }
        
        for key, tensor in split_data.items():
            if hasattr(self, key):
                delattr(self, key)
            self.register_buffer(key, tensor)
        self._sync_buffers_to_dict()
    
    def save_data(self):
        self.D.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, self.D.path)
    
    def load_data(self):
        loaded_data = torch.load(self.D.path, weights_only=True)
        self.set_data(loaded_data)
    
    def set_normalizer_x(self, normalizer_x):
        self.normalizer_x = normalizer_x
    
    def set_normalizer_y(self, normalizer_y):
        self.normalizer_y = normalizer_y
    
    def set_encoder_x(self, encoder_x):
        self.encoder_x = encoder_x
    
    def __len__(self):
        return self.data['x'].size(0)
    
    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        return x, y
    
    def normalize(self):
        self.x = self.normalizer_x(self.x)
        self.y = self.normalizer_y(self.y)
        self._sync_buffers_to_dict()
    
    def encode_x(self):
        self.x = self.encoder_x(self.x).to(torch.uint8)
        self._sync_buffers_to_dict()