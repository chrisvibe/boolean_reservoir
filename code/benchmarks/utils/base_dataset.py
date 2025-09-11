import torch
from torch.utils.data import Dataset
from benchmarks.utils.parameters import DatasetParameters
from math import floor

class BaseDataset(Dataset):
    def __init__(self, D: DatasetParameters):
        self.D = D

    def split_dataset(self, split=[0.8, 0.1, 0.1]):
        split_train = split[0] if self.D.split is None else self.D.split.train
        split_dev = split[1] if self.D.split is None else self.D.split.dev
        split_test = split[2] if self.D.split is None else self.D.split.test
        assert float(sum((split_train, split_dev, split_test))) == 1.0, "Split ratios must sum to 1."
        x, y = self.data['x'], self.data['y']
        idx = torch.randperm(x.size(0))

        train_end, dev_end = floor(split_train * x.size(0)), floor((split_train + split_dev) * x.size(0))

        self.data = {
            'x': x[idx[:train_end]],
            'y': y[idx[:train_end]],
            'x_dev': x[idx[train_end:dev_end]],
            'y_dev': y[idx[train_end:dev_end]],
            'x_test': x[idx[dev_end:]],
            'y_test': y[idx[dev_end:]],
        }

    def to(self, device):
        for key in self.data.keys():
            self.data[key] = self.data[key].to(device)
        return self
    
    def save_data(self):
        self.D.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, self.D.path)

    def load_data(self):
        self.data = torch.load(self.D.path, weights_only=True)
    
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
        self.data['x'] = self.normalizer_x(self.data['x'])
        self.data['y'] = self.normalizer_y(self.data['y'])
    
    def encode_x(self):
        self.data['x'] = self.encoder_x(self.data['x']) 