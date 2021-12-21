import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl

import h5py


def standardize_simple(x, mu=None, var=None):
    if mu is not None and var is not None:
        x_normed = (x - mu ) / var
        return x_normed
    else:
        mu = x.mean(0, keepdim=True)[0]
        var = x.std(0, keepdim=True)[0]
        x_normed = (x - mu ) / var
        return x_normed, mu, var


class DS(Dataset):
    def __init__(self, X, y, problem='strohman'):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        if problem == 'strohman':
            self.X = self.X.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataModuleClass(pl.LightningDataModule):
    """
    The Data Module Class, which will help with our data loading needs

    Parameters
    ----------

    batch_size: int
        Batch size of data loaders
    num_workers: int
        if doing MP for dataloading, specify how many cpus will be used
    data_dir: str
        where is the file with the train-val-test data is located
    problem: str
        Depends on data, but can be either 'strohman' or 'density_and_temperature'
    """

    def __init__(self, data_dir: str = '../processed/profile_database_v1_psi22.hdf5', num_workers: int =1, batch_size: int = 512, **params):
        super().__init__()
        self.batch_size = batch_size
        self.file_loc = data_dir
        self.num_workers = num_workers
        if 'problem' in params.keys():
            self.problem = params['problem']
        else:
            self.problem = 'strohman'

    def prepare_data(self):
        with h5py.File(self.file_loc, 'r') as file:
            group = file['processed_datasets/PSI22/density']
            X_train, y_train = group['train']['X'][:], group['train']['y'][:]
            X_val, y_val = group['valid']['X'][:], group['valid']['y'][:]
            X_test, y_test = group['test']['X'][:], group['test']['y'][:]

        self.X_train, self.y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        self.X_val, self.y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
        self.X_test, self.y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

        self.y_train[torch.isnan(self.y_train)] = 0.0
        self.y_val[torch.isnan(self.y_val)] = 0.0
        self.y_test[torch.isnan(self.y_test)] = 0.0

        assert torch.isnan(self.y_train).any() == False
        assert torch.isnan(self.X_train).any() == False
    def setup(self,stage=None):

        self.max_X = torch.max(self.X_train)
        if len(self.X_train.shape) == 3:
            self.X_train[:, 0] = (X_train[:, 0] / self.max_X)
            self.X_val[:, 0] = (X_val[:, 0] / self.max_X)
            self.X_test[:, 0] = (X_test[:, 0] / self.max_X)
        else:
            self.X_train, self.y_train = (self.X_train / self.max_X), self.y_train
            self.X_val, self.y_val = (self.X_val / self.max_X), self.y_val
            self.X_test, self.y_test = (self.X_test / self.max_X), self.y_test

        # Normalize the machine parameters
        self.y_train, mu_train, var_train = standardize_simple(self.y_train)
        self.y_val = standardize_simple(self.y_val, mu_train, var_train)
        self.y_test = standardize_simple(self.y_test, mu_train, var_train)

        # Nesep is in the machine parameters so we have to take it out (last column)

        self.y_train, self.y_val, self.y_test = self.y_train[:, :13], self.y_val[:, :13], self.y_test[:, :13]
        self.mu_normalizer = mu_train
        self.var_normalizer = var_train

        self.y_train, self.y_val, self.y_test = self.y_train.float(), self.y_val.float(), self.y_test.float()

        self.train_set = DS(self.X_train, self.y_train, self.problem)
        self.val_set = DS(self.X_val, self.y_val, self.problem)
        self.test_set = DS(self.X_test, self.y_test, self.problem)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
