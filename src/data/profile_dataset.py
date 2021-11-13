import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl

import h5py
class DS(Dataset):
    def __init__(self, X, y, conv=True):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        # print(self.X.unsqueeze(1).shape)
        if conv and len(self.X.shape) != 3:
            self.X = self.X.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataModuleClass(pl.LightningDataModule):
  def __init__(self, **params):
      super().__init__()
      self.batch_size = params['batch_size']

  def prepare_data(self):
      with h5py.File('/home/kitadam/ENR_Sven/moxie/data/processed/pedestal_profile_dataset_v2.hdf5', 'r') as file:
          X, y = file['strohman']['X'][:], file['strohman']['y'][:]


      self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
      # self.val_set =  DS(X_val, y_val, custom_scale=train_set.max_X)

  def setup(self,stage=None):
      data = (self.X, self.y)
      # train_size = int(0.75*len(self.X))
      # val_size = len(self.X) - train_size
      X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=30)
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=30)
      self.max_X = torch.max(X_train)
      if len(self.X.shape) == 3:
          X_train[:, 0] = (X_train[:, 0] / self.max_X)
          X_val[:, 0] = (X_val[:, 0] / self.max_X)
          X_test[:, 0] = (X_test[:, 0] / self.max_X)
      else:
          X_train, y_train = (X_train / self.max_X), y_train
          X_val, y_val = (X_val / self.max_X), y_val
          X_test, y_test = (X_test / self.max_X), y_test
      self.train_set = DS(X_train, y_train,)
      self.val_set = DS(X_val, y_val)
      self.test_set = DS(X_test, y_test)

  def train_dataloader(self):
      return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
      return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

  def test_dataloader(self):
      return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)
