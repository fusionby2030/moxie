from VAE import VanillaVAE

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


with h5py.File('/home/kitadam/ENR_Sven/moxie/data/processed/pedestal_profile_for_nesep.hdf5', 'r') as file:
    X, y = file['strohman']['X'][:], file['strohman']['y'][:]

class DS(Dataset):
    def __init__(self, X, y, scale=True, custom_scale=None):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y
        self.X, self.y= self.X.type(torch.DoubleTensor),self.y.type(torch.DoubleTensor)
        if scale:
            if custom_scale is None:
                self.max_X = torch.max(self.X)
            else:
                self.max_X = custom_scale
            self.X = self.X / self.max_X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
train_set = DS(X_train, y_train)
val_set =  DS(X_val, y_val, custom_scale=train_set.max_X)
train_dl = DataLoader(train_set, batch_size=1024, shuffle=True)
val_dl = DataLoader(val_set, batch_size=1024, shuffle=True)

model = VanillaVAE(in_dim = 63, latent_dim=5).double()

with torch.no_grad():
    for i, (inputs_val, targets_val) in enumerate(val_dl):
        outputs_val = model(inputs_val)
        print(inputs_val)
        print(outputs_val)
        break
