import pickle
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset



# https://github.com/tcapelle/torchdata/blob/main/02_Custom_timeseries_datapipe.ipynb

EXP_DIR = '/home/kitadam/ENR_Sven/moxie-1/experiments/AUG_JET_REMIX/simple_vae/'
PICKLED_CLASSES_FILELOC = EXP_DIR + 'profile_data_classes.pickle'
class DATASET(Dataset):
    """
    A simple torch dataset that takes two numpy arrays, X and y, and convers them to torch tensors (if they are not already).

    TODO: Allow for squeezing or unsqueezing for just density choices? Probably don't care, can call your X as X[:, 0:1, :]

    """
    def __init__(self, X, y, mask = None, norm_dicts = None, ids=None, neseps=None):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ProfileDataModule(pl.LightningDataModule): 
    def __init__(self, data_dir: str = PICKLED_CLASSES_FILELOC, batch_size=512): 
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        with open(PICKLED_CLASSES_FILELOC, 'rb') as file: 
            self.all_profs = pickle.load(file)
    
    def normalize_profiles(self, profiles, N_mean=None, N_var=None, T_mean=None, T_var=None): 
        def standardize(x, mu, var):
            if mu is None and var is None:
                mu = x.mean(0, keepdim=True)[0]
                var = x.std(0, keepdim=True)[0]
            x_normed = (x - mu ) / var
            return x_normed, mu, var

        profiles[:, 0], N_mean, N_var = standardize(profiles[:, 0], N_mean, N_var)
        profiles[:, 1], T_mean, T_var = standardize(profiles[:, 1], T_mean, T_var)
        return profiles, N_mean, N_var, T_mean, T_var, 

    
    def setup(self, stage: Optional[str] = None): 
        AUG_PULSE_LIST = list(set([prof.pulse_id for prof in self.all_profs if prof.device == 'AUG']))
        
        train_val_ids = random.sample(AUG_PULSE_LIST, k=int(0.85*len(AUG_PULSE_LIST)))
        train_ids = random.sample(train_val_ids, k=int(0.75*len(train_val_ids)))
        val_ids = list(set(train_val_ids)  - set(train_ids))
        test_ids = list(set(AUG_PULSE_LIST) - set(train_val_ids))

        train_profs = np.array([prof.get_ML_ready_array() for prof in self.all_profs if (prof.device == 'AUG' and prof.pulse_id in train_ids)])
        val_profs = np.array([prof.get_ML_ready_array() for prof in self.all_profs if (prof.device == 'AUG' and prof.pulse_id in val_ids)])
        test_profs = np.array([prof.get_ML_ready_array() for prof in self.all_profs if (prof.device == 'AUG' and prof.pulse_id in test_ids)])
        
        train_profs, val_profs, test_profs = torch.from_numpy(train_profs), torch.from_numpy(val_profs), torch.from_numpy(test_profs)
        self.train_profs, N_mean, N_var, T_mean, T_var = self.normalize_profiles(train_profs)
        self.val_profs, *_ = self.normalize_profiles(val_profs, N_mean, N_var, T_mean, T_var)
        self.test_profs, *_ = self.normalize_profiles(test_profs, N_mean, N_var, T_mean, T_var)

        self.train_mps = torch.zeros(size=(len(train_profs), 1))
        self.val_mps = torch.zeros(size=(len(val_profs), 1))
        self.test_mps = torch.zeros(size=(len(test_profs), 1))
        
        self.train_set = DATASET(self.train_profs, self.train_mps)
        self.val_set = DATASET(self.val_profs, self.val_mps)
        self.test_set = DATASET(self.test_profs, self.test_mps)
        self.N_mean, self.N_var, self.T_mean, self.T_var = N_mean, N_var, T_mean, T_var

    def return_normalizers(self): 
        return self.N_mean, self.N_var, self.T_mean, self.T_var
    def train_dataloader(self): 
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): 
        return DataLoader(self.val_set, batch_size=self.batch_size)
    def test_dataloader(self): 
        return DataLoader(self.test_set, batch_size=self.batch_size)

def load_processed_data(): 
    with open(PICKLED_CLASSES_FILELOC, 'rb') as file: 
        all_profs = pickle.load(file)
    return all_profs

def main(): 
    datacls = ProfileDataModule()
    datacls.setup()

    for batch in datacls.train_dataloader(): 
        _prof, _ = batch 
        print(_prof.shape)
        break 

if __name__ == '__main__': 
    main()
