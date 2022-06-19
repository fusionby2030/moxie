import pickle
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt 

def parse_set(set): 
    t_0, t_1 = set
    profs_t1 = t_0.get_ML_ready_array()
    profs_t2 = t_1.get_ML_ready_array()

    return [profs_t1, profs_t2]

def main(): 
    datacls = ProfileSetModule()
    datacls.setup()

    # all_pulse_sets = datacls.all_pulse_sets
    # train_pulse_sets = datacls.train_set
    """
    train_set = np.array(train_pulse_sets.sets)
    train_set_densities = np.array([profile.ne for profile in train_set[:, 0]])
    train_set_temperatures = np.array([profile.te for profile in train_set[:, 0]])
    ne_mu, ne_var = np.mean(train_set_densities, 0), np.std(train_set_densities, 0)
    te_mu, te_var = np.mean(train_set_temperatures, 0), np.std(train_set_temperatures, 0)
    """
    
    train_iter = datacls.train_dataloader()
    for n, batch in enumerate(train_iter): 
        
        t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
        plt.plot(t_0_batch[0, 0, :], color='blue', lw=5)
        plt.plot(t_1_batch[0, 0, :], color='green', lw=5)

        plt.plot(t_0_batch[1, 0, :], color='dodgerblue', ls='--', lw=5)
        plt.plot(t_1_batch[1, 0, :], color='forestgreen', ls='--', lw=5)
        plt.show()
        break
    

# https://github.com/tcapelle/torchdata/blob/main/02_Custom_timeseries_datapipe.ipynb


EXP_DIR = '/home/kitadam/ENR_Sven/test_moxie/experiments/AUG_JET_REMIX/'
PICKLED_CLASSES_FILELOC = EXP_DIR + 'profile_data_classes.pickle'

def setup_data(all_profs = None, train_test_split=True) -> List: 
    if all_profs is None: 
        with open(PICKLED_CLASSES_FILELOC, 'rb') as file: 
            all_profs = pickle.load(file)
    AUG_PULSE_LIST = list(set([prof.pulse_id for prof in all_profs if prof.device == 'AUG']))
    
    if train_test_split: 
        train_val_ids = random.sample(AUG_PULSE_LIST, k=int(0.85*len(AUG_PULSE_LIST)))
        train_ids = random.sample(train_val_ids, k=int(0.75*len(train_val_ids)))
        val_ids = list(set(train_val_ids)  - set(train_ids))
        test_ids = list(set(AUG_PULSE_LIST) - set(train_val_ids))

        all_pulse_sets, train_pulse_sets, val_pulse_sets, test_pulse_sets = [], [], [], []
        for pulse_id in AUG_PULSE_LIST: 
            pulse = PULSE(device='AUG', pulse_id=pulse_id, profiles=[prof for prof in all_profs if prof.pulse_id == pulse_id])
            sets = pulse.get_time_obervables()
            all_pulse_sets.extend(sets)
            if pulse_id in train_ids: 
                train_pulse_sets.extend(sets)
            elif pulse_id in val_ids: 
                val_pulse_sets.extend(sets)
            else: 
                test_pulse_sets.extend(sets)
        return all_pulse_sets, (train_pulse_sets, val_pulse_sets, test_pulse_sets)
    else: 
        all_pulses =[]
        for pulse_id in AUG_PULSE_LIST: 
            pulse_profs = [prof for prof in all_profs if prof.pulse_id == pulse_id]
            all_pulses.append(PULSE(device='AUG', pulse_id=pulse_id, profiles=pulse_profs))
        
        all_pulse_sets = []
        for pulse in all_pulses: 
            pulse_sets = pulse.get_time_obervables()
            all_pulse_sets.extend(pulse_sets)
        return all_pulse_sets


class PROFILE: 
    device: str 
    pulse_id: int 
    time_stamp: float 
    ne: Union[List[float], np.array]
    te: Union[List[float], np.array]
    dne: Union[List[float], np.array]
    dte: Union[List[float], np.array]
    radius: Union[List[float], np.array]
    elm_frac: float = None
    def get_ML_ready_array(self, from_SOL: bool = True) -> np.array:
        if from_SOL: 
            if self.device == 'AUG':
                beg_idx = -60
                end_idx = -10
            elif self.device == 'JET': 
                beg_idx = -20
                end_idx = -1
        else: 
            beg_idx = 0
            end_idx = -1
        return np.vstack((1e-19*self.ne[beg_idx:end_idx], self.te[beg_idx:end_idx]))

@dataclass 
class PULSE: 
    device: str 
    pulse_id: int 
    profiles: List[PROFILE]

    def get_time_obervables(self, window_size=2):
        # return a list of profile pairs, one after the other 
        current_set = []
        full_set = []
        it = iter(self.profiles)
        while True: 
            try: 
                while len(current_set) < window_size: 
                    current_set.append(next(it))
                full_set.append(current_set)
                current_set = [current_set[1]]
            except StopIteration: 
                return full_set    


@dataclass 
class DATASET(Dataset):
    sets: List[List[PROFILE]]
    norms: Tuple = None 
    def denormalize_profiles(self, profiles): 
        def de_standardize(x, mu, var): 
            return x*var + mu
        N_mean, N_var, T_mean, T_var = self.norms
        N_mean, N_var, T_mean, T_var = N_mean[-60:-10], N_var[-60:-10], T_mean[-60:-10], T_var[-60:-10]
        profiles[:, 0, :] = de_standardize(profiles[:, 0, :], N_mean, N_var)
        profiles[:, 1, :] = de_standardize(profiles[:, 1, :], T_mean, T_var)
        return profiles
    def normalize_profiles(self, profiles): 
        def standardize(x, mu, var):
            if mu is None and var is None:
                mu = x.mean(0, keepdim=True)[0]
                var = x.std(0, keepdim=True)[0]
            x_normed = (x - mu ) / var
            return x_normed, mu, var
        N_mean, N_var, T_mean, T_var = self.norms
        N_mean, N_var, T_mean, T_var = N_mean[-60:-10], N_var[-60:-10], T_mean[-60:-10], T_var[-60:-10]
        profiles[0, :], _, _ = standardize(profiles[0, :], N_mean, N_var)
        profiles[1, :], _, _ = standardize(profiles[1, :], T_mean, T_var)
        return profiles#,  N_mean, N_var, T_mean, T_var, 
    def get_normalizing(self, from_SOL=True): 
        # Should really not do the from_SOL here...
        train_set = np.array(self.sets)
        train_set_densities = np.array([1e-19*profile.ne for profile in train_set[:, 0]])
        train_set_temperatures = np.array([profile.te for profile in train_set[:, 0]])
        ne_mu, ne_var = np.mean(train_set_densities, 0), np.std(train_set_densities, 0)
        te_mu, te_var = np.mean(train_set_temperatures, 0), np.std(train_set_temperatures, 0)
        self.norms = (ne_mu, ne_var, te_mu, te_var)
        return ne_mu, ne_var, te_mu, te_var
    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx, normalize=True):

        
        item = self.sets[idx]
        t_0, t_1 = item 
        profs_t1 = t_0.get_ML_ready_array()
        profs_t2 = t_1.get_ML_ready_array()
        if normalize == True: 
            profs_t1 = self.normalize_profiles(profs_t1)
            profs_t2 = self.normalize_profiles(profs_t2)
        return np.array([profs_t1, profs_t2])


class ProfileSetModule(pl.LightningDataModule): 
    def __init__(self, data_dir: str = PICKLED_CLASSES_FILELOC, batch_size=512): 
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        with open(PICKLED_CLASSES_FILELOC, 'rb') as file: 
            self.all_profs = pickle.load(file)

    
    def setup(self, stage: Optional[str] = None): 
        self.all_pulse_sets, (train_set, val_set, test_set) = setup_data(self.all_profs)
        
        self.train_pulse_sets = train_set # This is a List[List[PROFILE]]
        self.train_set = DATASET(train_set)
        self.ne_mu, self.ne_var, self.te_mu, self.te_var = self.train_set.get_normalizing()
        self.val_set = DATASET(val_set, norms=(self.ne_mu, self.ne_var, self.te_mu, self.te_var))
        self.test_set = DATASET(test_set, norms=(self.ne_mu, self.ne_var, self.te_mu, self.te_var))
        
    def get_normalizers(self): 
        return self.ne_mu, self.ne_var, self.te_mu, self.te_var 
    def train_dataloader(self): 
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): 
        return DataLoader(self.val_set, batch_size=self.batch_size)
    def test_dataloader(self): 
        return DataLoader(self.test_set, batch_size=self.batch_size)



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


if __name__ == '__main__': 
    main()
