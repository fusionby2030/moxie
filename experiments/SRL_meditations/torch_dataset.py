from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union, List

import numpy as np

import pickle
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
from python_dataset import PULSE, PROFILE, MACHINEPARAMETER, ML_ENTRY
import random 
import torch 

PERSONAL_DATA_DIR = '/home/kitadam/ENR_Sven/moxie/data/raw/'
PERSONAL_DATA_DIR_PROC = '/home/kitadam/ENR_Sven/moxie/data/processed/'

PICKLED_AUG_PULSES = PERSONAL_DATA_DIR_PROC + 'AUG_PDB_PYTHON_PULSES.pickle'


@dataclass
class PULSE_DATASET(Dataset): 
    pulses: List[PULSE]
    t0_profs: Union[List[np.array], np.array] = field(default_factory=list)
    t1_profs: Union[List[np.array], np.array] = field(default_factory=list)
    t0_mps: Union[List[np.array], np.array] = field(default_factory=list)
    t1_mps: Union[List[np.array], np.array] = field(default_factory=list)
    mps_delta: Union[List[np.array], np.array] = field(default_factory=list)
    norms: Union[List, Tuple, np.array] = None
    def __post_init__(self): 
        print('Creating ML Ready array')
        for pulse in self.pulses: 
            profiles, mps = pulse.get_ML_ready_array()
            t0, t1 = profiles[:-1, :, :], profiles[1:, :, :]
            mps_t0, mps_t1 = mps[:-1, :], mps[1:, :]
            self.t0_profs.append(t0)
            self.t1_profs.append(t1)
            self.t1_mps.append(mps_t1)
            self.t0_mps.append(mps_t0)
            self.mps_delta.append(mps_t1 - mps_t0)
        self.t0_profs = np.concatenate(self.t0_profs, axis=0, dtype=np.double)
        self.t1_profs = np.concatenate(self.t1_profs, axis=0, dtype=np.double)
        self.t0_mps = np.concatenate(self.t0_mps, axis=0, dtype=np.double)
        self.t1_mps = np.concatenate(self.t1_mps, axis=0, dtype=np.double)
        self.mps_delta = np.concatenate(self.mps_delta, axis=0, dtype=np.double)
        if self.norms == None: 
            self.get_normalizing()
            self.normalize_all()

    def denormalize_profiles(self, profiles): 
        def de_standardize(x, mu, var): 
            if isinstance(x, torch.Tensor): 
                mu, var = torch.from_numpy(mu), torch.from_numpy(var)
            return x*var + mu
        N_mean, N_var, T_mean, T_var, _, _ = self.norms
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
        N_mean, N_var, T_mean, T_var, _, _ = self.norms
        profiles[:, 0, :], _, _ = standardize(profiles[:, 0, :], N_mean, N_var)
        profiles[:, 1, :], _, _ = standardize(profiles[:, 1, :], T_mean, T_var)
        # return profiles
    def normalize_mps(self, mps): 
        def standardize(x, mu, var):
            if mu is None and var is None:
                mu = x.mean(0, keepdim=True)[0]
                var = x.std(0, keepdim=True)[0]
            x_normed = (x - mu ) / var
            return x_normed, mu, var
        _, _, _, _, MP_mu, MP_var = self.norms
        mps, _, _ = standardize(mps, MP_mu, MP_var)
        return mps
    def denormalize_mps(self, mps): 
        def de_standardize(x, mu, var):
            if isinstance(x, torch.Tensor): 
                mu, var = torch.from_numpy(mu), torch.from_numpy(var)
            x_normed = x*var + mu
            return x_normed, mu, var
        _, _, _, _, MP_mu, MP_var = self.norms
        mps, _, _ = de_standardize(mps, MP_mu, MP_var)
        return mps
    def get_normalizing(self):       
        train_set_profs = self.t0_profs
        train_set_densities = train_set_profs[:, 0, :] 
        train_set_temperatures = train_set_profs[:, 1, :]
        ne_mu, ne_var = np.mean(train_set_densities, 0), np.std(train_set_densities, 0)
        te_mu, te_var = np.mean(train_set_temperatures, 0), np.std(train_set_temperatures, 0)
        mp_mu, mp_var = np.mean(self.t0_mps ,0), np.std(self.t0_mps,0)
        self.norms = (ne_mu, ne_var, te_mu, te_var, mp_mu, mp_var)

    def normalize_all(self): 
        self.normalize_profiles(self.t0_profs)
        self.normalize_profiles(self.t1_profs)
        self.t0_mps = self.normalize_mps(self.t0_mps)
        self.t1_mps = self.normalize_mps(self.t1_mps)
        self.mps_delta = self.normalize_mps(self.mps_delta)

    def __len__(self): 
        return len(self.t0_profs)

    def __getitem__(self, index: Any, normalize=True) -> Tuple[np.array]:
        profs_t0, profs_t1, mps_t0, mps_t1, mps_delta = self.t0_profs[index], self.t1_profs[index], self.t0_mps[index], self.t1_mps[index], self.mps_delta[index]
        return profs_t0, profs_t1, mps_t0, mps_t1, mps_delta


def main(): 
    # create_pulse_classes_from_raw() 
    AUG_PULSES = load_classes_from_pickle()
    random.shuffle(AUG_PULSES)

    train_count, val_count = int(0.9*len(AUG_PULSES)), int(0.1*len(AUG_PULSES))
    train_pulses = np.array(AUG_PULSES)[:train_count]
    test_pulses = np.array(AUG_PULSES)[train_count:]
    val_pulses = train_pulses[:val_count]
    train_pulses = train_pulses[val_count:]
    train_set = PULSE_DATASET(pulses=train_pulses)
    
    train_dl = DataLoader(train_set, batch_size=10)
    
    for n, batch in enumerate(train_dl): 
        import matplotlib.pyplot as plt 
        profs_t0, profs_t1, mps_t0, mps_t1 = batch 
        plt.plot(profs_t0[0, 0, :], color='blue', lw=5)
        plt.plot(profs_t1[0, 0, :], color='green', lw=5)

        plt.plot(profs_t0[1, 0, :], color='dodgerblue', ls='--', lw=5)
        plt.plot(profs_t1[1, 0, :], color='forestgreen', ls='--', lw=5)

        plt.plot(profs_t0[2, 0, :], color='red', ls=':', lw=5)
        plt.plot(profs_t1[2, 0, :], color='pink', ls=':', lw=5)

        plt.plot(profs_t0[3, 0, :], color='orange', ls='-.', lw=2)
        plt.plot(profs_t1[3, 0, :], color='salmon', ls='-.',lw=2)

        plt.show()
        break 
    """
    for pulse in train_pulses: 
        profiles, mps = pulse.get_ML_ready_array()
        print(profiles.shape, mps.shape)
        t0, t1 = profiles[:-1, :, :], profiles[1:, :, :]
        print(t0.shape, t1.shape)
        
        import matplotlib.pyplot as plt 
        plt.plot(t0[0, 0, :], color='blue', lw=5)
        plt.plot(t1[0, 0, :], color='green', lw=5)

        plt.plot(t0[1, 0, :], color='dodgerblue', ls='--', lw=5)
        plt.plot(t1[1, 0, :], color='forestgreen', ls='--', lw=5)

        plt.plot(t0[2, 0, :], color='red', ls=':', lw=5)
        plt.plot(t1[2, 0, :], color='pink', ls=':', lw=5)

        plt.show()

        plt.plot(t0[-2, 0, :], color='blue', lw=5)
        plt.plot(t1[-2, 0, :], color='green', lw=5)

        plt.plot(t0[-1, 0, :], color='dodgerblue', ls='--', lw=5)
        plt.plot(t1[-1, 0, :], color='forestgreen', ls='--', lw=5)
        plt.show()
        break 
    """
def load_classes_from_pickle() -> List[PULSE]: 
    with open(PICKLED_AUG_PULSES, 'rb') as file: 
        AUG_PULSES = pickle.load(file)
    return AUG_PULSES


def save_proccessed_classes(list_of_classes, file_name:str): 
    with open(PERSONAL_DATA_DIR_PROC + file_name, 'wb') as file: 
        pickle.dump(list_of_classes, file)

if __name__ == '__main__': 
    main()