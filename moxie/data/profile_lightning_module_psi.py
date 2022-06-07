import pytorch_lightning as pl
from .profile_dataset_torch import DATASET_AK
from .utils_ import *
import pickle  # data loading issues
import torch
from torch.utils.data import DataLoader
import numpy as np

def replace_q95_with_qcly(mp_set):
    mu_0 = 1.25663706e-6 # magnetic constant
    mp_set[:, 0] = ((1 + 2*mp_set[:, 6]**2) / 2.0) * (2*mp_set[:, 9]*torch.pi*mp_set[:, 2]**2) / (mp_set[:, 1] * mp_set[:, 8] * mu_0)
    return mp_set

class PLDATAMODULE_AK(pl.LightningDataModule):
    """
    pl.lightning datamodule class, which will help with our dataloading needs :--)
    # TODO: Implement a mask or not mask ask.
    """
    def __init__(self, data_dir: str = '', num_workers: int = 1, batch_size: int = 512, dataset_choice='ALL', elm_style_choice='simple', **params):
        super().__init__()

        self.batch_size = batch_size # batch size of dataloaders
        self.file_loc = data_dir # Location of massive dict
        self.num_workers = num_workers # CPU's to use in dataloaders
        self.mu_T, self.var_T = None, None # We are normalizing shit here you dig.
        self.mu_D, self.var_D = None, None # Density normailizing constants
        self.mu_MP, self.var_MP = None, None # Machine params normailizing constant
        self.elm_style_choice = elm_style_choice
        if elm_style_choice == 'simple':
            self.in_channels = 3
        else:
            self.in_channels = 2

        # Sometimes we want to pin the memory to a GPU so this can get passed as a kwarg.
        if 'pin_memory' in params.keys():
            self.pin_memory = params['pin_memory']
        else:
            self.pin_memory = False

    def prepare_data(self):
        # Grab the dataset
        with open(self.file_loc, 'rb') as file:
            full_dict = pickle.load(file)
            train_X, train_y, train_mask, train_ids, train_elms = full_dict['train']['profiles'],  full_dict['train']['machine_parameters'],  full_dict['train']['profiles_mask'], full_dict['train']['timings'], full_dict['train']['elm_fractions']


            val_X, val_y, val_mask, val_ids, val_elms = full_dict['val']['profiles'],  full_dict['val']['machine_parameters'], full_dict['val']['profiles_mask'], full_dict['val']['timings'], full_dict['val']['elm_fractions']
            
            test_X, test_y, test_mask, test_ids, test_elms = full_dict['test']['profiles'],  full_dict['test']['machine_parameters'], full_dict['test']['profiles_mask'], full_dict['test']['timings'], full_dict['test']['elm_fractions']

        # Convert to float torch tensors
        self.X_train, self.y_train = torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float()
        self.X_val, self.y_val = torch.from_numpy(val_X).float(), torch.from_numpy(val_y).float()
        self.X_test, self.y_test = torch.from_numpy(test_X).float(), torch.from_numpy(test_y).float()

        # The mask is tricky, as it is originally a bool list, which is True for all vals to be masked.
        # Then is is converted into numpy (through the padding procedure), which results in a 0 for all vals to be masked, then 1 for vals not to be masked.
        # here we convert them to torch bool tensors again
        # and unsqueeze them to match the same dimensionality as the profiles [#datapoints, 2, #spatial resoultion]
        # This is necesary for the mask_fill functions.

        self.train_mask, self.val_mask, self.test_mask = torch.from_numpy(train_mask) > 0, torch.from_numpy(val_mask) > 0, torch.from_numpy(test_mask) > 0
        self.train_mask, self.val_mask, self.test_mask = self.train_mask.unsqueeze(1),  self.val_mask.unsqueeze(1), self.test_mask.unsqueeze(1)
        self.train_mask, self.val_mask, self.test_mask = torch.repeat_interleave(self.train_mask, self.in_channels, 1 ), torch.repeat_interleave(self.val_mask, self.in_channels, 1), torch.repeat_interleave(self.test_mask, self.in_channels, 1)

        # Sanity check(s)
        assert torch.isnan(self.y_train).any() == False

        # Normalize the profiles

        self.X_train, self.mu_D, self.var_D, self.mu_T, self.var_T = normalize_profiles(self.X_train)
        self.X_val = normalize_profiles(self.X_val, self.mu_T, self.var_T, self.mu_D, self.var_D)
        self.X_test = normalize_profiles(self.X_test, self.mu_T, self.var_T, self.mu_D, self.var_D)
        if self.elm_style_choice == 'simple':
            ELM_train = torch.repeat_interleave(torch.from_numpy(train_elms).unsqueeze(1), 20, 1).unsqueeze(1).float()
            ELM_val = torch.repeat_interleave(torch.from_numpy(val_elms).unsqueeze(1), 20, 1).unsqueeze(1).float()
            ELM_test = torch.repeat_interleave(torch.from_numpy(test_elms).unsqueeze(1), 20, 1).unsqueeze(1).float()
            self.X_train, self.X_val, self.X_test = torch.concat((self.X_train, ELM_train), 1), torch.concat((self.X_val, ELM_val), 1), torch.concat((self.X_test, ELM_test), 1)
        # Normalize the machine parameters
        self.y_train, self.y_val, self.y_test = replace_q95_with_qcly(self.y_train), replace_q95_with_qcly(self.y_val), replace_q95_with_qcly(self.y_test)
        self.y_train, self.mu_MP, self.var_MP = standardize_simple(self.y_train)
        self.y_val = standardize_simple(self.y_val, self.mu_MP, self.var_MP)
        self.y_test =  standardize_simple(self.y_test, self.mu_MP, self.var_MP)

        if self.elm_style_choice == 'simple':
            self.y_train = torch.column_stack((self.y_train, torch.from_numpy(train_elms).float()))
            self.y_val = torch.column_stack((self.y_val, torch.from_numpy(val_elms).float()))
            self.y_test = torch.column_stack((self.y_test, torch.from_numpy(test_elms).float()))

        # make sure the ids are there
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

    def setup(self, stage=None):
        self.train_set = DATASET_AK(self.X_train, self.y_train, mask = self.train_mask, ids = self.train_ids)
        self.val_set = DATASET_AK(self.X_val, self.y_val, mask = self.val_mask, ids = self.val_ids)
        self.test_set = DATASET_AK(self.X_test, y = self.y_test, mask = self.test_mask, ids = self.test_ids)

    def get_temperature_norms(self,device=None):
        if device is not None:
            return self.mu_T.to(device), self.var_T.to(device)
        return self.mu_T, self.var_T

    def get_density_norms(self, device=None):
        if device is not None:
            return self.mu_D.to(device), self.var_D.to(device)
        return self.mu_D, self.var_D

    def get_machine_norms(self, device=None):
        if device is not None:
            return self.mu_MP.to(device), self.var_MP.to(device)
        return self.mu_MP, self.var_MP

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)