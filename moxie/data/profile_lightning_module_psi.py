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
        def convert_to_tensors(data):
            X, y, mask, ids, elms = data
            X, y, elms = torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(elms).float()
            mask = torch.from_numpy(mask) > 0
            mask  = torch.repeat_interleave(mask.unsqueeze(1), self.in_channels, 1)
            if self.elm_style_choice == 'simple':
                ELM_data = torch.repeat_interleave(elms.unsqueeze(1), 20, 1).unsqueeze(1)
                X = torch.concat((X, ELM_data), 1)
                y = torch.column_stack((y, elms))
            elif self.elm_style_choice == 'mp_only':
                y = torch.column_stack((y, elms))
            else: pass
            return X, y, mask, ids, elms
        def normalize_tensors(data,prof_norms={'mu_T': None, 'var_T': None, 'mu_D': None, 'var_D': None}, mp_norms={'mu': None, 'var': None}):
            X, y, mask, ids, elms = data
            X_norm, y_norm = torch.clone(X), torch.clone(y)
            X_norm, *prof_norms_new = normalize_profiles(X, **prof_norms)
            y_norm = replace_q95_with_qcly(y_norm)
            if self.elm_style_choice in ['simple', 'mp_only']:
                y_norm[:, :-1], *mp_norms_new  = standardize_simple(y_norm[:, :-1], **mp_norms)
            else:
                y_norm, *mp_norms_new  = standardize_simple(y_norm, **mp_norms)
            return (X_norm, y_norm, mask, ids, elms), (dict(zip(prof_norms.keys(), prof_norms_new)), (dict(zip(mp_norms.keys(), mp_norms_new))))

        with open(self.file_loc, 'rb') as file:
            full_dict = pickle.load(file)
            # train_data = train_X, train_y, train_mask, train_ids, train_elms
            train_data = full_dict['train']['profiles'],  full_dict['train']['machine_parameters'],  full_dict['train']['profiles_mask'], full_dict['train']['timings'], full_dict['train']['elm_fractions']
            val_data = full_dict['val']['profiles'],  full_dict['val']['machine_parameters'], full_dict['val']['profiles_mask'], full_dict['val']['timings'], full_dict['val']['elm_fractions']
            test_data = full_dict['test']['profiles'],  full_dict['test']['machine_parameters'], full_dict['test']['profiles_mask'], full_dict['test']['timings'], full_dict['test']['elm_fractions']

        train_data_tensor, val_data_tensor, test_data_tensor = convert_to_tensors(train_data), convert_to_tensors(val_data), convert_to_tensors(test_data)
        train_data_norm, (prof_norms, mp_norms) = normalize_tensors(train_data_tensor)
        val_data_norm, _ = normalize_tensors(val_data_tensor, prof_norms, mp_norms)
        test_data_norm, _ = normalize_tensors(test_data_tensor, prof_norms, mp_norms)
        self.X_train, self.y_train, self.train_mask, self.train_ids, self.train_elms = train_data_norm
        self.X_val, self.y_val, self.val_mask, self.val_ids, self.val_elms = val_data_norm
        self.X_test, self.y_test, self.test_mask, self.test_ids, self.test_elms = test_data_norm
        self.mu_T, self.var_T,  self.mu_D, self.var_D = prof_norms.values()
        self.mu_MP, self.var_MP = mp_norms.values()

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
