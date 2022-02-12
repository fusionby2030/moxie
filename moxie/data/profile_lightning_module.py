import pytorch_lightning as pl 
from .profile_dataset_torch import DATASET_AK
from .utils_ import * 
import pickle  # data loading issues 
import torch 
from torch.utils.data import DataLoader
class PLDATAMODULE_AK(pl.LightningDataModule): 
    """
    pl.lightning datamodule class, which will help with our dataloading needs :--)
    # TODO: Implement a mask or not mask ask. 
    """
    def __init__(self, data_dir: str = '', num_workers: int = 1, batch_size: int = 512, dataset_choice='padded', **params):
        super().__init__()

        self.batch_size = batch_size
        self.file_loc = data_dir
        self.num_workers = num_workers
        self.mu_T, self.var_T = None, None # We are normalizing shit here you dig. 
        self.mu_D, self.var_D = None, None # Density normailizing constants 
        self.mu_MP, self.var_MP = None, None # Machine params normailizing constant 
        self.dataset_choice = dataset_choice


        # Sometimes we want to pin the memory to a GPU so this can get passed as a kwarg. 
        if 'pin_memory' in params.keys():
            self.pin_memory = params['pin_memory']
        else:
            self.pin_memory = False

    def prepare_data(self): 
        # Grab the dataset
        with open(self.file_loc, 'rb') as file:
            full_dict = pickle.load(file)
            train_X, train_y, train_mask, train_radii = full_dict['train_dict'][self.dataset_choice]['profiles'],  full_dict['train_dict'][self.dataset_choice]['controls'],  full_dict['train_dict'][self.dataset_choice]['masks'], full_dict['train_dict'][self.dataset_choice]['radii']
            val_X, val_y, val_mask = full_dict['val_dict'][self.dataset_choice]['profiles'],  full_dict['val_dict'][self.dataset_choice]['controls'], full_dict['val_dict'][self.dataset_choice]['masks']
            test_X, test_y, test_mask = full_dict['test_dict'][self.dataset_choice]['profiles'],  full_dict['test_dict'][self.dataset_choice]['controls'], full_dict['test_dict'][self.dataset_choice]['masks']
        # Convert to torch tensors, although this won't work for the raw datasets!!
        self.X_train, self.y_train = torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float()
        self.X_val, self.y_val = torch.from_numpy(val_X).float(), torch.from_numpy(val_y).float()
        self.X_test, self.y_test = torch.from_numpy(test_X).float(), torch.from_numpy(test_y).float()
        self.train_mask, self.val_mask, self.test_mask = torch.from_numpy(train_mask) > 0, torch.from_numpy(val_mask) > 0, torch.from_numpy(test_mask) > 0
        self.train_mask, self.val_mask, self.test_mask = self.train_mask.unsqueeze(1),  self.val_mask.unsqueeze(1), self.test_mask.unsqueeze(1)
        self.train_mask, self.val_mask, self.test_mask = torch.repeat_interleave(self.train_mask, 2, 1 ), torch.repeat_interleave(self.val_mask, 2, 1), torch.repeat_interleave(self.test_mask, 2, 1)
        assert torch.isnan(self.y_train).any() == False 
        
        # Normalize the profiles 
        self.X_train[:, 0], self.mu_D, self.var_D = standardize_simple(self.X_train[:, 0])
        self.X_val[:, 0] = standardize_simple(self.X_val[:, 0], mu=self.mu_D, var=self.var_D)
        self.X_test[:, 0] = standardize_simple(self.X_test[:, 0], mu=self.mu_D, var=self.var_D)

        self.X_train[:, 1], self.mu_T, self.var_T = standardize_simple(self.X_train[:, 1])
        self.X_val[:, 1] = standardize_simple(self.X_val[:, 1], mu=self.mu_T, var=self.var_T)
        self.X_test[:, 1] = standardize_simple(self.X_test[:, 1], mu=self.mu_T, var=self.var_T)

        # Normalize the machine parameters 

        self.y_train, self.mu_MP, self.var_MP = standardize_simple(self.y_train)
        self.y_val = standardize_simple(self.y_val, self.mu_MP, self.var_MP)
        self.y_test =  standardize_simple(self.y_test, self.mu_MP, self.var_MP)

        # TODO: Convert everything to float? 
        
    def setup(self, stage=None):
        self.train_set = DATASET_AK(self.X_train, self.y_train, mask = self.train_mask)
        self.val_set = DATASET_AK(self.X_val, self.y_val, mask = self.val_mask)
        self.test_set = DATASET_AK(self.X_test, y = self.y_test, mask = self.test_mask)
    
    def get_temperature_norms(self): 
        return self.mu_T, self.var_T 

    def get_density_norms(self): 
        return self.mu_D, self.var_D 

    def get_machine_norms(self): 
        return self.mu_MP, self.var_MP

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
