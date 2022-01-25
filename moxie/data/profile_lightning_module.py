import pytorch_lightning as pl 
from .profile_dataset_torch import DATASET_AK
from .utils_ import * 
import pickle  # data loading issues 
import torch 
from torch.utils.data import DataLoader
class PLDATAMODULE_AK(pl.LightningDataModule): 
    """
    pl.lightning datamodule calss, which will help with our dataloading needs :--)


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
            train_X, train_y, train_mask, train_radii = full_dict['train'][self.dataset_choice].values()
            val_X, val_y, val_mask, val_radii = full_dict['valid'][self.dataset_choice].values()
            test_X, test_y, test_mask, test_radii = full_dict['test'][self.dataset_choice].values()
        
        # Convert to torch tensors, although this won't work for the raw datasets!!
        # TODO: Implement a load based on the datset_choice 
        self.X_train, self.y_train = torch.from_numpy(train_X), torch.from_numpy(train_y)
        self.X_val, self.y_val = torch.from_numpy(val_X), torch.from_numpy(val_y)
        self.X_test, self.y_test = torch.from_numpy(test_X), torch.from_numpy(test_y)
        
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
        self.train_set = DATASET_AK(self.X_train, self.y_train)
        self.val_set = DATASET_AK(self.X_val, self.y_val)
        self.test_set = DATASET_AK(self.X_test, self.y_test)
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
