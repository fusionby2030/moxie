import torch 
from torch.utils.data import Dataset 

class DATASET_AK(Dataset):
    """
    A simple torch dataset that takes two numpy arrays, X and y, and convers them to torch tensors (if they are not already). 

    TODO: Allow for squeezing or unsqueezing for just density choices? Probably don't care, can call your X as X[:, 0:1, :]
    
    """
    def __init__(self, X, y, norm_dicts = None):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        if norm_dicts is not None:
            self.norm_dicts = norm_dicts

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
