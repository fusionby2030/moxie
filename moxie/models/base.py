"""
Will contain the base class for all DIVA models. 
"""

from .utils_ import *
from abc import abstractmethod
import torch.nn as nn

class Base(nn.Module):
    """
    A base VAE class. All VAEs (should) implement the following methods. 
    """

    def __init__(self) -> None:
        super(Base, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError 
    
    def decode(self, input: Tensor) -> Any: 
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
