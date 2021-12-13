from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from .base import BaseVAE
from .utils import get_conv_output_size, get_trans_output_size, get_final_output

Tensor = TypeVar('torch.tensor')


class DECODER(nn.Module):
    """ The DECODER """
    def __init__(self):
        super(DECODER, self).__init__()
        pass

    def forward(self, x):
        pass

class ENCODER(nn.Module):
    """ The Encoder """
    def __init__(self):
        super(ENCODER, self).__init__()
        pass

    def forward(self, x):
        pass


class PRIORreg(nn.Module):
    """
    A regressor to define the prior for Z_mach
    """
    def __init__(self):
        super(PRIORreg, self).__init__()

        self.out = nn.Linear(something, 2)
        pass

    def forward(self, x):
        pass


class AUXreg(nn.Module):
    """
    An Auxiliarly Regressor

    This will take us from Z_mach to machine params
    """
    def __init__(self, mp_size, z_mach_dim):
        super(AUXreg, self).__init__()

        self.fc1 = nn.Linear(z_mach_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, mp_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)


class DIVA(nn.Module):
    """
    A base VAE class. All VAEs will have the following methods implemented.
    """

    def __init__(self) -> None:
        super(DIVA, self).__init__()

        num_machine_params = 14
        self.stoch_latent_dim = 5
        self.mach_latent_dim = 14
        # Encoder

        self.encoder = ENCODER()

        # Prior Regressor

        self.prior_reg = PRIORreg()

        # Latent Space

        self.fc_mu_stoch = nn.Linear(hidden_dims[-1] * end_conv_size, self.stoch_latent_dim)
        self.fc_var_stoch = nn.Linear(hidden_dims[-1] * end_conv_size, self.stoch_latent_dim)

        self.fc_mu_mach = nn.Linear(hidden_dims[-1] * end_conv_size, self.mach_latent_dim)
        self.fc_var_mach = nn.Linear(hidden_dims[-1] * end_conv_size, self.mach_latent_dim)

        # Decoder

        self.decoder = DECODER()

        # Auxiliarly Regressor

        self.aux_reg = AUXreg(self.mach_latent_dim, num_machine_params)


    def q_zy(self, profile: Tensor) -> List[Tensor]:
        """
        Encode profiles into latent space

        Parameters
        ----------

        profile: Tensor
            A profile with shape [BS, in_channels, in_length]
            For just the density profiles, psi22 dataset, this is [BS, 1, 63]

        Returns
        -------

        latent_space: List[Tensor]
            The latent space is parameterized by mean and logvar of the stoch and mach subspaces
            i.e., [mu_stoch, var_stoch, mu_mach, var_mach]
        """

        encoded_prof = self.encoder(profile)

        mu_stoch = self.fc_mu_stoch(encoded_prof)
        var_stoch = self.fc_var_stoch(encoded_prof)
        mu_mach = self.fc_mu_mach(encoded_prof)
        var_mach = self.fc_var_mach(encoded_prof)

        return [mu_stoch, var_stoch, mu_mach, var_mach]

    def p_zmachx(self, machine_params: Tensor) -> List[Tensor]:
        """
        Provide the conditional priors for Z_mach

        Parameters
        ----------

        machine_params: Tensor
            The machine params which have shape [BS, #machine_params]
            For just the density profiles, psi22 dataset, this is [BS, 14]

        Returns
        -------

        conditional_prior: List[Tensor]
            The supposed conditional prior to be exerted onto the Z_mach,
            i.e., [c_mu_mach, c_var_mach] with shapes???
        """

        # conditional_priors

        return [c_mu_mach, c_var_mach]

    def p_yhatz(self, latent_spaces: Tensor) -> Tensor:
         """
         Decode reparametrized latent space into profiles

         Parameters
         ----------

         latent_spaces: Tensor
             The reparameterization of Z_mach and Z_stoch


         Returns
         -------

         hat_profile: Tensor
             A generated profile with shape [BS, in_channels, in_length]
             For just the density profiles, psi22 dataset, this is [BS, 1, 63]
         """

         return hat_profile

    def q_hatxzmach(self, z_mach: Tensor):
        """
        Predict the machine parameters from Z_mach

        Parameters
        ----------

        z_mach: Tensor
            The reparameterized latent space

        Returns
        -------

        hat_machine_params: Tensor
            A guess of the machine params
        """

        return hat_machine_params

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
