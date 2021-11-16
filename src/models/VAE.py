from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from .base import BaseVAE
Tensor = TypeVar('torch.tensor')


# In DIM = 1 for us
class VanillaVAE(BaseVAE):
    """ Will be a vanilla feed forward VAE """
    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: List=None, **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.kld_weight = 0.0001
        self.latent_dim = latent_dim

        out_dim = in_dim

        modules = []

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        # Encoder

        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ))
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()
        # Encoder
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.Sigmoid(),
                nn.Dropout(0.001)
            ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], out_dim)


    def encode(self, input: Tensor) -> List[Tensor]:
        """Encodes the input and returns latent codes"""
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """ Maps latent codes into profile space """

        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparam trick, sampling from N(mu, var) """

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = 0.001 # kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss * kwargs['M_N'] + self.kld_weight * kwargs['M_N'] * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        profilie space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input profile x, returns the reconstructed profile
        """
        return self.forward(x)[0]
