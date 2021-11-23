from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from .base import BaseVAE
Tensor = TypeVar('torch.tensor')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):

        out = input.view(input.size(0), self.size, 63)
        return out



class BetaVAE(BaseVAE):

    def __init__(self, in_ch: int, latent_dim: int, hidden_dims: List=None, out_dim: int = 63, beta: float = 4.0, **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        in_dim = out_dim
        start_k = 1
        modules = []

        if hidden_dims is None:
            hidden_dims = [4, 8, 16, 32]
        # Encoder

        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels=start_k, out_channels=h_dim, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            start_k = h_dim

        modules.append(Flatten())

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * in_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * in_dim, latent_dim)

        # Decoder

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*in_dim)

        modules.append(UnFlatten(size=hidden_dims[-1]))

        hidden_dims.reverse()

        hidden_dims.append(in_ch)

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.Sigmoid()
            ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(

                            nn.Linear(out_dim, out_dim),
                            )

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

        # loss = recons_loss * kwargs['M_N'] + self.beta * kwargs['M_N'] * kld_loss
        loss = recons_loss  + self.beta  * kld_loss
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