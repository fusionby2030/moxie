from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from .base import BaseVAE
Tensor = TypeVar('torch.tensor')


class Flatten(nn.Module):
    def forward(self, input):

        # print('Flattened Shape', input.view(input.size(0), -1).shape)
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size, length):
        super(UnFlatten, self).__init__()
        self.size = size
        self.length = length

    def forward(self, input):

        out = input.view(input.size(0), self.size, self.length)
        # print('UnFlattened Shape', out.shape)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int, max_pool_kernel_size: int, num_conv_blocks: int = 1):
        """
        A convoution block, with form:
        input [BS, in_channel, len] -> [CONV(*N) -> Max Pool] -> output [BS, out_channel, length / max_pool_kernel_size]

        :params:
            :in_channels: input channel size
            :out_channels: output channel size
            :conv_kernel_size: size of the convolution kernel
            :max_pool_kernel_size: (int) size of kernel in max pooling, which has the end effect of dividing the length of the sequence by its value, i.e., 2 -> halves sequence
            :num_conv_blocks: (int) number of convolution blocks
        :returns:
            :block: nn.ModuleList
        """
        super(ConvBlock, self).__init__()

        self.block = nn.ModuleList()
        for i in range(num_conv_blocks):
            self.block.append(nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=1, padding='same'))
            self.block.append(nn.ReLU())
            in_channels = out_channels
        self.block.append(nn.MaxPool1d(max_pool_kernel_size))

    def forward(self, x):
        for blc in self.block:
        #     print('Encoding', blc)
            x = blc(x)
        # print('Block Done with shape: ', x.shape)
        return x

class TranspConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int, max_pool_kernel_size: int, num_conv_blocks: int = 1):
        """
        A transpose convoution block, with form:
        input [BS, in_channel, len] -> [TransConv(*N) -> Max Pool] -> output [BS, out_channel, length / max_pool_kernel_size]

        :params:
            :in_channels: input channel size
            :out_channels: output channel size
            :conv_kernel_size: size of the TransConv kernel
            :max_pool_kernel_size: (int) size of kernel in max pooling, which has the end effect of dividing the length of the sequence by its value, i.e., 2 -> halves sequence
            :num_conv_blocks: (int) number of tranposed convolution blocks
        :returns:
            :block: nn.ModuleList
        """
        super(TranspConvBlock, self).__init__()

        self.block = nn.ModuleList()
        for i in range(num_conv_blocks):
            self.block.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=2, padding=1))
            self.block.append(nn.ReLU())
            in_channels = out_channels
        # self.block.append(nn.MaxUnpool1d(max_pool_kernel_size))

    def forward(self, x):
        for blc in self.block:
        #     print('Decoding', blc)
            x = blc(x)
        # print('Block Done with shape: ', x.shape)
        return x


class VisualizeBetaVAE(BaseVAE):
    num_iter = 0
    def __init__(self, in_ch: int, latent_dim: int, hidden_dims: List=None, out_dim: int = 63, beta: float = 4.0, **kwargs) -> None:
        super(VisualizeBetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        num_conv_blocks = 2
        num_trans_conv_blocks = 2
        trans_stride = 2
        in_dim = out_dim
        start_k = in_ch
        self.loss_type = 'B'
        self.gamma = 1

        # Encoder
        modules = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = [4, 8]

        for h_dim in hidden_dims:
            modules.append(ConvBlock(in_ch, h_dim, 3, 2, num_conv_blocks))
            in_ch = h_dim
        modules.append(Flatten())

        self.encoder = nn.Sequential(*modules)



        # Latent Space
        self.fc_mu = nn.Linear(hidden_dims[-1] * int(in_dim /  (len(hidden_dims) * 2) ), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * int(in_dim /  (len(hidden_dims) * 2) ), latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]* int(in_dim /  (len(hidden_dims) * 2) ))

        # Decoder
        hidden_dims.reverse()


        modules = nn.ModuleList()

        modules.append(UnFlatten(size=hidden_dims[0], length= int(in_dim /  (len(hidden_dims) * 2))))
        hidden_dims.append(start_k)

        for i in range(len(hidden_dims) - 1):
            modules.append(TranspConvBlock(hidden_dims[i], hidden_dims[i+1], 3, 2, num_trans_conv_blocks))


        self.decoder = nn.Sequential(*modules)

        # Decoder
        final_layer_size = int(in_dim /  (2*(len(hidden_dims) -1))) * (len(hidden_dims) -1) * num_trans_conv_blocks*trans_stride
        print(final_layer_size, int(in_dim /  ((len(hidden_dims) -1) * 2)), (len(hidden_dims) -1)*num_trans_conv_blocks*trans_stride)

        self.final_layer = nn.Linear(int(in_dim /  (2*(len(hidden_dims) -1)))**2, out_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """Encodes the input and returns latent codes"""
        result = self.encoder(input)
        # print('Encoding Final Shape: ', result.shape)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # print('Mu Shape', mu.shape)
         #print('Var Shape', log_var.shape)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """ Maps latent codes into profile space """
        result = self.decoder_input(z)
        #print('ready for decoder', result.shape)
        result = self.decoder(result)
        # print('Decoding Shape', result.shape)
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
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = 0.001 # kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)



        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # loss = recons_loss * kwargs['M_N'] + self.beta * kwargs['M_N'] * kld_loss
        if self.loss_type == 'B':
            loss = recons_loss  + self.beta  * kld_loss
        elif self.loss_type == 'G':
            loss = recons_loss + self.beta * self.num_iter / (self.gamma) * kld_loss
        else:
            raise ValueError('Undefined Loss type, choose between B or G (beta or gamma)')
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
