from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from .base import BaseVAE
from .utils import get_conv_output_size, get_trans_output_size, get_final_output
Tensor = TypeVar('torch.tensor')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size, length):
        super(UnFlatten, self).__init__()
        self.size = size
        self.length = length

    def forward(self, input):
        out = input.view(input.size(0), self.size, self.length)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int, max_pool_kernel_size: int, stride: int = 1, num_conv_blocks: int = 1,  padding='same'):
        """
        A convoution block, with form:
        input [BS, in_channel, len] -> [CONV(*N) -> Max Pool] -> output [BS, out_channel, length / max_pool_kernel_size]


        The output of a single convolution layer will have the output length of size given by:
        [((in_length) + 2*padding - (kernel_size - 1)) / stride ]+ 1

        The max pool layer effectlivey halves the size of the input length

        Parameters
        ----------

        in_channels: int
            input channel size
        out_channels: int
            output channel size
        conv_kernel_size: int
            size of the Conv kernel
        num_conv_blocks: int
            number of convolution blocks
        stride: int
            stride of  conv. blocks
        padding: int
            padding of conv. blocks

        Returns
        -------
            block: nn.ModuleList
        """
        super(ConvBlock, self).__init__()

        self.block = nn.ModuleList()
        for i in range(num_conv_blocks):
            self.block.append(nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=stride, padding=padding))
            self.block.append(nn.ReLU())
            in_channels = out_channels
        self.block.append(nn.MaxPool1d(max_pool_kernel_size))

    def forward(self, x):
        for blc in self.block:
            x = blc(x)
        return x

class TranspConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int, num_conv_blocks: int = 1, stride: int = 2, padding: int=1):
        """
        A transpose convoution block, with form:
        input [BS, in_channel, len] -> [TransConv(*N) -> Max Pool] -> output [BS, out_channel, length / max_pool_kernel_size]

        The output of a single tranpose conv layer has the folowing size:

        [(input_size - 1)*stride - 2*padding + (kernel_size -1 ) + 1]

        Parameters
        ----------

        in_channels: int
            input channel size
        out_channels: int
            output channel size
        conv_kernel_size: int
            size of the TransConv kernel
        num_conv_blocks: int
            number of tranposed convolution blocks
        stride: int
            stride of trans. conv. blocks
        padding: int
            padding of trans. conv. blocks

        Returns
        -------
            block: nn.ModuleList
        """
        super(TranspConvBlock, self).__init__()

        self.block = nn.ModuleList()
        for i in range(num_conv_blocks):
            self.block.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=stride, padding=padding))
            self.block.append(nn.ReLU())
            in_channels = out_channels

    def forward(self, x):
        for blc in self.block:
            x = blc(x)
        return x



class VisualizeBetaVAE(BaseVAE):
    """ A BetaGammaVAE for 1D Signal Processing (Work in Progress)
    You have a Encoder -> Latent Space -> Decoder

    Parameters
    ----------
    in_ch: int
        The channel dimensions of input. For an RGB image this would be 3. For us this is 1 (could be 2 in the future when we include Te profiles)
    latent_dim: int
        The size of the latent space, should be between 3-20
    hidden_dims: List
        The channel size of hidden convolution layers,
        e.g., with [2, 4], there are two convolution and transposed convolution blocks
        The first conv. block goes from in_ch -> 2, then next from 2-> 4 before entering latent space.
        The list is reversed for the decoder (transposed)
    out_length: int
        The output length of the profile. This should always be 63, until I figure out a way to upsample.
    beta: int or float
        Weighting term on KL-Div, should be greater than 1.
        This constrains latent variables to be independent, therefore with high beta and low latent dim, it is possible the model will not converge
    gamma: int or float
        How many iterations before beta term on KL-Div is applied fully.
        Basically the weighting term on KL-Div is beta*num_batches_trained/gamma, so after gamma number of batches, beta is fully applied
    loss_type: str
        Two options, 'G' or 'B', which refer to Gamma and Beta respectively.
        'B' ignores gamma, and just directly applies Beta to KL-Div term of loss at epoch 0
        'G' uses gamma
    num_conv_blocks: int
        Number of convolution layers to use in each convolution block.
        This will not change the output size at the moment, as the padding is set to 'same'
    conv_kernel_size: int
        Size of convolution kernel to use in encoder convolution layers. This should be 1, since we are using padding='same'!
    conv_stride: int
        Stride of convolution layer in encoders
    conv_padding: str or int
        Don't touch this unless I know what I am doing, and at the moment I do not.
    num_trans_conv_blocks: int
        Number of transposed convolution layers to use in each trans block in decoder.
        This does change the output length, but we have it covered with utility functions!
    trans_kernel_size: int
        see conv_kernel_size, but for trans.
    trans_stride
        Tranpsosed conv. stride
    trans_padding
        Tranposed conv. padding
    Returns
    -------

    None
    """
    num_iter = 0
    def __init__(self, in_ch: int, latent_dim: int, hidden_dims: List=None, out_length: int = 63,
                beta: float = 4.0, gamma: float = 3000000.,  loss_type: str = 'B',
                num_conv_blocks: int = 2, conv_kernel_size: int = 3, conv_stride: int = 1, conv_padding = 'same',
                num_trans_conv_blocks: int = 2, trans_kernel_size: int = 3, trans_stride: int = 2, trans_padding: int = 1,
                **kwargs) -> None:
        super(VisualizeBetaVAE, self).__init__()

        self.latent_dim = latent_dim

        # Loss params
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma


        # Convolution Layer Params
        self.num_conv_blocks = num_conv_blocks
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.max_pool_stride = 2

        # Transpose convolution layer params
        self.num_trans_conv_blocks = num_trans_conv_blocks
        self.trans_kernel_size = trans_kernel_size
        self.trans_stride = trans_stride
        self.trans_padding = trans_padding

        # Deep Params
        self.hidden_dims = hidden_dims

        in_length = out_length
        start_k = in_ch

        # Get the length of the output after (convolution + max pool blocks)
        end_conv_size = get_conv_output_size(in_length, len(self.hidden_dims))

        # Get the length of the output after decoder (transposed convolution blocks)
        final_size = get_final_output(end_conv_size, len(self.hidden_dims), self.num_trans_conv_blocks, self.trans_stride, self.trans_padding, self.trans_kernel_size)


        # Build Encoder
        modules = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = [4, 8]

        for h_dim in hidden_dims:
            modules.append(ConvBlock(in_ch, h_dim, self.conv_kernel_size, self.max_pool_stride, self.conv_stride, self.num_conv_blocks, self.conv_padding))
            in_ch = h_dim
        modules.append(Flatten())

        self.encoder = nn.Sequential(*modules)


        # Latent Space
        self.fc_mu = nn.Linear(hidden_dims[-1] * end_conv_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * end_conv_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]* end_conv_size)

        # Decoder
        hidden_dims.reverse()

        modules = nn.ModuleList()

        modules.append(UnFlatten(size=hidden_dims[0], length= end_conv_size))
        hidden_dims.append(start_k)

        for i in range(len(hidden_dims) - 1):
            modules.append(TranspConvBlock(hidden_dims[i], hidden_dims[i+1], trans_kernel_size, num_trans_conv_blocks, trans_stride, trans_padding))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(final_size, out_length)

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
        # print('ready for decoder', result.shape)
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
            loss =  kwargs['M_N']*recons_loss  + self.beta  *  kwargs['M_N']* kld_loss
        elif self.loss_type == 'G':
            loss = recons_loss * kwargs['M_N'] +  kwargs['M_N']*self.beta * self.num_iter / (self.gamma) * kld_loss
        else:
            raise ValueError('Undefined Loss type, choose between B or G (beta or gamma)')
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        profilie space map.

        Parameters
        ----------

        num_samples: int
            Number of samples
        current_device: something, I am not sure yet
            Device to run the model


        Returns
        -------

        samples: torch.Tensor
            The corresponding profile space mapping of num_samples
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
