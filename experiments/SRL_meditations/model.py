import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from abc import abstractmethod
from typing import List, Any, TypeVar

Tensor = torch.Tensor# TypeVar('torch.tensor')

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
def get_conv_output_size(initial_input_size, number_blocks, max_pool=True):
    """ The conv blocks we use keep the same size but use max pooling, so the output of all convolution blocks will be of length input_size / 2"""
    if max_pool==False:
        return initial_input_size
    out_size = initial_input_size
    for i in range(number_blocks):
        out_size = int(out_size / 2)
    return out_size

def get_trans_output_size(input_size, stride, padding, kernel_size):
    """ A function to get the output length of a vector of length input_size after a tranposed convolution layer"""
    return (input_size -1)*stride - 2*padding + (kernel_size - 1) +1

def get_final_output(initial_input_size, number_blocks, number_trans_per_block, stride, padding, kernel_size):
    """A function to get the final output size after tranposed convolution blocks"""
    out_size = initial_input_size
    for i in range(number_blocks):
        for k in range(number_trans_per_block):
            out_size = get_trans_output_size(out_size, stride, padding, kernel_size)
    return out_size 
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
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int, max_pool_kernel_size: int, stride: int = 1, num_conv_blocks: int = 1,  padding='same', max_pool=True):
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


class DECODER(nn.Module):
    """ The DECODER """
    def __init__(self, end_ch = 2, hidden_dims: List=[4, 2, 2], num_trans_conv_blocks: int = 2, trans_kernel_size: int = 3, trans_stride: int = 2, trans_padding: int = 1, end_conv_size=1):
        super(DECODER, self).__init__()
        self.num_trans_conv_blocks = num_trans_conv_blocks
        self.trans_kernel_size = trans_kernel_size
        self.trans_stride = trans_stride
        self.trans_padding = trans_padding
        self.hidden_dims = hidden_dims

        self.block = nn.ModuleList()

        self.block.append(UnFlatten(size=hidden_dims[0], length=end_conv_size))
        if self.hidden_dims[-1] != end_ch:
            self.hidden_dims.append(end_ch)
        for i in range(len(hidden_dims) - 1):
            self.block.append(TranspConvBlock(hidden_dims[i], hidden_dims[i+1], trans_kernel_size, num_trans_conv_blocks, trans_stride, trans_padding))
            # self.block.append(nn.ReLU())
        self.final_size = get_final_output(end_conv_size, len(self.hidden_dims) -1, self.num_trans_conv_blocks, self.trans_stride, self.trans_padding, self.trans_kernel_size)
    def forward(self, x):
        for lay in self.block:
            x = lay(x)
        return x


class ENCODER(nn.Module):
    """ The Encoder """
    def __init__(self, in_ch: int = 1, hidden_dims: List=[2, 4], num_conv_blocks: int = 2, conv_kernel_size: int = 3, conv_stride: int = 1, conv_padding = 'same', max_pool=True):
        super(ENCODER, self).__init__()
        # Convolution Layer Params
        self.num_conv_blocks = num_conv_blocks
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.max_pool_stride = 2
        self.max_pool = max_pool

        self.block = nn.ModuleList()
        for h_dim in hidden_dims:
            self.block.append(ConvBlock(in_ch, h_dim, self.conv_kernel_size, self.max_pool_stride, self.conv_stride, self.num_conv_blocks, self.conv_padding, max_pool=self.max_pool))
            # self.block.append(nn.ReLU())
            in_ch = h_dim
        self.block.append(Flatten())

    def forward(self, x):
        for lay in self.block:
            x = lay(x)
        return x
class FORWARDMODULE(nn.Module): 
    def __init__(self, latent_dim=5): 
        super(FORWARDMODULE, self).__init__()

        self.block = nn.ModuleList() 

        self.block.append(nn.Linear(latent_dim, 10))
        self.block.append(nn.ReLU())
        self.block.append(nn.Linear(10, latent_dim))
    def forward(self, z): 
        for lay in self.block: 
            z = lay(z)
        return z
class ForwardVAE(Base): 
    def __init__(self): 
        super(ForwardVAE, self).__init__()

        latent_dim = 3
        in_ch = 2
        hidden_dims = [2, 4] # [4, 6, 8]
        out_length = 50
        encoder_end_dense_size = 36

        end_conv_size = get_conv_output_size(out_length, len(hidden_dims))
        self.encoder = ENCODER(in_ch=in_ch, hidden_dims=hidden_dims, num_conv_blocks=2, conv_kernel_size=3)
        self.encoder_end = nn.Linear((hidden_dims[-1] * end_conv_size), encoder_end_dense_size)

        self.fc_mu = nn.Linear(encoder_end_dense_size, latent_dim)
        self.fc_var = nn.Linear(encoder_end_dense_size, latent_dim)

        self.forward_module = FORWARDMODULE(latent_dim=latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*end_conv_size)
        reversed_dims = hidden_dims[::-1]
        reversed_dims.append(in_ch)
        self.decoder = DECODER(end_ch=in_ch, hidden_dims = reversed_dims, end_conv_size=end_conv_size)
        final_size = self.decoder.final_size
        self.final_layer = nn.Linear(final_size, out_length)
        
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparam trick, sampling from N(mu, var) """

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu    
        
    
    def forward(self, in_profs: Tensor) -> Tensor: 
        encoded_prof = self.encoder_end(self.encoder(in_profs))
        

        mu, var = self.fc_mu(encoded_prof), self.fc_var(encoded_prof)
        z_t0 = self.reparameterize(mu, var)
        # -> pass to the forward model 
        z_t1 = self.forward_module(z_t0)

        out_profs_t0 = self.decoder_input(z_t0)
        out_profs_t0 = self.decoder(out_profs_t0)
        out_profs_t0 = self.final_layer(out_profs_t0)

        out_profs_t1 = self.decoder_input(z_t1)
        out_profs_t1 = self.decoder(out_profs_t1)
        out_profs_t1 = self.final_layer(out_profs_t1)
        return out_profs_t0, out_profs_t1, mu, var

    def loss_function(in_profs, out_profs): 
        pass 
