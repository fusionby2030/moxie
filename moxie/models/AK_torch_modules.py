import torch 
import torch.nn as nn 
from .utils_ import * 

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
    def __init__(self, hidden_dims: List=[4, 2, 1], num_trans_conv_blocks: int = 2, trans_kernel_size: int = 3, trans_stride: int = 2, trans_padding: int = 1, end_conv_size=1):
        super(DECODER, self).__init__()
        self.num_trans_conv_blocks = num_trans_conv_blocks
        self.trans_kernel_size = trans_kernel_size
        self.trans_stride = trans_stride
        self.trans_padding = trans_padding
        self.hidden_dims = hidden_dims

        self.block = nn.ModuleList()

        self.block.append(UnFlatten(size=hidden_dims[0], length=end_conv_size))

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


class PRIORreg(nn.Module):
    """
    A regressor to define the prior for Z_mach
    """
    def __init__(self, mach_latent_dim=10):
        super(PRIORreg, self).__init__()
        self.block = nn.ModuleList()
        self.block.append(nn.Linear(13, 32))
        self.block.append(nn.ReLU())
        self.block.append(nn.Linear(32, 16))
        self.block.append(nn.ReLU())
        self.out_mu = nn.Linear(16, mach_latent_dim)
        self.out_var = nn.Linear(16, mach_latent_dim)
        # self.block.append(nn.Linear(16, 2))
        # self.out = nn.Linear(100, 2)

    def forward(self, x):
        for lay in self.block:
            x = lay(x)
        return self.out_mu(x), self.out_var(x)


class AUXreg(nn.Module):
    """
    An Auxiliarly Regressor

    This will take us from Z_mach to machine params
    """
    def __init__(self, mp_size, z_mach_dim):
        super(AUXreg, self).__init__()
        self.block = nn.ModuleList()
        self.block.append(nn.Linear(z_mach_dim, 64))
        self.block.append(nn.ReLU())
        self.block.append(nn.Linear(64, 32))
        self.block.append(nn.ReLU())
        self.block.append(nn.Linear(32, 16))
        self.block.append(nn.ReLU())
        self.block.append(nn.Linear(16, mp_size))

    def forward(self, x):
        for lay in self.block:
            x = lay(x)
        return x
