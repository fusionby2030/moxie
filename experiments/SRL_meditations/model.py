import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List 
import numpy as np 
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
class ENCODER(nn.Module): 
    def __init__(self, filter_sizes: List[int], in_length: int = 75): 
        super().__init__()
        in_ch = 2
        hidden_channels = filter_sizes
        self.kernel_size = 3
        self.padding = 0
        self.stride = 1
        self.pool_padding = 0
        self.pool_dilation = 1
        self.pool_kernel_size = 2
        self.pool_stride = self.pool_kernel_size
        
        self.block = nn.ModuleList()
        self.end_conv_size = in_length # [(W - K + 2P) / S] + 1
        for dim in hidden_channels: 
            self.block.append(nn.Conv1d(in_ch, dim, kernel_size=self.kernel_size, padding=self.padding))
            self.end_conv_size = ((self.end_conv_size - self.kernel_size + 2*self.padding) / self.stride) + 1
            self.block.append(nn.ReLU())
            self.block.append(nn.MaxPool1d(self.pool_kernel_size, padding=self.pool_padding, dilation=self.pool_dilation, ))
            self.end_conv_size = ((self.end_conv_size + 2*self.pool_padding - self.pool_dilation*(self.pool_kernel_size -1)-1) / self.pool_stride) + 1
            
            in_ch = dim
        self.block.append(Flatten())

        self.end_conv_size = int(self.end_conv_size)
    def forward(self, x): 
        # print('Encoder in shape', x.shape)
        for lay in self.block: 
            x = lay(x)
            # print(x.shape)
        # print('encoder out shape', x.shape)
        return x

class DECODER(nn.Module): 
    def __init__(self, filter_sizes: List[int], end_conv_size: int, ): 
        super().__init__()
        in_ch = 2
        self.hidden_channels = filter_sizes
        self.end_conv_size = end_conv_size
        self.num_trans_conv_blocks = 1
        self.trans_stride = 1
        self.trans_padding = 0
        self.trans_kernel_size = 2

        self.block = nn.ModuleList()
        if self.hidden_channels[-1] != in_ch: 
            self.hidden_channels.append(in_ch)
        self.block.append(UnFlatten(size=self.hidden_channels[0], length=self.end_conv_size))
        # Needs trasnpose kernel instead 
        for i in range(len(self.hidden_channels) - 1):
            self.block.append(nn.ConvTranspose1d(self.hidden_channels[i], self.hidden_channels[i+1], kernel_size=self.trans_kernel_size))
            self.block.append(nn.ReLU())
        self.final_size = get_final_output(end_conv_size, len(self.hidden_channels) -1, self.num_trans_conv_blocks, self.trans_stride, self.trans_padding, self.trans_kernel_size)
    def forward(self, x): 
        # print('Decoder in shape', x.shape)
        for lay in self.block: 
            x = lay(x)
        # print('Decoder out shape', x.shape)
        return x

class TRANSFERLAYER(nn.Module): 
    def __init__(self, latent_dim: int, hidden_dims: List[int] = [10, 20, 10]): 
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.block = nn.ModuleList()
        last_dim = self.latent_dim
        for dim in self.hidden_dims: 
            self.block.append(nn.Linear(last_dim, dim))
            self.block.append(nn.ReLU())
            last_dim = dim 
        # self.block.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
    def forward(self, z):
        # print('Enter transfer dim', z.shape)
        for lay in self.block: 
            z = lay(z) 
        # print('Exit transfer dim', z.shape)
        return z

class VAE_LLD(nn.Module): 
    """
    Implementation of E2C linear latent dimensions
    """
    def __init__(self, input_dim: int, latent_dim: int, transfer_hidden_dims: List[int], conv_filter_sizes: List[int], out_length: int = 75, act_dim: int=None, act_fn=None): 
        super(VAE_LLD, self).__init__()

        
        self.input_dim = input_dim 
        self.latent_dim = latent_dim 
        self.transfer_hidden_dims = transfer_hidden_dims
        self.act_dim = 2
        self.conv_filter_sizes = conv_filter_sizes
        self.trans_conv_filter_sizes = conv_filter_sizes[::-1]

        # Building the network 
        self.encoder = ENCODER(filter_sizes=self.conv_filter_sizes) 
        self.decoder = DECODER(filter_sizes=self.trans_conv_filter_sizes, end_conv_size=self.encoder.end_conv_size) 
        
        self.z_mu = nn.Linear(self.encoder.end_conv_size*self.conv_filter_sizes[-1], self.latent_dim)
        self.z_var = nn.Linear(self.encoder.end_conv_size*self.conv_filter_sizes[-1], self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, self.trans_conv_filter_sizes[0]*self.encoder.end_conv_size)
        final_size = self.decoder.final_size
        self.output_layer = nn.Linear(final_size, out_length)

        # Linear transformation layers 

        self.tranfer_layer = TRANSFERLAYER(self.latent_dim, hidden_dims=self.transfer_hidden_dims)
        self.v_t = nn.Linear(self.transfer_hidden_dims[-1], self.latent_dim)
        self.r_t = nn.Linear(self.transfer_hidden_dims[-1], self.latent_dim)
        self.o_t = nn.Linear(self.transfer_hidden_dims[-1], self.latent_dim)

        # self.B_t = nn.Linear(self.transfer_hidden_dims[-1], self.latent_dim*self.act_dim)
        

    def forward(self, x_t, x_t_1): 
        # x_t -> z_t 
        z_t, mu_t, var_t = self.x2z(x_t)

        # x_t_1 -> z_t_1
        z_t_1, mu_t_1, var_t_1 = self.x2z(x_t_1)
        
        # z_t -> x_hat_t 
        x_hat_t = self.z2x(z_t)

        # z_t_1 -> x_hat_t_1
        x_hat_t_1 = self.z2x(z_t_1)

        # z_t -> z_hat_t_1 
        z_hat_t_1, A_t, o_t = self.zt2zt_1(z_t)
        # z_hat_t_1 -> x_hathat_t_1
        x_hat_hat_t_1 = self.z2x(z_hat_t_1)
        return x_hat_t, x_hat_t_1, x_hat_hat_t_1,  (mu_t, var_t), (mu_t_1, var_t_1), (A_t, o_t)

    def reparameterize(self, mu, var): 
        z = mu*var
        return z
    def x2z(self, x): 
        enc = self.encoder(x)
        mu, var = self.z_mu(enc), self.z_var(enc)
        z = self.reparameterize(mu, var)
        return z, mu, var

    def z2x(self, z): 
        # print('Latent out', z.shape)
        z = self.decoder_input(z)
        dec = self.decoder(z)      
        out_prof = self.output_layer(dec)  
        return out_prof

    def zt2zt_1(self, z): 
        zt_1 = self.tranfer_layer(z)
        v_t = self.v_t(zt_1)
        r_t = self.r_t(zt_1)
        v_t, r_t_T = v_t.unsqueeze(-1), r_t.unsqueeze(1),
        o_t = self.o_t(zt_1)
        A_t = torch.eye(self.latent_dim) + torch.matmul(v_t, r_t_T)
        z_1 = (torch.matmul(A_t, z.unsqueeze(2))).squeeze(2) + o_t

        # print(A_t.shape, zt_1.shape,z.shape )
        # B_t = self.B_t(zt_1)
        # B_t_T = torch.reshape(B_t, (-1,self.act_dim,  self.latent_dim))
        # print(A_t.shape, B_t_T.shape, zt_1.shape, zt_1.unsqueeze(2).shape )
        # torch.matmul(A_t, zt_1.unsqueeze(2))
        # torch.matmul(B_t_T, zt_1.unsqueeze(2))
        # print((torch.matmul(A_t, zt_1.unsqueeze(2))).shape, (torch.matmul(B_t_T, zt_1.unsqueeze(2))).shape)
        # torch.matmul(A_t, zt_1.unsqueeze(2)) + torch.matmul(B_t_T, zt_1.unsqueeze(2))
        
        # z_1 = torch.matmul(A_t, B_t_T) + o_t
        return z_1, A_t, o_t


    
        