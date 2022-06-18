import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class ENCODER(nn.Module): 
    def __init__(self): 
        super().__init__()
        in_ch = 2
        hidden_channels = [2, 4]
        self.block = nn.ModuleList()
        # self.conv_1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3)
        for dim in hidden_channels: 
            self.block.append(nn.Conv1d(in_ch, dim, kernel_size=2))
            self.block.append(nn.ReLU())
            self.block.append(nn.MaxPool1d())
            in_ch = dim
    def forward(self, x): 
        for lay in self.block: 
            x = lay(x)
        return x

class DECODER(nn.Module): 
    def __init__(self): 
        super().__init__()
        in_ch = 2
        hidden_channels = [2, 4]
        self.block = nn.ModuleList()
        # Needs trasnpose kernel instead 
        for dim in hidden_channels: 
            self.block.append(nn.Conv1d(in_ch, dim, kernel_size=2))
            self.block.append(nn.ReLU())
            self.block.append(nn.MaxPool1d())
            in_ch = dim
    def forward(self, x): 
        for lay in self.block: 
            x = lay(x)
        return x

class VAE_LLD(nn.Modules): 
    """
    Implementation of E2C linear latent dimensions
    """
    def __init__(self, input_dim: int, latent_dim: int, act_dim: int=None, act_fn=None): 
        super(VAE_LLD, self).__init__()

        
        self.input_dim = input_dim 
        self.latent_dim = latent_dim 

        # Building the network 
        self.encoder = ENCODER() 
        self.decoder = DECODER() 

        
        self.z_mu = nn.Linear()
        self.z_var = nn.Linear()

    def forward(self, x_t): 
        # x_t -> z_t 

        # x_t_1 -> z_t_1

        # z_t -> x_hat_t 

        # z_t -> z_hat_t_1 

        # z_hat_t_1 -> x_hat_t_1
        pass 

    def reparameterize(self, mu, var): 
        z = mu*var
        return z
    def x2z(self, x): 
        enc = self.encoder(x)
        mu, var = self.z_mu(enc), self.z_var(enc)
        z = self.reparameterize(mu, var)
        return z

    def z2x(self, z): 
        dec = self.decoder(z)        
        return dec

    def zt2zt_1(self, z): 
        pass 

    
        