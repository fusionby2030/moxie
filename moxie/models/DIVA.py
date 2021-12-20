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
    def __init__(self):
        super(PRIORreg, self).__init__()
        self.block = nn.ModuleList()
        self.block.append(nn.Linear(13, 32))
        self.block.append(nn.ReLU())
        self.block.append(nn.Linear(32, 16))
        self.block.append(nn.ReLU())
        self.out_mu = nn.Linear(16, 1)
        self.out_var = nn.Linear(16, 1)
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

        self.fc1 = nn.Linear(z_mach_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, mp_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)


class DIVA_v1(BaseVAE):
    """
    DIVA
    """
    num_iterations = 0
    def __init__(self, alpha_prof: float = 1., alpha_mach: float=1., 
                        beta_stoch: float =  0.01, beta_mach: float = 1000., 
                        mach_latent_dim: int = 13, stoch_latent_dim: int = 5,
                        loss_type: str = 'supervised', **kwargs) -> None:
        super(DIVA_v1, self).__init__()

        num_machine_params = 13
        self.stoch_latent_dim = stoch_latent_dim
        self.mach_latent_dim = mach_latent_dim
        self.hidden_dims = [2, 4]
        end_conv_size = get_conv_output_size(63, len(self.hidden_dims))

        # Loss hyperparams
        self.alpha_prof = alpha_prof
        self.alpha_mach = alpha_mach
        self.beta_stoch = beta_stoch
        self.beta_mach = self.beta_stoch / beta_mach


        self.loss_type = loss_type

        # Encoder

        self.encoder = ENCODER()

        # Prior Regressor

        self.prior_reg = PRIORreg()

        # Latent Space

        self.fc_mu_stoch = nn.Linear(self.hidden_dims[-1] * end_conv_size, self.stoch_latent_dim)
        self.fc_var_stoch = nn.Linear(self.hidden_dims[-1] * end_conv_size, self.stoch_latent_dim)

        self.fc_mu_mach = nn.Linear(self.hidden_dims[-1] * end_conv_size, self.mach_latent_dim)
        self.fc_var_mach = nn.Linear(self.hidden_dims[-1] * end_conv_size, self.mach_latent_dim)


        # Decoder

        self.decoder_input = nn.Linear(self.stoch_latent_dim + self.mach_latent_dim, self.hidden_dims[-1]*end_conv_size)
        self.decoder = DECODER(end_conv_size=end_conv_size)
        final_size = self.decoder.final_size
        print(end_conv_size)
        print(final_size)
        self.final_layer = nn.Linear(final_size, 63)

        # Auxiliarly Regressor

        self.aux_reg = AUXreg(z_mach_dim=self.mach_latent_dim, mp_size=num_machine_params)


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
        c_mu_mach, c_var_mach = self.prior_reg(machine_params)

        return [c_mu_mach, c_var_mach]

    def p_yhatz(self, z: Tensor) -> Tensor:
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
         result = self.decoder_input(z)
         result = self.decoder(result)
         out_prof = self.final_layer(result)

         return out_prof

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
        hat_machine_params = self.aux_reg(z_mach)

        return hat_machine_params


    def forward(self, in_profs: Tensor, in_mp: Tensor = None) -> Tensor:
        prior_mu, prior_stoch = self.p_zmachx(in_mp)
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.q_zy(in_profs)
        z_stoch, z_mach = self.reparameterize(mu_stoch, log_var_stoch), self.reparameterize(mu_mach, log_var_mach)
        z = torch.cat((z_stoch, z_mach), 1)
        out_profs = self.p_yhatz(z)
        out_mp = self.q_hatxzmach(z_mach)
        return {'prior_mu': prior_mu, 'prior_stoch': prior_stoch,
                'mu_stoch': mu_stoch, 'log_var_stoch': log_var_stoch, 'mu_mach': mu_mach, 'log_var_mach': log_var_mach,
                'out_profs': out_profs, 'in_profs': in_profs,
                'out_mp': out_mp, 'in_mp': in_mp}

    def loss_function(self, *args, **kwargs) -> Tensor:
        # Assumes that the above dictionary is sent
        prior_mu = kwargs['prior_mu']
        prior_stoch = kwargs['prior_stoch']
        mu_stoch = kwargs['mu_stoch']
        log_var_stoch = kwargs['log_var_stoch']
        mu_mach = kwargs['mu_mach']
        log_var_mach = kwargs['log_var_mach']
        out_profs = kwargs['out_profs']
        in_profs = kwargs['in_profs']
        out_mp  =kwargs['out_mp']
        in_mp = kwargs['in_mp']


        # Reconstruction losses
        recon_prof_loss = F.mse_loss(out_profs, in_profs)
        recon_mp_loss = F.mse_loss(out_mp, in_mp)

        # Z_stoch latent space losses

        stoch_kld_loss = torch.distributions.kl.kl_divergence(
            torch.distributions.normal.Normal(mu_stoch, torch.exp(0.5*log_var_stoch)),
            torch.distributions.normal.Normal(0, 1)
            ).mean(0).sum()

        """if self.num_iterations%2 ==0:
            self.loss_type == 'unsupervised'
        else:
            self.loss_type = 'supervised'"""
        self.num_iterations += 1

        if self.loss_type=='unsupervised':
        # Z_machine latent space losses
            unsup_kld_loss = torch.distributions.kl.kl_divergence(
                 torch.distributions.normal.Normal(mu_mach, torch.exp(0.5*log_var_mach)),
                 torch.distributions.normal.Normal(0, 1)
                 ).mean(0).sum()

            unsupervised_loss = self.alpha_prof * recon_prof_loss + self.beta_stoch * stoch_kld_loss + self.beta_mach * unsup_kld_loss
            return {'loss': unsupervised_loss, 'KLD_stoch': stoch_kld_loss, 'KLD_mach': unsup_kld_loss, 'Reconstruction_Loss': recon_prof_loss, 'Reconstruction_Loss_mp': recon_mp_loss}
        elif self.loss_type == 'supervised':

            sup_kld_loss =torch.distributions.kl.kl_divergence(
                 torch.distributions.normal.Normal(mu_mach, torch.exp(0.5*log_var_mach)),
                 torch.distributions.normal.Normal(prior_mu, torch.exp(0.5*prior_stoch))
                 ).mean(0).sum()

            supervised_loss = self.alpha_prof * recon_prof_loss + self.alpha_mach * recon_mp_loss + self.beta_stoch * stoch_kld_loss + self.beta_mach * sup_kld_loss
            return {'loss': supervised_loss, 'KLD_stoch': stoch_kld_loss, 'KLD_mach': supervised_loss, 'Reconstruction_Loss_mp': recon_mp_loss, 'Reconstruction_Loss': recon_prof_loss}
        elif self.loss_type == 'semi-supervised': 
            if self.num_iterations%2 == 0:
                sup_kld_loss =torch.distributions.kl.kl_divergence(
                 torch.distributions.normal.Normal(mu_mach, torch.exp(0.5*log_var_mach)),
                 torch.distributions.normal.Normal(prior_mu, torch.exp(0.5*prior_stoch))
                 ).mean(0).sum()

                supervised_loss = self.alpha_prof * recon_prof_loss + self.alpha_mach * recon_mp_loss + self.beta_stoch * stoch_kld_loss + self.beta_mach * sup_kld_loss

                return {'loss': supervised_loss, 'KLD_stoch': stoch_kld_loss, 'KLD_mach': supervised_loss, 'Reconstruction_Loss_mp': recon_mp_loss, 'Reconstruction_Loss': recon_prof_loss}
            else: 
                unsup_kld_loss = torch.distributions.kl.kl_divergence(
                 torch.distributions.normal.Normal(mu_mach, torch.exp(0.5*log_var_mach)),
                 torch.distributions.normal.Normal(0, 1)
                 ).mean(0).sum()

                unsupervised_loss = self.alpha_prof * recon_prof_loss + self.beta_stoch * stoch_kld_loss + self.beta_mach * unsup_kld_loss
                return {'loss': unsupervised_loss, 'KLD_stoch': stoch_kld_loss, 'KLD_mach': unsup_kld_loss, 'Reconstruction_Loss': recon_prof_loss, 'Reconstruction_Loss_mp': recon_mp_loss}
        else:
            raise ValueError('Undefined Loss type, choose between unsupervised or supervised')

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparam trick, sampling from N(mu, var) """

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
