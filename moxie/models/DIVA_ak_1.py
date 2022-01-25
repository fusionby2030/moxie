from .utils_ import * 
from .base import Base 
from .AK_torch_modules import PRIORreg, ENCODER, DECODER, AUXreg

import torch 
import torch.nn as nn 
from torch.nn import functional as F 


class DIVAMODEL(Base): 
    """
    A Diva model.

    Parameters
    ----------

    in_ch: int
        The channel dimensions of input. For an RGB image this would be 3. For us this is 1 (could be 2 in the future when we include Te profiles)
    out_length: int
        The output length of the profile. This should always be 63, until I figure out a way to upsample.
    alpha_prof: float
        Weighting term for reconstruction loss of profile. Higher (>>1) means profile is higher importance. 
    alpha_mach: float
        Weighting term for reconstruction of machine parameters. 
    beta_stoch: float
        Weighting term for KL-Div loss of stochastic latent space. 
    beta_mach: float
        Weighting term for KL-Div loss of machine latent space (larger latent space, which we assume encodes the machine params)
        To ensure that Mach latent space encodes more information that stoch latent space, 
        beta_mach = beta_stoch / beta_mach, i.e., we divide beta_stoch by what beta_mach was supplied
    mach_latent_dim: int
        The size of the machine latent space
    stoch_latent_dim: int
        The size of the stoch latent space. Should always be smaller than mach_latent_dim
    loss_type: str
        Three options: 'supervised', 'unsupervised', 'semi-supervised'
        'unsupervised': Prof recon Loss + KLD_stoch(vs N(0, 1)) + KLD_mach(vs N(0, 1))
        'supervised':  Prof recon Loss + MP recon loss +  KLD_stoch(vs N(0, 1)) + KLD_mach(vs N(conditional_mu, conditional_var)) 
        'semi-supervised': Alternate between supervsied and unsupervised every epoch 
    Returns
    -------

    None
    """
    num_iterations = 0 # Trickery for the semi-supervsied loss, 
    def __init__(self, in_ch: int=2, out_length: int = 24, 
                        alpha_prof: float = 1., alpha_mach: float = 1., 
                        beta_stoch: float = 0.01, beta_mach: float = 100., 
                        mach_latent_dim: int = 15, stoch_latent_dim: int = 5, 
                        loss_type: str = 'semi-supervised', **kwargs) -> None: 

        super(DIVAMODEL, self).__init__()
        
        # Architecture params 
        num_machine_params = 13 # maybe make its own variable? Or try from KWARGS
        self.stoch_latent_dim = stoch_latent_dim 
        self.stoch_latent_dim = stoch_latent_dim
        self.mach_latent_dim = mach_latent_dim
        self.encoder_end_dense_size = 128 # Future versions this would be a variable to test ablations to size of output from encoder. 
        self.hidden_dims = [2, 4] # Future versions would make this a variable to test ablations to amount of conv filtering/channel kerneling, blah blah 

        end_conv_size = get_conv_output_size(24, len(self.hidden_dims)) # TODO: Not implemented yet

        # Loss hyperparams
        self.alpha_prof = alpha_prof
        self.alpha_mach = alpha_mach
        self.beta_stoch = beta_stoch
        self.beta_mach = self.beta_stoch / beta_mach


        self.loss_type = loss_type
        
         # Encoders

        self.encoder_n = ENCODER()
        self.encoder_t = ENCODER()

        self.encoder_end = nn.Linear(2*(self.hidden_dims[-1] * end_conv_size), self.encoder_end_dense_size)


        # Prior Regressor

        self.prior_reg = PRIORreg(mach_latent_dim)

        # Latent Space

        self.fc_mu_stoch = nn.Linear(self.encoder_end_dense_size, self.stoch_latent_dim)
        self.fc_var_stoch = nn.Linear(self.encoder_end_dense_size, self.stoch_latent_dim)

        self.fc_mu_mach = nn.Linear(self.encoder_end_dense_size, self.mach_latent_dim)
        self.fc_var_mach = nn.Linear(self.encoder_end_dense_size, self.mach_latent_dim)


        # Decoder

        self.decoder_input = nn.Linear(self.stoch_latent_dim + self.mach_latent_dim, self.hidden_dims[-1]*end_conv_size)
        self.decoder_n = DECODER(end_conv_size=end_conv_size)
        self.decoder_t = DECODER(end_conv_size=end_conv_size)
        final_size = self.decoder_n.final_size
        self.final_layer_n = nn.Linear(final_size, out_length)
        self.final_layer_t = nn.Linear(final_size, out_length)

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
        # print(profile[0, :])
        # print(profile[:, :, 0])
        # print(profile[:, : ,0].shape)
        # print(profile[0, :])
        encoded_prof_n = self.encoder_n(profile[:, 0:1, :])
        encoded_prof_t = self.encoder_t(profile[:, 1:, :])
        concat = torch.cat((encoded_prof_n, encoded_prof_t), 1)
        encoded_prof = self.encoder_end(concat)

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
         result_n = self.decoder_n(result)
         result_t = self.decoder_t(result)
         out_prof_n = self.final_layer_n(result_n)
         out_prof_t = self.final_layer_t(result_t)
         out_prof = torch.cat((out_prof_n, out_prof_t), 1)

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



