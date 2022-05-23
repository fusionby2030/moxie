from .utils_ import *
from .base import Base
from .AK_torch_modules import PRIORreg, ENCODER, DECODER, AUXreg


import torch
import torch.nn as nn
from torch.nn import functional as F

class PSI_MODEL(Base):
    """
    The PSI Model.

    Main components:
        Single Encder -> Decoder, i.e., ne, te (and ELM perc) are all passed through the same encoder
    """
    num_iterations = 0 # Trickery for the semi-supervsied loss,
    def __init__(self, in_ch: int=2, out_length: int = 19, elm_style_choice: str = 'none',
                        alpha_prof: float = 1., alpha_mach: float = 1.,
                        beta_stoch: float = 0.01,
                        beta_mach_unsup: float = 0.01, beta_mach_sup: float = 1.,
                        mach_latent_dim: int = 15, stoch_latent_dim: int = 5,
                        encoder_end_dense_size: int = 128,
                        hidden_dims = [2,4],  mp_hdims_cond = [64, 32],mp_hdims_aux = [64, 32],
                        physics: bool = False, gamma_stored_energy: float = 0.0,gamma_bpol: float = 0.0, gamma_beta:float = 0.0,
                        loss_type: str = 'semi-supervised', **kwargs) -> None:

        super(PSI_MODEL, self).__init__()
        if elm_style_choice == 'simple':
            num_machine_params = 14
            in_ch = 3
        else:
            num_machine_params = 13
            in_ch = 2 # maybe make its own variable? Or try from KWARGS
        self.stoch_latent_dim = stoch_latent_dim
        self.mach_latent_dim = mach_latent_dim
        self.encoder_end_dense_size = encoder_end_dense_size
        self.hidden_dims = hidden_dims
        self.mp_hdims_cond = mp_hdims_cond
        self.mp_hdims_aux = mp_hdims_aux



        # Loss hyperparams

        self.alpha_prof = alpha_prof
        self.alpha_mach = alpha_mach
        self.beta_stoch = beta_stoch
        self.beta_mach_unsup = beta_mach_unsup
        if beta_mach_sup == 0.0:
            self.beta_mach_sup = self.beta_mach_unsup
        else:
            self.beta_mach_sup = beta_mach_sup

        self.physics = physics
        self.gamma_stored_energy = gamma_stored_energy
        self.gamma_bpol = gamma_bpol
        self.gamma_beta = gamma_beta
        self.loss_type = loss_type

        # ENCDER & DECDER
        end_conv_size = get_conv_output_size(out_length, len(self.hidden_dims))
        self.encoder = ENCODER(in_ch=in_ch, hidden_dims=hidden_dims)

        self.encoder_end = nn.Linear((self.hidden_dims[-1] * end_conv_size), self.encoder_end_dense_size)

        # Prior Regressor

        self.prior_reg = PRIORreg(in_dims=num_machine_params, mach_latent_dim=self.mach_latent_dim, hidden_dims=self.mp_hdims_cond)

        # Latent Space

        self.fc_mu_stoch = nn.Linear(self.encoder_end_dense_size, self.stoch_latent_dim)
        self.fc_var_stoch = nn.Linear(self.encoder_end_dense_size, self.stoch_latent_dim)

        self.fc_mu_mach = nn.Linear(self.encoder_end_dense_size, self.mach_latent_dim)
        self.fc_var_mach = nn.Linear(self.encoder_end_dense_size, self.mach_latent_dim)

        # DECODER

        self.decoder_input = nn.Linear(self.stoch_latent_dim + self.mach_latent_dim, self.hidden_dims[-1]*end_conv_size)
        reversed_dims = hidden_dims[::-1]
        reversed_dims.append(in_ch)
        self.decoder = DECODER(end_ch=in_ch, hidden_dims = reversed_dims, end_conv_size=end_conv_size) # DECODER(hidden_dims = self.hidden_dims[::-1], end_conv_size=end_conv_size)
        final_size = self.decoder.final_size
        self.final_layer = nn.Linear(final_size, out_length)
        # self.final_layer_t = nn.Linear(final_size, out_length)
        # self.final_layer_elm = nn.Linear(final_size, out_length)

        # AUX REG
        self.aux_reg = AUXreg(z_mach_dim=self.mach_latent_dim, mp_size=num_machine_params, hidden_dims=self.mp_hdims_aux)

    def forward(self, in_profs: Tensor, in_mp: Tensor = None) -> Tensor:

        prior_mu, prior_stoch = self.p_zmachx(in_mp)
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.q_zy(in_profs)
        z_stoch, z_mach = self.reparameterize(mu_stoch, log_var_stoch), self.reparameterize(mu_mach, log_var_mach)
        z = torch.cat((z_stoch, z_mach), 1)
        out_profs = self.p_yhatz(z)
        out_mp = self.q_hatxzmach(z_mach)
        assert in_profs.shape == out_profs.shape
        return {'prior_mu': prior_mu, 'prior_stoch': prior_stoch,
                'mu_stoch': mu_stoch, 'log_var_stoch': log_var_stoch, 'mu_mach': mu_mach, 'log_var_mach': log_var_mach,
                'out_profs': out_profs, 'in_profs': in_profs,
                'out_mp': out_mp, 'in_mp': in_mp}

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

        c_mu_mach, c_var_mach = self.prior_reg(machine_params)

        return [c_mu_mach, c_var_mach]

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
        # Propogate density and temperature profiles through respective encoders
        encoded_prof = self.encoder(profile)

        # Feed concat output into one last dense layer
        encoded_prof = self.encoder_end(encoded_prof)

        # Determine the latent variables
        mu_stoch = self.fc_mu_stoch(encoded_prof)
        var_stoch = self.fc_var_stoch(encoded_prof)
        mu_mach = self.fc_mu_mach(encoded_prof)
        var_mach = self.fc_var_mach(encoded_prof)

        return [mu_stoch, var_stoch, mu_mach, var_mach]

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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparam trick, sampling from N(mu, var) """

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

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
        out_mp = kwargs['out_mp']
        in_mp = kwargs['in_mp']
        device = in_profs.device
        start_sup_time = kwargs['start_sup_time']

        cutoff_2 = start_sup_time + 1000

        # This is really sub optimal, should cache these before.
        D_mu, D_var = kwargs['D_norms']
        T_mu, T_var = kwargs['T_norms']
        MP_mu, MP_var = kwargs['MP_norms']

        D_mu, D_var= D_mu.to(device), D_var.to(device)
        T_mu, T_var = T_mu.to(device), T_var.to(device)
        MP_mu, MP_var = MP_mu.to(device), MP_var.to(device)

        if 'mask' in kwargs:
            mask = kwargs['mask']
            recon_prof_loss = F.mse_loss(out_profs[mask], in_profs[mask])
        else:
            recon_prof_loss = F.mse_loss(out_profs, in_profs)

        #


        real_in_mps = torch.clone(in_mp)
        real_out_mps = torch.clone(out_mp)

        real_in_mps[:, :-1] = de_standardize(real_in_mps[:, :-1], MP_mu, MP_var)
        real_out_mps[:, :-1] = de_standardize(real_out_mps[:, :-1], MP_mu, MP_var)

        # Bpol approximation

        approx_bpol_in = bpol_approx(real_in_mps)
        approx_bpol_out = bpol_approx(real_out_mps)

        bpol_loss = F.mse_loss(approx_bpol_in, approx_bpol_out)

        # Static pressure stored energy
        real_in_profs = torch.clone(in_profs)
        real_in_profs = normalize_profiles(real_in_profs, T_mu, T_var, D_mu, D_var, de_norm=True)

        real_out_profs = torch.clone(out_profs)
        real_out_profs = normalize_profiles(real_out_profs, T_mu, T_var, D_mu, D_var, de_norm=True)

        stored_E_in = static_pressure_stored_energy_approximation(real_in_profs, mask)
        stored_E_out = static_pressure_stored_energy_approximation(real_out_profs, mask)

        stored_energy_loss = F.mse_loss(stored_E_in, stored_E_out)

        # Beta approximation
        approx_beta_in = beta_approximation(real_in_profs, real_in_mps)
        approx_beta_out = beta_approximation(real_out_profs, real_out_mps)

        beta_loss = F.mse_loss(approx_beta_in, approx_beta_out)

        physics_loss = torch.zeros_like(recon_prof_loss)

        if self.physics:
            physics_loss += self.gamma_stored_energy*stored_energy_loss
            physics_loss += self.gamma_bpol*bpol_loss
            physics_loss += self.gamma_beta*beta_loss

        recon_mp_loss = F.mse_loss(out_mp, in_mp)

        # Z_stoch latent space losses

        stoch_kld_loss = torch.distributions.kl.kl_divergence(
            torch.distributions.normal.Normal(mu_stoch, torch.exp(0.5*log_var_stoch)),
            torch.distributions.normal.Normal(0, 1)
            ).mean(0).sum()

        self.num_iterations += 1
        start_sup_time = 500
        if self.loss_type == 'semi-supervised-cutoff-increasing':
            if self.num_iterations > start_sup_time:
                beta_mach_unsup_new = get_new_beta_mach_sup(start_sup_time, cutoff_2, self.beta_mach_sup, self.beta_mach_unsup, self.num_iterations)
                sup_kld_loss =torch.distributions.kl.kl_divergence(
                 torch.distributions.normal.Normal(mu_mach, torch.exp(0.5*log_var_mach)),
                 torch.distributions.normal.Normal(prior_mu, torch.exp(0.5*prior_stoch))
                 ).mean(0).sum()
                supervised_loss = self.alpha_prof * recon_prof_loss + self.alpha_mach * recon_mp_loss + self.beta_stoch * stoch_kld_loss + beta_mach_unsup_new * sup_kld_loss + physics_loss

                return {'loss': supervised_loss, 'KLD_stoch': stoch_kld_loss, 'KLD_mach': sup_kld_loss, 'Reconstruction_Loss_mp': recon_mp_loss, 'Reconstruction_Loss': recon_prof_loss, 'Physics_all': physics_loss, 'static_stored_energy': stored_energy_loss, 'poloidal_field_approximation': bpol_loss, 'beta_approx': beta_loss}
            else:
                unsup_kld_loss = torch.distributions.kl.kl_divergence(
                 torch.distributions.normal.Normal(mu_mach, torch.exp(0.5*log_var_mach)),
                 torch.distributions.normal.Normal(0, 1)
                 ).mean(0).sum()

                unsupervised_loss = self.alpha_prof * recon_prof_loss + self.beta_stoch * stoch_kld_loss + self.beta_mach_unsup * unsup_kld_loss
                return {'loss': unsupervised_loss, 'KLD_stoch': stoch_kld_loss, 'KLD_mach': unsup_kld_loss, 'Reconstruction_Loss': recon_prof_loss, 'Reconstruction_Loss_mp': recon_mp_loss, 'Physics_all': physics_loss, 'static_stored_energy': stored_energy_loss, 'poloidal_field_approximation': bpol_loss, 'beta_approx': beta_loss}
        else:
            raise ValueError('Undefined Loss type, choose between unsupervised or supervised')
