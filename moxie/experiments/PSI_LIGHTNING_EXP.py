import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
import torch
from torch import optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

Tensor = TypeVar('torch.tensor')

def de_standardize(x, mu, var):
    return (x*var) + mu

class PSI_EXP(pl.LightningModule):
    def __init__(self, model=None, hparams: dict = {'LR': 0.001}) -> None:
        super(PSI_EXP, self).__init__()

        self.model = model
        self.params = hparams
        self.current_device = None
        self.learning_rate = hparams["LR"]
        self.scheduler_step_size = hparams["scheduler_step"]
        self.physics = hparams['physics']
        self.start_sup_time = hparams['start_sup_time']


    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_profile, machine_params, masks, ids = batch
        self.current_device = real_profile.device

        sample_batch_x, sample_batch_y = machine_params[0:1], real_profile[0:1]


        results = self.forward(real_profile, in_mp=machine_params)

        physics_dojo_results = (0.0, 0.0, 0.0, 0.0)
        train_loss = self.model.loss_function(**results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.train_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx, mask=masks, D_norms= self.trainer.datamodule.get_density_norms(), T_norms= self.trainer.datamodule.get_temperature_norms(), MP_norms = self.trainer.datamodule.get_machine_norms(), physics_dojo_results=physics_dojo_results, start_sup_time=self.start_sup_time)


        return train_loss

    def on_train_start(self):
        if self.current_epoch == 1:
            self.logger.experiment.add_hparams(hparam_dict= self.params)
            # self.logger.log_hyperparams(self.hparams, {"hp/final_loss": 0, "hp/recon": 0})

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        # Outputs is whatever that is returned from training_step

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD_stoch = torch.stack([x['KLD_stoch'] for x in outputs]).mean()
        avg_KLD_mach = torch.stack([x['KLD_mach'] for x in outputs]).mean()
        avg_recon_loss_mp = torch.stack([x['Reconstruction_Loss_mp'] for x in outputs]).mean()

        avg_physics_loss = torch.stack([x['Physics_all'] for x in outputs]).mean()
        avg_sse_loss = torch.stack([x['static_stored_energy'] for x in outputs]).mean()
        avg_bpol_loss = torch.stack([x['poloidal_field_approximation'] for x in outputs]).mean()
        avg_beta_loss = torch.stack([x['beta_approx'] for x in outputs]).mean()

        # else:
        #     avg_physics_loss = 0.0

        metrics = {'Loss/Train': avg_loss,
                    'ReconLoss/Train': avg_recon_loss,
                    'ReconLossMP/Train': avg_recon_loss_mp,
                    'KLD_stoch/Train': avg_KLD_stoch,
                    'KLD_mach/Train': avg_KLD_mach,
                    'physics/all/Train': avg_physics_loss,
                    'physics/sse/Train': avg_sse_loss,
                    'physics/bpol/Train': avg_bpol_loss,
                    'physics/beta/Train': avg_beta_loss}

        self.log_dict(metrics)
        epoch_dictionary = {'loss': avg_loss}


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_profile, machine_params, masks, ids = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)

        val_loss = self.model.loss_function(**results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.val_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx, mask=masks, D_norms= self.trainer.datamodule.get_density_norms(), T_norms= self.trainer.datamodule.get_temperature_norms(), MP_norms = self.trainer.datamodule.get_machine_norms(), start_sup_time=self.start_sup_time)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_recon_loss_mp = torch.stack([x['Reconstruction_Loss_mp'] for x in outputs]).mean()
        avg_KLD_stoch = torch.stack([x['KLD_stoch'] for x in outputs]).mean()
        avg_KLD_mach = torch.stack([x['KLD_mach'] for x in outputs]).mean()
        avg_bpol_loss = torch.stack([x['poloidal_field_approximation'] for x in outputs]).mean()

        # Physics losses
        avg_physics_loss = torch.stack([x['Physics_all'] for x in outputs]).mean()
        avg_sse_loss = torch.stack([x['static_stored_energy'] for x in outputs]).mean()
        avg_beta_loss = torch.stack([x['beta_approx'] for x in outputs]).mean()

        metrics = {'Loss/Valid': avg_loss,
                    'ReconLoss/Valid': avg_recon_loss,
                    'ReconLossMP/Valid': avg_recon_loss_mp,
                    'KLD_stoch/Valid': avg_KLD_stoch,
                    'KLD_mach/Valid': avg_KLD_mach,
                    'physics/all/Valid': avg_physics_loss,
                    'physics/sse/Valid': avg_sse_loss,
                    'physics/bpol/Valid': avg_bpol_loss,
                    'physics/beta/Valid': avg_beta_loss}


        self.log_dict(metrics)


    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, machine_params, masks, ids = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)
        test_loss = self.model.loss_function(**results, machine_params=machine_params, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx, mask=masks, D_norms= self.trainer.datamodule.get_density_norms(), T_norms= self.trainer.datamodule.get_temperature_norms(), start_sup_time=self.start_sup_time, MP_norms = self.trainer.datamodule.get_machine_norms(), )
        # Log the computational Graph!
        # self.logger.experiment.add_graph(self.model, [real_profile, machine_params], use_strict_trace=False)

        return test_loss

    def test_epoch_end(self, outputs):

        # self.compare_generate_with_real()
        all_components = self.get_cond_enc_real_for_comparison()

        self.compare_pressures(all_components)
        self.compare_cond_with_real(all_components)
        self.compare_correlations()

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        if self.scheduler_step_size > 0.0:
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=0.5),
                'name': 'StepLR'
                }
        else:
            return [optimizer]
        return [optimizer], [lr_scheduler]

    def physics_dojo(self, batch_x, interp_size=100, mp_idx=-5, mp_lims=(-0.5e6, -5e6)):
        MP_norm, MP_var = self.trainer.datamodule.get_machine_norms(device=self.current_device)
        D_norm, D_var = self.trainer.datamodule.get_density_norms(device=self.current_device)
        T_norm, T_var = self.trainer.datamodule.get_temperature_norms(device=self.current_device)

        mp_interp = standardize(torch.linspace(mp_lims[0], mp_lims[1], interp_size, device=self.current_device), MP_norm[mp_idx], MP_var[mp_idx])

        interp_sample_mp = torch.repeat_interleave(batch_x, interp_size, dim=0)
        interp_sample_mp[:, mp_idx] = mp_interp

        # calculate new Q!
        interp_sample_mp = replace_q95_with_qcly(interp_sample_mp)

        # Feed to the Prior Reg
        cond_prior_mu, cond_prior_var = self.model.p_zmachx(interp_sample_mp)

        # The latent space from prior reg
        cond_z_mach, z_stoch = self.model.reparameterize(cond_prior_mu, cond_prior_var), torch.distributions.normal.Normal(0, 1).sample((interp_size, 3)).to(self.current_device)
        z_cond = torch.cat((z_stoch, cond_z_mach), 1)

        # Predict the profiles and the machine parameters
        out_profs_cond, out_mp_cond = self.model.p_yhatz(z_cond), self.model.q_hatxzmach(cond_z_mach)

        # Check the density limit in the out profs  (negative densities)
        out_profs_cond_destand = torch.clone(out_profs_cond)
        out_profs_cond_destand[:, 0] = de_standardize(out_profs_cond_destand[:, 0], D_norm, D_var)
        out_profs_cond_destand[:, 1] = de_standardize(out_profs_cond_destand[:, 1], T_norm, T_var)


        out_profs_cond_destand_clamped = torch.clamp(out_profs_cond_destand, min=None, max=0.0)

        out_profs_cond_stand_clamped = torch.clone(out_profs_cond_destand_clamped)
        out_profs_cond_stand_clamped[:, 0] = standardize(out_profs_cond_stand_clamped[:, 0], D_norm, D_var)
        out_profs_cond_stand_clamped[:, 1] = standardize(out_profs_cond_stand_clamped[:, 1], D_norm, D_var)

        out_profs_comparison = torch.zeros_like(out_profs_cond_destand)
        out_profs_comparison[:, 0] = standardize(out_profs_comparison[:, 0], D_norm, D_var)
        out_profs_comparison[:, 1] = standardize(out_profs_comparison[:, 1], D_norm, D_var)

        # compare_density_limit = F.mse_loss(out_profs_comparison, out_profs_cond_stand_clamped)
        # Feed profiles to the encoder
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(out_profs_cond)
        encoded_z_stoch, encoded_z_mach = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)
        z_encoded = torch.cat((encoded_z_stoch, encoded_z_mach), 1)

        # Grab the predicted machine parameters
        out_mp_encoded = self.model.q_hatxzmach(encoded_z_mach)

        # Compare two out mps
        # compare_mp_loss = F.mse_loss(out_mp_encoded, out_mp_cond)

        return out_profs_comparison, out_profs_cond_stand_clamped, out_mp_encoded, interp_sample_mp

    def get_cond_enc_real_for_comparison(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof_og, train_mp, train_mask, ids = next(train_data_iter)
        val_prof_og, val_mp, val_mask, ids = next(val_data_iter)
        test_prof_og, test_mp, test_mask, ids = next(test_data_iter)

        mu_D, var_D = self.trainer.datamodule.get_density_norms()
        mu_T, var_T = self.trainer.datamodule.get_temperature_norms()

        with torch.no_grad():
            c_mu, c_var = self.model.p_zmachx(train_mp)
            z_mach_train = self.model.reparameterize(c_mu, c_var)
            z_stoch_train = torch.distributions.normal.Normal(0,1).sample((z_mach_train.shape[0], self.model.stoch_latent_dim))
            z_train = torch.cat((z_stoch_train, z_mach_train), 1)
            out_profs_train = self.model.p_yhatz(z_train)
            out_mp_train = self.model.q_hatxzmach(z_mach_train)

            c_mu, c_var = self.model.p_zmachx(val_mp)
            z_mach_val = self.model.reparameterize(c_mu, c_var)
            z_stoch_val = torch.distributions.normal.Normal(0,1).sample((z_mach_val.shape[0], self.model.stoch_latent_dim))
            z_val = torch.cat((z_stoch_val, z_mach_val), 1)
            out_profs_val = self.model.p_yhatz(z_val)
            out_mp_val = self.model.q_hatxzmach(z_mach_val)

            c_mu, c_var = self.model.p_zmachx(test_mp)
            z_mach_test = self.model.reparameterize(c_mu, c_var)
            z_stoch_test = torch.distributions.normal.Normal(0,1).sample((z_mach_test.shape[0], self.model.stoch_latent_dim))
            z_test = torch.cat((z_stoch_test, z_mach_test), 1)
            out_profs_test = self.model.p_yhatz(z_test)
            out_mp_test = self.model.q_hatxzmach(z_mach_test)

            train_results_enc = self.model.forward(train_prof_og, train_mp) # recons, input, mu, logvar
            val_results_enc = self.model.forward(val_prof_og, val_mp) # recons, input, mu, logvar
            test_results_enc = self.model.forward(test_prof_og, test_mp)

        # Density from conditional
        train_density_cond = out_profs_train[:, 0:1, :]
        val_density_cond = out_profs_val[:, 0:1, :]
        test_density_cond = out_profs_test[:, 0:1, :]

        train_density_cond = de_standardize(train_density_cond, mu_D, var_D)
        val_density_cond = de_standardize(val_density_cond, mu_D, var_D)
        test_density_cond = de_standardize(test_density_cond, mu_D, var_D)

        # Density from encoder
        train_density_enc = train_results_enc['out_profs'][:, 0:1,  :]
        val_density_enc = val_results_enc['out_profs'][:, 0:1, :]
        test_density_enc = test_results_enc['out_profs'][:, 0:1, :]

        train_density_enc = de_standardize(train_density_enc, mu_D, var_D)
        val_density_enc = de_standardize(val_density_enc, mu_D, var_D)
        test_density_enc = de_standardize(test_density_enc, mu_D, var_D)

        # Real Density
        train_density_real = train_prof_og[:, 0:1,:]
        val_density_real = val_prof_og[:, 0:1, :]
        test_density_real = test_prof_og[:, 0:1, :]

        train_density_real = de_standardize(train_density_real, mu_D, var_D)
        val_density_real = de_standardize(val_density_real, mu_D, var_D)
        test_density_real = de_standardize(test_density_real, mu_D, var_D)

        # Temperature from conditional
        train_temperature_cond = out_profs_train[:, 1:2, :]
        val_temperature_cond = out_profs_val[:, 1:2, :]
        test_temperature_cond = out_profs_test[:, 1:2, :]

        train_temperature_cond = de_standardize(train_temperature_cond, mu_T, var_T)
        val_temperature_cond = de_standardize(val_temperature_cond, mu_T, var_T)
        test_temperature_cond = de_standardize(test_temperature_cond, mu_T, var_T)

        # Temperature from encoder
        train_temperature_enc = train_results_enc['out_profs'][:, 1:2,  :]
        val_temperature_enc = val_results_enc['out_profs'][:, 1:2, :]
        test_temperature_enc = test_results_enc['out_profs'][:, 1:2, :]

        train_temperature_enc = de_standardize(train_temperature_enc, mu_T, var_T)
        val_temperature_enc = de_standardize(val_temperature_enc, mu_T, var_T)
        test_temperature_enc = de_standardize(test_temperature_enc, mu_T, var_T)

        # Real Temperature
        train_temperature_real = train_prof_og[:, 1:2,:]
        val_temperature_real = val_prof_og[:, 1:2, :]
        test_temperature_real = test_prof_og[:, 1:2, :]


        train_temperature_real = de_standardize(train_temperature_real, mu_T, var_T)
        val_temperature_real = de_standardize(val_temperature_real, mu_T, var_T)
        test_temperature_real = de_standardize(test_temperature_real, mu_T, var_T)

        return (train_density_cond, val_density_cond, test_density_cond), (train_density_enc, val_density_enc, test_density_enc), (train_density_real, val_density_real, test_density_real),(train_temperature_cond, val_temperature_cond, test_temperature_cond), (train_temperature_enc, val_temperature_enc, test_temperature_enc), (train_temperature_real, val_temperature_real, test_temperature_real)

    def compare_pressures(self, all_components):
        (train_density_cond, val_density_cond, test_density_cond), (train_density_enc, val_density_enc, test_density_enc), (train_density_real, val_density_real, test_density_real),(train_temperature_cond, val_temperature_cond, test_temperature_cond), (train_temperature_enc, val_temperature_enc, test_temperature_enc), (train_temperature_real, val_temperature_real, test_temperature_real) = all_components

        # Train plot
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True, sharey='col',  sharex=True)

        k = 50
        axs[0, 0].plot(train_temperature_cond[k].squeeze(), label='Cond', lw=4)
        axs[0, 1].plot(train_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[0, 2].plot(train_density_cond[k].squeeze() * train_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[0, 0].plot(train_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[0, 1].plot(train_density_real[k].squeeze(), label='Real', lw=4)
        axs[0, 2].plot(train_density_real[k].squeeze() * train_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[0, 0].plot(train_temperature_enc[k].squeeze(), label='Enc', lw=4)
        axs[0, 1].plot(train_density_enc[k].squeeze(), label='Real', lw=4)
        axs[0, 2].plot(train_density_enc[k].squeeze() * train_temperature_enc[k].squeeze(), label='Real', lw=4)

        k = 150
        axs[1, 0].plot(train_temperature_cond[k].squeeze(), label='Cond', lw=4)
        axs[1, 1].plot(train_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[1, 2].plot(train_density_cond[k].squeeze() * train_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[1, 0].plot(train_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[1, 1].plot(train_density_real[k].squeeze(), label='Real', lw=4)
        axs[1, 2].plot(train_density_real[k].squeeze() * train_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[1, 0].plot(train_temperature_enc[k].squeeze(), label='Enc', lw=4)
        axs[1, 1].plot(train_density_enc[k].squeeze(), label='Enc', lw=4)
        axs[1, 2].plot(train_density_enc[k].squeeze() * train_temperature_enc[k].squeeze(), label='Real', lw=4)


        k = 250
        axs[2, 0].plot(train_temperature_cond[k].squeeze(), label='Cond', lw=4)
        axs[2, 1].plot(train_density_cond[k].squeeze(), label='Cond', lw=4)
        axs[2, 2].plot(train_density_cond[k].squeeze() * train_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[2, 0].plot(train_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[2, 1].plot(train_density_real[k].squeeze(), label='Real', lw=4)
        axs[2, 2].plot(train_density_real[k].squeeze() * train_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[2, 0].plot(train_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[2, 1].plot(train_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[2, 2].plot(train_density_enc[k].squeeze() * train_temperature_enc[k].squeeze(), label='Real', lw=4)

        axs[0, 0].legend()
        axs[0, 1].set_ylabel('$n_e \; \;$ m$^{-3})$')
        axs[0, 0].set_ylabel('$T_e \; \; (eV)$', size='xx-large')
        axs[0, 2].set_ylabel('$p_e \; \;$ (Pa)')

        fig.suptitle('Training Profiles Reconstruction')
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('pressure_profs/training', fig)

        # Validation
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True, sharey='col',  sharex=True)

        k = 75
        axs[0, 0].plot(val_temperature_cond[k].squeeze(), label='Conditional', lw=4)
        axs[0, 1].plot(val_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[0, 2].plot(val_density_cond[k].squeeze() * val_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[0, 0].plot(val_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[0, 1].plot(val_density_real[k].squeeze(), label='Real', lw=4)
        axs[0, 2].plot(val_density_real[k].squeeze() * val_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[0, 0].plot(val_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[0, 1].plot(val_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[0, 2].plot(val_density_enc[k].squeeze() * val_temperature_enc[k].squeeze(), label='Encoder', lw=4)

        k = 126
        axs[1, 0].plot(val_temperature_cond[k].squeeze(), label='Conditional', lw=4)
        axs[1, 1].plot(val_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[1, 2].plot(val_density_cond[k].squeeze() * val_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[1, 0].plot(val_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[1, 1].plot(val_density_real[k].squeeze(), label='Real', lw=4)
        axs[1, 2].plot(val_density_real[k].squeeze() * val_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[1, 0].plot(val_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[1, 1].plot(val_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[1, 2].plot(val_density_enc[k].squeeze() * val_temperature_enc[k].squeeze(), label='Encoder', lw=4)


        k = 250
        axs[2, 0].plot(val_temperature_cond[k].squeeze(), label='Conditional', lw=4)
        axs[2, 1].plot(val_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[2, 2].plot(val_density_cond[k].squeeze() * val_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[2, 0].plot(val_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[2, 1].plot(val_density_real[k].squeeze(), label='Real', lw=4)
        axs[2, 2].plot(val_density_real[k].squeeze() * val_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[2, 0].plot(val_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[2, 1].plot(val_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[2, 2].plot(val_density_enc[k].squeeze() * val_temperature_enc[k].squeeze(), label='Encoder', lw=4)

        axs[0, 1].set_ylabel('$n_e \; \;$ m$^{-3})$', size='xx-large')
        axs[0, 0].set_ylabel('$T_e \; \; (eV)$', size='xx-large')
        axs[0, 2].set_ylabel('$p_e \; \;$ (Pa)', size='xx-large')

        axs[0, 0].legend()
        fig.suptitle('Val Profiles Reconstruction')
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('pressure_profs/val', fig)
        """
        # Test
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True, sharey='col',  sharex=True)

        k = 43
        axs[0, 0].plot(test_temperature_cond[k].squeeze(), label='Conditional', lw=4)
        axs[0, 1].plot(test_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[0, 2].plot(test_density_cond[k].squeeze() * test_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[0, 0].plot(test_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[0, 1].plot(test_density_real[k].squeeze(), label='Real', lw=4)
        axs[0, 2].plot(test_density_real[k].squeeze() * test_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[0, 0].plot(test_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[0, 1].plot(test_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[0, 2].plot(test_density_enc[k].squeeze() * test_temperature_enc[k].squeeze(), label='Encoder', lw=4)

        k = 124
        axs[1, 0].plot(test_temperature_cond[k].squeeze(), label='Conditional', lw=4)
        axs[1, 1].plot(test_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[1, 2].plot(test_density_cond[k].squeeze() * test_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[1, 0].plot(test_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[1, 1].plot(test_density_real[k].squeeze(), label='Real', lw=4)
        axs[1, 2].plot(test_density_real[k].squeeze() * test_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[1, 0].plot(test_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[1, 1].plot(test_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[1, 2].plot(test_density_enc[k].squeeze() * test_temperature_enc[k].squeeze(), label='Encoder', lw=4)


        k = 233
        axs[2, 0].plot(test_temperature_cond[k].squeeze(), label='Conditional', lw=4)
        axs[2, 1].plot(test_density_cond[k].squeeze(), label='Conditional', lw=4)
        axs[2, 2].plot(test_density_cond[k].squeeze() * test_temperature_cond[k].squeeze(), label='Conditional', lw=4)

        axs[2, 0].plot(test_temperature_real[k].squeeze(), label='Real', lw=4)
        axs[2, 1].plot(test_density_real[k].squeeze(), label='Real', lw=4)
        axs[2, 2].plot(test_density_real[k].squeeze() * test_temperature_real[k].squeeze(), label='Real', lw=4)

        axs[2, 0].plot(test_temperature_enc[k].squeeze(), label='Encoder', lw=4)
        axs[2, 1].plot(test_density_enc[k].squeeze(), label='Encoder', lw=4)
        axs[2, 2].plot(test_density_enc[k].squeeze() * test_temperature_enc[k].squeeze(), label='Encoder', lw=4)

        axs[0, 0].legend()
        axs[0, 1].set_ylabel('$n_e \; \;$ m$^{-3})$')
        axs[0, 0].set_ylabel('$T_e \; \; (eV)$', size='xx-large')
        axs[0, 2].set_ylabel('$p_e \; \;$ (Pa)')

        fig.suptitle('Test Profiles Reconstruction')
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('pressure_profs/test', fig)
        """


    def compare_cond_with_real(self, all_components):

        (train_density_cond, val_density_cond, test_density_cond), (train_density_enc, val_density_enc, test_density_enc), (train_density_real, val_density_real, test_density_real),(train_temperature_cond, val_temperature_cond, test_temperature_cond), (train_temperature_enc, val_temperature_enc, test_temperature_enc), (train_temperature_real, val_temperature_real, test_temperature_real) = all_components

        fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True, sharex=True, sharey=True)

        for k in [0, 1, 2]:

            axs[0, k].plot(train_temperature_cond[k*100].squeeze(), label='Conditional', lw=4)
            axs[0, k].plot(train_temperature_real[k*100].squeeze(), label='Real', lw=4)
            axs[0, k].plot(train_temperature_enc[k*100].squeeze(), label='Encoder', lw=4)

            axs[1, k].plot(val_temperature_cond[k*100].squeeze(), label='Conditional', lw=4)
            axs[1, k].plot(val_temperature_real[k*100].squeeze(), label='Real', lw=4)
            axs[1, k].plot(val_temperature_enc[k*100].squeeze(), label='Encoder', lw=4)

            axs[2, k].plot(test_temperature_cond[k*100].squeeze(), label='Conditional', lw=4)
            axs[2, k].plot(test_temperature_real[k*100].squeeze(), label='Real', lw=4)
            axs[2, k].plot(test_temperature_enc[k*100].squeeze(), label='Encoder', lw=4)

            if k == 0:
                axs[0, k].set_ylabel('Train:') # {:.4}'.format(train_loss['Reconstruction_Loss']) , size='x-large')
                axs[1, k].set_ylabel('Valid')#  {:.4}'.format(val_loss['Reconstruction_Loss']), size='x-large')
                axs[2, k].set_ylabel('Test')#  {:.4}'.format(test_loss['Reconstruction_Loss']), size='x-large')
                axs[0, k].legend()
                axs[1, k].legend()
                axs[2, k].legend()

        fig.supxlabel('R', size='xx-large')
        fig.supylabel('$T_e \; \; $', size='xx-large')

        fig.suptitle('DIVA From Conditional Priors'.format(self.model.stoch_latent_dim, self.model.mach_latent_dim))
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('temperature_from_cond', fig)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True, sharex=True, sharey=True)

        for k in [0, 1, 2]:

            axs[0, k].plot(train_density_cond[k*100].squeeze(), label='Conditional', lw=4)
            axs[0, k].plot(train_density_real[k*100].squeeze(), label='Real', lw=4)
            axs[0, k].plot(train_density_enc[k*100].squeeze(), label='Encoder', lw=4)

            axs[1, k].plot(val_density_real[k*100].squeeze(), label='Conditional', lw=4)
            axs[1, k].plot(val_density_cond[k*100].squeeze(), label='Real', lw=4)
            axs[1, k].plot(val_density_enc[k*100].squeeze(), label='Encoder', lw=4)

            axs[2, k].plot(test_density_real[k*100].squeeze(), label='Conditional', lw=4)
            axs[2, k].plot(test_density_cond[k*100].squeeze(), label='Real', lw=4)
            axs[2, k].plot(test_density_enc[k*100].squeeze(), label='Encoder', lw=4)

            if k == 0:
                axs[0, k].set_ylabel('Train:') # {:.4}'.format(train_loss['Reconstruction_Loss']) , size='x-large')
                axs[1, k].set_ylabel('Valid')#  {:.4}'.format(val_loss['Reconstruction_Loss']), size='x-large')
                axs[2, k].set_ylabel('Test')#  {:.4}'.format(test_loss['Reconstruction_Loss']), size='x-large')
                axs[0, k].legend()
                axs[1, k].legend()
                axs[2, k].legend()

        fig.supxlabel('R', size='xx-large')
        fig.supylabel('$n_e \; \; (10^{20}$ m$^{-3})$', size='xx-large')
        fig.suptitle('DIVA From Conditional Priors'.format(self.model.stoch_latent_dim, self.model.mach_latent_dim))
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('density_from_cond', fig)



    def compare_generate_with_real(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof_og, train_mp, train_mask, ids = next(train_data_iter)
        val_prof_og, val_mp, val_mask, ids = next(val_data_iter)
        test_prof_og, test_mp, test_mask, ids = next(test_data_iter)

        mu_D, var_D = self.trainer.datamodule.get_density_norms()
        mu_T, var_T = self.trainer.datamodule.get_temperature_norms()

        train_results = self.model.forward(train_prof_og, train_mp) # recons, input, mu, logvar
        val_results = self.model.forward(val_prof_og, val_mp) # recons, input, mu, logvar
        test_results = self.model.forward(test_prof_og, test_mp) # recons, input, mu, logvar

        train_loss = self.model.loss_function(**train_results, mask=train_mask, D_norms=(mu_D, var_D), T_norms=(mu_T, var_T))

        val_loss = self.model.loss_function(**val_results, mask=val_mask, D_norms=(mu_D, var_D), T_norms=(mu_T, var_T))
        test_loss = self.model.loss_function(**test_results, mask=test_mask, D_norms=(mu_D, var_D), T_norms=(mu_T, var_T))



        train_res = train_results['out_profs'][:, 0:1,  :]
        val_res = val_results['out_profs'][:, 0:1, :]
        test_res = test_results['out_profs'][:, 0:1, :]

        train_res = de_standardize(train_res, mu_D, var_D)
        val_res = de_standardize(val_res, mu_D, var_D)
        test_res = de_standardize(test_res, mu_D, var_D)

        train_mask = train_mask[:, 0:1, :].squeeze()
        val_mask = val_mask[:, 0:1, :].squeeze()
        test_mask = test_mask[:, 0:1, :].squeeze()

        train_prof=train_prof_og[:, 0:1,:]
        val_prof=val_prof_og[:, 0:1, :]
        test_prof=test_prof_og[:, 0:1, :]


        train_prof = de_standardize(train_prof, mu_D, var_D)
        val_prof = de_standardize(val_prof, mu_D, var_D)
        test_prof = de_standardize(test_prof, mu_D, var_D)
        fig, axs = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True, sharex=True, sharey=True)

        for k in [0, 1, 2]:

            axs[0, k].plot(train_res[k*100].squeeze()[train_mask[k*100]], label='Generated', lw=4)
            axs[0, k].plot(train_prof[k*100].squeeze()[train_mask[k*100]], label='Real', lw=4)
            axs[1, k].plot(val_res[k*100].squeeze()[val_mask[k*100]], label='Generated', lw=4)
            axs[1, k].plot(val_prof[k*100].squeeze()[val_mask[k*100]], label='Real', lw=4)
            axs[2, k].plot(test_res[k*100].squeeze()[test_mask[k*100]], label='Generated', lw=4)
            axs[2, k].plot(test_prof[k*100].squeeze()[test_mask[k*100]], label='Real', lw=4)

            if k == 0:
                axs[0, k].set_ylabel('Train: {:.4}'.format(train_loss['Reconstruction_Loss']) , size='x-large')
                axs[1, k].set_ylabel('Valid {:.4}'.format(val_loss['Reconstruction_Loss']), size='x-large')
                axs[2, k].set_ylabel('Test {:.4}'.format(test_loss['Reconstruction_Loss']), size='x-large')
                axs[0, k].legend()
                axs[1, k].legend()
                axs[2, k].legend()

        fig.supxlabel('R', size='xx-large')
        fig.supylabel('$n_e \; \; (10^{20}$ m$^{-3})$', size='xx-large')
        fig.suptitle('DIVA: {}-D Stoch, {}-D Mach'.format(self.model.stoch_latent_dim, self.model.mach_latent_dim))
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('comparison_density', fig)

        train_res = train_results['out_profs'][:, 1:, :]
        val_res = val_results['out_profs'][:, 1:, :]
        test_res = test_results['out_profs'][:, 1:, :]

        train_prof=train_prof_og[:, 1:, :]
        val_prof=val_prof_og[:, 1:, :]
        test_prof=test_prof_og[:, 1:, :]



        # print(train_res[:, :])
        train_res = de_standardize(train_res, mu_T, var_T)

        val_res = de_standardize(val_res, mu_T, var_T)
        test_res = de_standardize(test_res, mu_T, var_T)

        train_prof = de_standardize(train_prof, mu_T, var_T)
        val_prof = de_standardize(val_prof, mu_T, var_T)
        test_prof = de_standardize(test_prof, mu_T, var_T)


        fig, axs = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True, sharex=True, sharey=True)

        for k in [0, 1, 2]:
            axs[0, k].plot(train_res[k*100].squeeze()[train_mask[k*100]], label='Generated', lw=4)
            axs[0, k].plot(train_prof[k*100].squeeze()[train_mask[k*100]], label='Real', lw=4)
            axs[1, k].plot(val_res[k*100].squeeze()[val_mask[k*100]], label='Generated', lw=4)
            axs[1, k].plot(val_prof[k*100].squeeze()[val_mask[k*100]], label='Real', lw=4)
            axs[2, k].plot(test_res[k*100].squeeze()[test_mask[k*100]], label='Generated', lw=4)
            axs[2, k].plot(test_prof[k*100].squeeze()[test_mask[k*100]], label='Real', lw=4)

            if k == 0:
                axs[0, k].set_ylabel('Train: {:.4}'.format(train_loss['Reconstruction_Loss']) , size='x-large')
                axs[1, k].set_ylabel('Valid {:.4}'.format(val_loss['Reconstruction_Loss']), size='x-large')
                axs[2, k].set_ylabel('Test {:.4}'.format(test_loss['Reconstruction_Loss']), size='x-large')
                axs[0, k].legend()
                axs[1, k].legend()
                axs[2, k].legend()

        fig.supxlabel('R', size='xx-large')
        fig.supylabel('$T_e \; \; $', size='xx-large')
        fig.suptitle('DIVA: {}-D Stoch, {}-D Mach'.format(self.model.stoch_latent_dim, self.model.mach_latent_dim))
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('comparison_temperature', fig)
        # plt.show()
    def compare_correlations(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        # train_prof_og, train_mp, train_mask, ids
        train_prof, train_mp, _, _, = next(train_data_iter)
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(train_prof)
        z_stoch, z_mach = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)

        z_mach_all = z_mach
        z_stoch_all = z_stoch
        in_all = train_mp

        for (train_prof, train_mp, _, _) in train_data_iter:
            mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(train_prof)
            z_stoch, z_mach = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)
            z_mach_all = torch.vstack((z_mach, z_mach_all))
            z_stoch_all = torch.vstack((z_stoch, z_stoch_all))
            in_all = torch.vstack((train_mp, in_all))
        self.plot_corr_matrix(z_mach_all, in_all)
        self.plot_corr_matrix(z_stoch_all, in_all, title='Z_stoch')


    def plot_corr_matrix(self, z, val_params, title='Z_machine'):
        LABEL = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'ELON', 'POHM', 'IPLA', 'BVAC', 'NBI', 'ICRH', 'ELER']

        # LABEL = ['BT', 'CR0', 'ELER', 'ELON', 'POHM', 'P_ICRH', 'P_NBI', 'Q95', 'RGEO', 'TRIL', 'TRIU', 'VOLM', 'XIP']

        fig, axs = plt.subplots(figsize=(10,10))
        all_cors = []
        for i in range(z.shape[1]):
            # val_params[np.isnan(val_params)] = 0
            single_dim = z[:, i]
            correlation = np.cov(single_dim,val_params, rowvar=False) # the first column is the correlation with hidden dim and other params
            correlation = correlation[:, 0][1:]
            all_cors.append(correlation)
        all_cors = np.stack(all_cors)
        all_cors = torch.from_numpy(all_cors)
        im = axs.imshow(all_cors, cmap='viridis')
        axs.set_xticks(np.arange(len(LABEL)))
        axs.set_xticklabels(LABEL)
        axs.set_yticks(np.arange(z.shape[1]))
        plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        plt.colorbar(im)
        plt.title(title)
        self.logger.experiment.add_figure('correlation_' + title, fig)
        return fig
