import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
Tensor = TypeVar('torch.tensor')
import PIL.Image
import torch
from torch import optim
# from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import io
# from..models.VAE import BaseVAE


# def plot_to_image(figure):

def de_standardize(x, mu, var):
    return (x*var) + mu


class DIVA_EXP(pl.LightningModule):

    def __init__(self, vae_model: None, params: dict) -> None:
        super(DIVA_EXP, self).__init__()

        self.model = vae_model
        self.params = params
        self.current_device = None
        self.learning_rate = params['LR']
        self.save_hyperparameters(params)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_profile, machine_params = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)
        train_loss = self.model.loss_function(**results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.train_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        return train_loss

    def on_train_start(self):
        if self.current_epoch == 1:
            self.logger.log_hyperparams(self.hparams, {"hp/final_loss": 0, "hp/recon": 0})

    def training_epoch_end(self, outputs):
        # Outputs is whatever that is returned from training_step

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD_stoch = torch.stack([x['KLD_stoch'] for x in outputs]).mean()
        avg_KLD_mach = torch.stack([x['KLD_mach'] for x in outputs]).mean()
        avg_recon_loss_mp = torch.stack([x['Reconstruction_Loss_mp'] for x in outputs]).mean()

        metrics = {'Loss/Train': avg_loss,
                'ReconLoss/Train': avg_recon_loss,
                'ReconLossMP/Train': avg_recon_loss_mp,
                'KLD_stoch/Train': avg_KLD_stoch,
                'KLD_mach/Train': avg_KLD_mach}

        self.log_dict(metrics)
        epoch_dictionary = {'loss': avg_loss}


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_profile, machine_params = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)

        val_loss = self.model.loss_function(**results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.val_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_recon_loss_mp = torch.stack([x['Reconstruction_Loss_mp'] for x in outputs]).mean()
        avg_KLD_stoch = torch.stack([x['KLD_stoch'] for x in outputs]).mean()
        avg_KLD_mach = torch.stack([x['KLD_mach'] for x in outputs]).mean()


        metrics = {'Loss/Valid': avg_loss,
                'ReconLoss/Valid': avg_recon_loss,
                'ReconLossMP/Valid': avg_recon_loss_mp,
                'KLD_stoch/Valid': avg_KLD_stoch,
                'KLD_mach/Valid': avg_KLD_mach}


        self.log_dict(metrics)


    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, machine_params = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)
        test_loss = self.model.loss_function(**results, machine_params=machine_params, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)
        # Log the computational Graph!
        # self.logger.experiment.add_graph(self.model, [real_profile, machine_params], use_strict_trace=False)

        return test_loss

    def test_epoch_end(self, outputs):

        self.compare_generate_with_real()
        self.compare_correlations()
        self.diversity_of_generation()

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        return optimizer

    def compare_means(self):
        pass

    def diversity_of_generation(self):
        """
        Want to check how the latent dimensions change the output profile.
        First up is the Z_stoch, i.e., how does varying Z_stoch change the output profile?
        """

        # Get training, val, test data

        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof_og, train_mp = next(train_data_iter)
        val_prof_og, val_mp = next(val_data_iter)
        test_prof_og, test_mp = next(test_data_iter)

        training_sample_profile = train_prof_og[0:1]
        val_sample_profile = val_prof_og[0:1]
        test_sample_profile = test_prof_og[0:1]

        # Encode training data into latent space
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(training_sample_profile)
        z_stoch_og, z_mach_og = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)

        print('\n# Original Stoch')
        print(z_stoch_og)
        print('\n# Original Mach')
        print(z_mach_og)

        # Decode Original Data
        z_og = torch.cat((z_stoch_og, z_mach_og), 1)
        out_profs_og = self.model.p_yhatz(z_og)

        # Plot data
        # Rows correspond to different sampling: 5
        # Columns corrspond to density vs temperature: 2

        # Plot original data
        density_gen_og, temperature_gen_og = out_profs_og[:, 0], out_profs_og[:, 1]
        density_real_og, temperature_real_og = training_sample_profile[:, 0], training_sample_profile[:, 1]

        mu_T, var_T = self.trainer.datamodule.get_data_norms()

        # print(train_res[:, :])
        temperature_gen_og = de_standardize(temperature_gen_og, mu_T, var_T)
        temperature_real_og = de_standardize(temperature_real_og, mu_T, var_T)

        prior = torch.distributions.normal.Normal(0, 1)

        fig, axs = plt.subplots(5, 2, figsize=(18, 18), constrained_layout=True)
        axs[0, 0].plot(density_gen_og.squeeze(), label='Generated')
        axs[0, 0].plot(density_real_og.squeeze(), label='Real')
        axs[0, 1].plot(temperature_gen_og.squeeze(), label='Generated')
        axs[0, 1].plot(temperature_real_og.squeeze(), label='Real')
        axs[0, 0].set(title='Density', ylabel='Original')
        axs[0, 1].set(title='Temperature')
        axs[0, 0].legend()


        for k in [1, 2, 3, 4]:

            # Sample from Z_stoch = N(0, 1)
            new_stoch = prior.sample(sample_shape=z_stoch_og.shape)
            z_new = torch.cat((new_stoch, z_mach_og), 1)
            out_profs_new_stoch = self.model.p_yhatz(z_new)
            density_gen_stoch, temperature_gen_stoch = out_profs_new_stoch[:, 0], out_profs_new_stoch[:, 1]

            # Sample from Z_mach = N(0, 1)
            new_mach = prior.sample(sample_shape=z_mach_og.shape)

            z_new = torch.cat((z_stoch_og, new_mach), 1)
            out_profs_new_mach = self.model.p_yhatz(z_new)
            density_gen_mach, temperature_gen_mach = out_profs_new_mach[:, 0], out_profs_new_mach[:, 1]

            print('\n# Sample Stoch')
            print(new_stoch)
            print('\n# Sample Mach')
            print(new_mach)
            # Restandardize for plotting
            temperature_gen_stoch = de_standardize(temperature_gen_stoch, mu_T, var_T)
            temperature_gen_mach = de_standardize(temperature_gen_mach, mu_T, var_T)

            # plot
            axs[k, 0].plot(density_gen_og.squeeze())
            axs[k, 0].plot(density_real_og.squeeze())
            axs[k, 1].plot(temperature_gen_og.squeeze())
            axs[k, 1].plot(temperature_real_og.squeeze())


            axs[k, 0].plot(density_gen_stoch.squeeze(), label='Vary Z_stoch')
            axs[k, 1].plot(temperature_gen_stoch.squeeze(), label='Vary Z_stoch')
            axs[k, 0].plot(density_gen_mach.squeeze(), label='Vary Z_mach')
            axs[k, 1].plot(temperature_gen_mach.squeeze(), label='Vary Z_mach')


        axs[k, 0].legend()
        plt.show()



    def compare_generate_with_real(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof_og, train_mp = next(train_data_iter)
        val_prof_og, val_mp = next(val_data_iter)
        test_prof_og, test_mp = next(test_data_iter)

        train_results = self.model.forward(train_prof_og, train_mp) # recons, input, mu, logvar
        val_results = self.model.forward(val_prof_og, val_mp) # recons, input, mu, logvar
        test_results = self.model.forward(test_prof_og, test_mp) # recons, input, mu, logvar

        train_res = train_results['out_profs'][:, 0:1,  :]
        val_res = val_results['out_profs'][:, 0:1, :]
        test_res = test_results['out_profs'][:, 0:1, :]

        train_prof=train_prof_og[:, 0:1,:]
        val_prof=val_prof_og[:, 0:1, :]
        test_prof=test_prof_og[:, 0:1, :]

        fig, axs = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True, sharex=True, sharey=True)

        for k in [0, 1, 2]:

            axs[0, k].plot(train_res[k*100].squeeze(), label='Generated', lw=4)
            axs[0, k].plot(train_prof[k*100].squeeze(), label='Real', lw=4)
            axs[1, k].plot(val_res[k*100].squeeze(), label='Generated', lw=4)
            axs[1, k].plot(val_prof[k*100].squeeze(), label='Real', lw=4)
            axs[2, k].plot(test_res[k*100].squeeze(), label='Generated', lw=4)
            axs[2, k].plot(test_prof[k*100].squeeze(), label='Real', lw=4)

            if k == 0:
                axs[0, k].set_ylabel('Train', size='x-large')
                axs[1, k].set_ylabel('Valid', size='x-large')
                axs[2, k].set_ylabel('Test', size='x-large')
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

        mu_T, var_T = self.trainer.datamodule.get_data_norms()

        # print(train_res[:, :])
        train_res = de_standardize(train_res, mu_T, var_T)

        val_res = de_standardize(val_res, mu_T, var_T)
        test_res = de_standardize(test_res, mu_T, var_T)

        train_prof = de_standardize(train_prof, mu_T, var_T)
        val_prof = de_standardize(val_prof, mu_T, var_T)
        test_prof = de_standardize(test_prof, mu_T, var_T)


        fig, axs = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True, sharex=True, sharey=True)

        for k in [0, 1, 2]:

            axs[0, k].plot(train_res[k*100].squeeze(), label='Generated', lw=4)
            axs[0, k].plot(train_prof[k*100].squeeze(), label='Real', lw=4)
            axs[1, k].plot(val_res[k*100].squeeze(), label='Generated', lw=4)
            axs[1, k].plot(val_prof[k*100].squeeze(), label='Real', lw=4)
            axs[2, k].plot(test_res[k*100].squeeze(), label='Generated', lw=4)
            axs[2, k].plot(test_prof[k*100].squeeze(), label='Real', lw=4)

            if k == 0:
                axs[0, k].set_ylabel('Train', size='x-large')
                axs[1, k].set_ylabel('Valid', size='x-large')
                axs[2, k].set_ylabel('Test', size='x-large')
                axs[0, k].legend()
                axs[1, k].legend()
                axs[2, k].legend()

        fig.supxlabel('R', size='xx-large')
        fig.supylabel('$T_e \; \; $', size='xx-large')
        fig.suptitle('DIVA: {}-D Stoch, {}-D Mach'.format(self.model.stoch_latent_dim, self.model.mach_latent_dim))
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('comparison_temperature', fig)
        plt.show()

    def compare_correlations(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof, train_mp = next(train_data_iter)
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(train_prof)
        z_stoch, z_mach = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)

        z_mach_all = z_mach
        in_all = train_mp

        for (train_prof, train_mp) in train_data_iter:
            mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(train_prof)
            z_stoch, z_mach = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)
            z_mach_all = torch.vstack((z_mach, z_mach_all))
            in_all = torch.vstack((train_mp, in_all))
        self.plot_corr_matrix(z_mach_all, in_all)

    def plot_corr_matrix(self, z, val_params, title='Z_machine vs Machine Params'):
        LABEL = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'POHM', 'BT', 'ELER', 'P_NBI', 'P_ICRH']

        fig, axs = plt.subplots(figsize=(20,20))
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
        self.logger.experiment.add_figure('correlation', fig)
        return fig
