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

        logs = {'train_loss': train_loss}

        # self.logger.log_metrics({key: val.item() for key, val in train_loss.items()})

        batch_dictionary = {'loss': train_loss, 'log': logs}
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
        # avg_mach_loss = torch.stack([x['Machine_Loss'] for x in outputs]).mean()

        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLoss/Train', avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLossMP/Train', avg_recon_loss_mp, self.current_epoch)
        self.logger.experiment.add_scalar('KL_stoch/Train', avg_KLD_stoch, self.current_epoch)
        self.logger.experiment.add_scalar('KL_mach/Train', avg_KLD_mach, self.current_epoch)
        # self.logger.experiment.add_scalar('Mach_loss/Train', avg_mach_loss, self.current_epoch)

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
        avg_KLD_stoch = torch.stack([x['KLD_stoch'] for x in outputs]).mean()
        avg_KLD_mach = torch.stack([x['KLD_mach'] for x in outputs]).mean()
        avg_recon_loss_mp = torch.stack([x['Reconstruction_Loss_mp'] for x in outputs]).mean()
        # avg_mach_loss = torch.stack([x['Machine_Loss'] for x in outputs]).mean()

        self.logger.experiment.add_scalar('Loss/Valid', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLoss/Valid', avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLossMP/Valid', avg_recon_loss_mp, self.current_epoch)
        self.logger.experiment.add_scalar('KL_stoch/Valid', avg_KLD_stoch, self.current_epoch)
        self.logger.experiment.add_scalar('KL_mach/Valid', avg_KLD_mach, self.current_epoch)
        # self.logger.experiment.add_scalar('Mach_loss/Valid', avg_mach_loss, self.current_epoch)

        tensorboard_logs = {'avg_val_loss': avg_loss}


        self.log("hp/final_loss", avg_loss, on_epoch=True)
        self.log("hp_metric", avg_loss, on_epoch=True)
        self.log("hp/recon", avg_recon_loss, on_epoch=True)


    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, machine_params = batch
        # self.model.example_input_array = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)
        # generated_profiles = results[0]
        test_loss = self.model.loss_function(**results, machine_params=machine_params, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)


        # if batch_idx%25 == 0:
        #     mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.encode(real_profile)
        #    z_machine = self.model.reparameterize( mu_mach, log_var_mach)
        #    machine_space_corrs = self.plot_corr_matrix(z_machine, machine_params )
        #    self.logger.experiment.add_figure('machine_latent', machine_space_corrs)
        #    z_stoch = self.model.reparameterize(mu_stoch, log_var_stoch, )
        #    stoch_space_corrs = self.plot_corr_matrix(z_stoch, machine_params, title='Z_stoch vs Machine Params')
        #    self.logger.experiment.add_figure('stoch_latent', stoch_space_corrs)

        # val_params[:, -4] = val_params[:, -4]*(1E-21)
        """
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.encode(real_profile)
        z_stoch = self.model.reparameterize(mu_stoch, log_var_stoch)
        z_mach = self.model.reparameterize(mu_mach, log_var_mach)

        self.plot_latent_for_corr(z_stoch, machine_params)
        self.plot_latent_for_corr(z_mach, machine_params)
        plt.show()"""
        # Log the computational Graph!
        self.logger.experiment.add_graph(self.model, [real_profile, machine_params], use_strict_trace=False)

        return test_loss

    def test_epoch_end(self, outputs):
        # self.plot_per_latent_dim()
        # self.sample_single_profile()

        # self.plot_latent_space()
        # self.correlation_of_latent_space()
        # final_res = self.sample_profiles()
        # self.logger.experiment.add_figure('test_outputs', final_res)
        self.compare_generate_with_real()

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

    def compare_generate_with_real(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof, train_mp = next(train_data_iter)
        val_prof, val_mp = next(val_data_iter)
        test_prof, test_mp = next(test_data_iter)

        train_results = self.model.forward(train_prof, train_mp) # recons, input, mu, logvar
        val_results = self.model.forward(val_prof, val_mp) # recons, input, mu, logvar
        test_results = self.model.forward(test_prof, test_mp) # recons, input, mu, logvar

        train_res = train_results['out_profs']
        val_res = val_results['out_profs']
        test_res = test_results['out_profs']

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
        self.logger.experiment.add_figure('comparison', fig)
        plt.show()

    def sample_profiles(self):
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())
        fig = plt.figure(figsize=(18, 18), constrained_layout=True)
        gs = GridSpec(4, 3, figure=fig)
        # all_inputs, all_machine = torch.Tensor()
        for k in range(4):
            test_input, test_label = next(test_data_iter)
            data = self.model.forward(test_input)
            avg_loss = self.model.loss_function(*data,  machine_params= test_label,  M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
            recons, input, mu, logvar, _, _ = data


            ax = None
            for i in range(3):
                ax = fig.add_subplot(gs[k, i], sharey=ax, sharex=ax)
                # data = [recons[i], input[i], mu, logvar]
                # avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
                if len(test_input.shape) == 3:
                    input = input.squeeze()
                    recons= recons.squeeze()
                ax.set(ylabel='$n_e$')
                ax.plot(recons[-i]*self.trainer.datamodule.max_X, label='Generated')
                ax.plot(input[-i]*self.trainer.datamodule.max_X, label='Real')
                ax.set_xticks([])

            fig.suptitle('DualVae: {}-Hidden Layers {}-D Z_st'.format(len(self.model.hidden_dims), self.model.stoch_latent_dim))
        plt.legend()
        return fig
        # plt.show()

    def plot_corr_matrix(self, z, val_params, title='Z_machine vs Machine Params'):
        LABEL = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'POHM', 'BT', 'ELER', 'P_NBI', 'P_ICRH']

        # correlation = torch.cov(z, val_params, rowVar=False)

        fig, axs = plt.subplots(figsize=(20,20))
        all_cors = []
        for i in range(z.shape[1]):
            val_params[np.isnan(val_params)] = 0
            # print(val_params)
            # print(np.cov(val_params))

            single_dim = z[:, i]
            # print(single_dim.shape)
            correlation = np.cov(single_dim,val_params, rowvar=False) # the first column is the correlation with hidden dim and other params
            correlation = correlation[:, 0][1:]
            # print(correlation)
            all_cors.append(correlation)
        all_cors = np.stack(all_cors)
        all_cors = torch.from_numpy(all_cors)
        im = axs.imshow(all_cors, cmap='viridis')
        axs.set_xticks(np.arange(len(LABEL)))
        axs.set_xticklabels(LABEL)
        axs.set_yticks(np.arange(z.shape[1]))
        plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        plt.title(title)
        return fig


    def plot_latent_for_corr(self, z, val_params):
        fig, axs = plt.subplots(2, 7, figsize=(18, 18), sharey=True, sharex=True)
        axs = axs.ravel()
        for i in range(z.shape[1]-1):
            axs[i].scatter(z[:, i], z[:, i+1], c=val_params[:, 7])
            axs[i].set_ylabel('Z {}'.format(i +1))
            axs[i].set_xlabel('Z {}'.format(i))
            fig.suptitle('Latent Space vs $\Gamma$ - Dim = {}'.format(len(z[0])))
        pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        return optimizer
