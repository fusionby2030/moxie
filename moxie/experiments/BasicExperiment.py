import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
Tensor = TypeVar('torch.tensor')

import torch
from torch import optim

import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# from ..models.VAE import BaseVAE



class BasicExperiment(pl.LightningModule):

    def __init__(self, vae_model: None, params: dict) -> None:
        super(BasicExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.current_device = None
        self.learning_rate = params['LR']
        self.save_hyperparameters(params)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)


        train_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.train_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

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
        avg_KLD_loss = torch.stack([x['KLD'] for x in outputs]).mean()

        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLoss/Train', avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar('KLDLoss/Train', avg_KLD_loss, self.current_epoch)

        self.custom_histogram()

        epoch_dictionary = {'loss': avg_loss}


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)

        val_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.val_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD_loss = torch.stack([x['KLD'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        self.logger.experiment.add_scalar('Loss/Valid', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLoss/Valid', avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar('KLDLoss/Valid', avg_KLD_loss, self.current_epoch)

        self.log("hp/final_loss", avg_loss, on_epoch=True)
        self.log("hp_metric", avg_loss, on_epoch=True)
        self.log("hp/recon", avg_recon_loss, on_epoch=True)
        self.logger.experiment.add_scalar("weighting_factor", self.model.beta * self.model.num_iter / (self.model.gamma), self.current_epoch)

    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)
        # generated_profiles = results[0]
        test_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)


        # while self.plotting_stay:
        #     profile_plot_params = {'title': 'Reconstruction: {}-Hidden Layers {}-D Latent Space'.format(len(self.model.hidden_dims), self.model.latent_dim), 'ylabel': '$n_e (m^{-3})$', 'xlabel': 'R (m)', 'ylim': (-0.1, self.trainer.datamodule.max_X)}
        #     self.plotting_stay = plot_sample_profiles_from_batch(results, plot_params=profile_plot_params)

        # Log the computational Graph!
        if self.logger._log_graph:
            self.logger.experiment.add_graph(self.model, real_profile)


        return test_loss

    def test_epoch_end(self, outputs):
        # self.plot_per_latent_dim()
        # self.sample_single_profile()

        # self.plot_latent_space()
        self.correlation_of_latent_space()
        self.sample_profiles()
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


    def custom_histogram(self):
        for name, params in self.named_parameters():
            name_list = name.split('.')[1:]
            if len(name_list) >= 3:
                if name_list[2] == '0':
                    name_list[2] = 'Conv'
                else:
                    name_list[2] = 'BN'
            logger_name = '/'.join(name_list)
            self.logger.experiment.add_histogram(logger_name, params, self.current_epoch)

    def plot_latent_for_corr(self, z, val_params):
        fig, axs = plt.subplots(2, 5, figsize=(18, 18), sharey=True, sharex=True)
        axs = axs.ravel()
        for i in range(z.shape[1]-1):
            axs[i].scatter(z[:, i], z[:, i+1], c=val_params[:, 7])
            axs[i].set_ylabel('Z {}'.format(i +1))
            axs[i].set_xlabel('Z {}'.format(i))
            fig.suptitle('Latent Space vs $\Gamma$ - Dim = {}'.format(len(z[0])))
        pass

    def correlation_of_latent_space(self):
        LABEL = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'POHM', 'BT', 'ELER', 'P_NBI', 'P_ICRH', 'NE']
        validation_data_iter = iter(self.trainer.datamodule.val_dataloader())
        val_input, val_params = next(validation_data_iter)
        val_input, val_params = next(validation_data_iter)
        val_input, val_params = next(validation_data_iter)
        val_input, val_params = next(validation_data_iter)
        for k in range(2):
            val_input, val_params = next(validation_data_iter)
            # val_params[:, -4] = val_params[:, -4]*(1E-21)
            val_params = normalize(val_params)
            mu, logvar = self.model.encode(val_input)
            z = self.model.reparameterize(mu, logvar)
            self.plot_latent_for_corr(z, val_params)
            # print(z.shape, print(val_params.shape))

            # For each latent space, we want the correlation between it and the
            all_cors = []
            fig, axs = plt.subplots(figsize=(20,20))
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

            im = axs.imshow(all_cors, cmap='viridis')
            axs.set_xticks(np.arange(len(LABEL)))
            axs.set_xticklabels(LABEL)
            axs.set_yticks(np.arange(z.shape[1]))
            plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
            fig.colorbar(im)
            plt.show()


    def sample_single_profile(self):
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())
        for k in range(2):
            test_input, test_label = next(test_data_iter)
            data = self.model.forward(test_input)
            avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
            recons, input, mu, logvar = data
            fig = plt.figure(figsize=(18, 18), constrained_layout=True)
            # gs = GridSpec(1, 5, figure=fig)
            input = input.squeeze()
            recons= recons.squeeze()
            plt.plot(recons[0]*self.trainer.datamodule.max_X, label='Generated', lw=4)
            plt.plot(input[0]*self.trainer.datamodule.max_X, label='Real', lw=4)
            plt.xticks([])
            # plt.title()
            plt.legend()
            plt.show()

    def sample_profiles(self):
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())
        fig = plt.figure(figsize=(18, 18), constrained_layout=True)
        gs = GridSpec(4, 3, figure=fig)
        for k in range(4):
            test_input, test_label = next(test_data_iter)
            data = self.model.forward(test_input)
            avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
            recons, input, mu, logvar = data


            ax = None
            for i in range(3):
                ax = fig.add_subplot(gs[k, i], sharey=ax, sharex=ax)
                data = [recons[i], input[i], mu, logvar]
                avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
                if len(test_input.shape) == 3:
                    input = input.squeeze()
                    recons= recons.squeeze()
                ax.set(ylabel='$n_e$')
                ax.plot(recons[-i]*self.trainer.datamodule.max_X, label='Generated')
                ax.plot(input[-i]*self.trainer.datamodule.max_X, label='Real')
                ax.set_xticks([])

            fig.suptitle('Reconstruction: {}-Hidden Layers {}-D Latent Space'.format(len(self.model.hidden_dims), self.model.latent_dim))
        plt.legend()
        plt.show()

    def plot_latent_space(self):
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())
        fig, axs = plt.subplots(1, self.model.latent_dim -1, figsize=(18, 18), constrained_layout=True)
        # gs = GridSpec(1, self.model.latent_dim - 1, figure=fig)
        for k in range(len(test_data_iter)):
            test_input, test_label = next(test_data_iter)
            mu, logvar = self.model.encode(test_input)
            z = self.model.reparameterize(mu, logvar)
            # print(z)
            for i in range(len(z[0]) -1):
                axs[i].scatter(z[:, i], z[:, i+1], c=test_label[:, -1])
                axs[i].set(ylabel='Z {}'.format(i +1), xlabel='Z {}'.format(i))
        fig.suptitle('Latent Space vs Nesep - Dim = {}'.format(len(z[0])))
        plt.show()

    def plot_per_latent_dim(self):
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())
        # fig, axs = plt.subplots(1, self.model.latent_dim -1, figsize=(18, 18), constrained_layout=True)
        # gs = GridSpec(1, self.model.latent_dim - 1, figure=fig)
        #

        fig = plt.figure(figsize=(18, 18))
        # print(z)
        i = 0
        for k in range(len(test_data_iter)):
            test_input, test_label = next(test_data_iter)
            mu, logvar = self.model.encode(test_input)
            z = self.model.reparameterize(mu, logvar)
            plt.scatter(z[:, 0], z[:, 1], c=test_label[:, -1])
            plt.ylabel('Z {}'.format(i +1))
            plt.xlabel('Z {}'.format(i))
        fig.suptitle('Latent Space vs Nesep - Dim = {}'.format(len(z[0])))
        plt.show()
