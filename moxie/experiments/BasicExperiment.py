import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
Tensor = TypeVar('torch.tensor')

import torch
from torch import optim

import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


class BasicExperiment(pl.LightningModule):

    def __init__(self, vae_model: None, params: dict) -> None:
        super(BasicExperiment, self).__init__()

        """
        Needs params:
            LR
            batch_size
            weight_decay

        """

        self.model = vae_model
        self.params = params
        self.current_device = None
        self.learning_rate = params['LR']

        self.save_hyperparameters(params)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def on_train_start(self):
        if self.current_epoch == 1:
            self.logger.log_hyperparams(self.hparams, {"hp/final_loss": 0, "hp/recon": 0})

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)


        train_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.train_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        self.log('Loss/train', train_loss['loss'])
        self.log('ReconLoss/train', train_loss['Reconstruction_Loss'])
        self.log('KLD/train', train_loss['KLD'])
        logs = {'train_loss': train_loss}

        batch_dictionary = {'loss': train_loss, 'log': logs}
        return train_loss


    def training_epoch_end(self, outputs):
        # Outputs is whatever that is returned from training_step

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD_loss = torch.stack([x['KLD'] for x in outputs]).mean()

        # self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar('ReconLoss/Train', avg_recon_loss, self.current_epoch)
        # self.logger.experiment.add_scalar('KLDLoss/Train', avg_KLD_loss, self.current_epoch)

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

        # self.logger.experiment.add_scalar('Loss/Valid', avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar('ReconLoss/Valid', avg_recon_loss, self.current_epoch)
        # self.logger.experiment.add_scalar('KLDLoss/Valid', avg_KLD_loss, self.current_epoch)

        self.log("Loss/valid", avg_loss)
        self.log("ReconLoss/valid", avg_loss)
        self.log("KLD/valid", avg_KLD_loss)
        # self.logger.experiment.add_scalar("weighting_factor", self.model.beta * self.model.num_iter / (self.model.gamma), self.current_epoch)

    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)
        test_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        # Log the computational Graph!

        # if self.logger._log_graph:
        #     self.logger.experiment.add_graph(self.model, real_profile)


        return test_loss

    def test_epoch_end(self, outputs):

        self.generate_samples_compare_with_mean()
        self.compare_generate_with_real()
        self.plot_corr_matrix()

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

    def compare_generate_with_real(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof, train_mp = next(train_data_iter)
        val_prof, val_mp = next(val_data_iter)
        test_prof, test_mp = next(test_data_iter)

        train_results = self.model.forward(train_prof) # recons, input, mu, logvar
        val_results = self.model.forward(val_prof) # recons, input, mu, logvar
        test_results = self.model.forward(test_prof) # recons, input, mu, logvar

        train_res = train_results[0]
        val_res = val_results[0]
        test_res = test_results[0]

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
        fig.suptitle('{}-D Latent Space'.format(self.model.latent_dim))
        plt.setp(axs, xticks=[])
        self.logger.experiment.add_figure('comparison', fig)
        plt.show()


    def generate_samples_compare_with_mean(self):
        train_data = self.trainer.datamodule.X_train
        mean_profiles = torch.mean(train_data, 0)

        sampled_profiles = self.model.sample(10000, self.current_device)
        mean_sampled = torch.mean(sampled_profiles, 0)

        max_sample,  min_sample = torch.max(sampled_profiles, 0), torch.min(sampled_profiles, 0)
        max_sample, _ = max_sample
        min_sample, _ = min_sample

        fig, axs = plt.subplots(1, 1, figsize=(18, 18), constrained_layout=True)
        axs.plot(mean_sampled.squeeze(), label='VAE Mean', lw=3)
        axs.plot(mean_profiles, label='Train Mean', lw=3)
        axs.fill_between(np.arange(0, 63), max_sample.squeeze(), min_sample.squeeze(), color='grey', alpha=0.3)

        axs.legend()
        fig.supxlabel('R', size='xx-large')
        fig.supylabel('$n_e \; \; (10^{20}$ m$^{-3})$', size='xx-large')
        fig.suptitle('{}-D Latent Space'.format(self.model.latent_dim))
        plt.setp(axs, xticks=[], ylim=(-0.05, 1.0))

        self.logger.experiment.add_figure('samples_vs_mean', fig)

        plt.show()

    def plot_latent_for_corr(self, z, val_params):
        fig, axs = plt.subplots(2, 5, figsize=(18, 18), sharey=True, sharex=True)
        axs = axs.ravel()
        for i in range(z.shape[1]-1):
            axs[i].scatter(z[:, i], z[:, i+1], c=val_params[:, 7])
            axs[i].set_ylabel('Z {}'.format(i +1))
            axs[i].set_xlabel('Z {}'.format(i))
            fig.suptitle('Latent Space vs $\Gamma$ - Dim = {}'.format(len(z[0])))
        pass

    def plot_corr_matrix(self, title='Latent Space vs Machine Params'):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        train_prof, val_params = next(train_data_iter)
        mu_mach, log_var_mach = self.model.encode(train_prof)
        z = self.model.reparameterize( mu_mach, log_var_mach)

        LABEL = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'POHM', 'BT', 'ELER', 'P_NBI', 'P_ICRH']

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
        fig.colorbar(im, orientation="horizontal", pad=0.2)
        self.logger.experiment.add_figure('correlation_mparams', fig)
        plt.show()
        return fig

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
            # val_params = normalize(val_params)
            print(val_params)
            mu, logvar = self.model.encode(val_input)
            z = self.model.reparameterize(mu, logvar)
            # self.plot_latent_for_corr(z, val_params)
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
