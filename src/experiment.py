import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

import torch
from torch import optim

import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models.VAE import BaseVAE

class VAExperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAExperiment, self).__init__()

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

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD_loss = torch.stack([x['KLD'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        self.logger.experiment.add_scalar('Loss/Valid', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('ReconLoss/Valid', avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar('KLDLoss/Valid', avg_KLD_loss, self.current_epoch)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, labels = batch
        self.current_device = real_profile.device

        # Log the computational Graph!
        self.logger.experiment.add_graph(self.model, real_profile)

        results = self.forward(real_profile, labels=labels)
        test_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        return test_loss

    def test_epoch_end(self, outputs):
        self.sample_profiles()

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        self.log("hp/final_loss", avg_loss)
        self.log("hp_metric", avg_loss)
        self.log("hp/recon", avg_recon_loss)


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


    def sample_profiles(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))

        data = self.model.forward(test_input)
        avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
        recons, input, mu, logvar = data
        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs = GridSpec(1, 5, figure=fig)
        ax = None
        for i in range(5):
            ax = fig.add_subplot(gs[i], sharey=ax)
            data = [recons[-i], input[-i], mu, logvar]
            avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
            if len(test_input.shape) == 3:
                input = input.squeeze()
                recons= recons.squeeze()
            # print(recons.shape)
            ax.set_title('Recon: {:.4}'.format(avg_loss['Reconstruction_Loss']))
            ax.plot(recons[-i]*self.trainer.datamodule.max_X, label='Generated')
            ax.plot(input[-i]*self.trainer.datamodule.max_X, label='Real')

        plt.legend()
        plt.show()

    def plot_latent_space(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        mu, logvar = self.model.encode(test_input)
        z = self.model.reparameterize(mu, logvar)
        for i in range(len(z[0]) -1):
            fig = plt.figure()
            plt.scatter(z[:, i], z[:, i+1], c=test_label)
        plt.show()
        return z
