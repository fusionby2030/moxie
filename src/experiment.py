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


    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)

        train_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.train_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        self.logger.log_metrics({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)

        val_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.val_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        real_profile, labels = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, labels=labels)
        test_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx)

        return test_loss

    def test_epoch_end(self, outputs):
        self.sample_profiles()

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.logger.log_metrics(tensorboard_logs)
        # print(avg_loss)
        # lsp = self.plot_latent_space()
        return tensorboard_logs



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
            data = [recons[i], input[i], mu, logvar]
            avg_loss = self.model.loss_function(*data, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()))
            if len(test_input.shape) == 3:
                input = input.squeeze()
                recons= recons.squeeze()
            # print(recons.shape)
            ax.set_title('Recon: {:.4}'.format(avg_loss['Reconstruction_Loss']))
            ax.plot(recons[i]*self.trainer.datamodule.max_X, label='Generated')
            ax.plot(input[i]*self.trainer.datamodule.max_X, label='Real')

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

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
