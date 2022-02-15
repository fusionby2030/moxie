import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import pytorch_lightning as pl 
import torch 
from torch import optim 
import numpy as np

def de_standardize(x, mu, var):
    return (x*var) + mu


class EXAMPLE_DIVA_EXP_AK(pl.LightningModule):
    def __init__(self, model=None, params: dict = {'LR': 0.001}) -> None:
        super(EXAMPLE_DIVA_EXP_AK, self).__init__()

        self.model = model
        self.params = params
        self.current_device = None
        self.learning_rate = params["LR"]


    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_profile, machine_params, masks = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)
        train_loss = self.model.loss_function(**results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.train_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx, mask=masks, D_norms= self.trainer.datamodule.get_density_norms(), T_norms= self.trainer.datamodule.get_temperature_norms())
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
        avg_physics_loss = torch.stack([x['physics_loss'] for x in outputs]).mean()

        metrics = {'Loss/Train': avg_loss,
                    'ReconLoss/Train': avg_recon_loss,
                    'ReconLossMP/Train': avg_recon_loss_mp,
                    'KLD_stoch/Train': avg_KLD_stoch,
                    'KLD_mach/Train': avg_KLD_mach, 
                    'physics/Train': avg_physics_loss}

        self.log_dict(metrics)
        epoch_dictionary = {'loss': avg_loss}


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_profile, machine_params, masks = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)

        val_loss = self.model.loss_function(**results, M_N = self.params['batch_size']/ len(self.trainer.datamodule.val_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx, mask=masks, D_norms= self.trainer.datamodule.get_density_norms(), T_norms= self.trainer.datamodule.get_temperature_norms())
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_recon_loss_mp = torch.stack([x['Reconstruction_Loss_mp'] for x in outputs]).mean()
        avg_KLD_stoch = torch.stack([x['KLD_stoch'] for x in outputs]).mean()
        avg_KLD_mach = torch.stack([x['KLD_mach'] for x in outputs]).mean()

        avg_physics_loss = torch.stack([x['physics_loss'] for x in outputs]).mean()

        metrics = {'Loss/Valid': avg_loss,
                    'ReconLoss/Valid': avg_recon_loss,
                    'ReconLossMP/Valid': avg_recon_loss_mp,
                    'KLD_stoch/Valid': avg_KLD_stoch,
                    'KLD_mach/Valid': avg_KLD_mach, 
                    'physics/Valid': avg_physics_loss}


        self.log_dict(metrics)


    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_profile, machine_params, masks = batch
        self.current_device = real_profile.device

        results = self.forward(real_profile, in_mp=machine_params)
        test_loss = self.model.loss_function(**results, machine_params=machine_params, M_N = self.params['batch_size']/ len(self.trainer.datamodule.test_dataloader()), optimizer_idx=optimizer_idx, batch_idx = batch_idx, mask=masks, D_norms= self.trainer.datamodule.get_density_norms(), T_norms= self.trainer.datamodule.get_temperature_norms())
        # Log the computational Graph!
        # self.logger.experiment.add_graph(self.model, [real_profile, machine_params], use_strict_trace=False)

        return test_loss

    def test_epoch_end(self, outputs):

        self.compare_generate_with_real()
        self.compare_correlations()
        # self.sweep_dimension()
        # self.compare_latent_dims()
        # self.diversity_of_generation()

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
    def compare_generate_with_real(self):
        train_data_iter = iter(self.trainer.datamodule.train_dataloader())
        val_data_iter = iter(self.trainer.datamodule.val_dataloader())
        test_data_iter = iter(self.trainer.datamodule.test_dataloader())

        train_prof_og, train_mp, train_mask = next(train_data_iter)
        val_prof_og, val_mp, val_mask = next(val_data_iter)
        test_prof_og, test_mp, test_mask = next(test_data_iter)
        
        train_results = self.model.forward(train_prof_og, train_mp) # recons, input, mu, logvar
        val_results = self.model.forward(val_prof_og, val_mp) # recons, input, mu, logvar
        test_results = self.model.forward(test_prof_og, test_mp) # recons, input, mu, logvar

        train_loss = self.model.loss_function(**train_results, mask=train_mask)

        val_loss = self.model.loss_function(**val_results, mask=val_mask)
        test_loss = self.model.loss_function(**test_results, mask=test_mask)

        mu_D, var_D = self.trainer.datamodule.get_density_norms()

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

        mu_T, var_T = self.trainer.datamodule.get_temperature_norms()

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

        train_prof, train_mp, _= next(train_data_iter)
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = self.model.q_zy(train_prof)
        z_stoch, z_mach = self.model.reparameterize(mu_stoch, log_var_stoch), self.model.reparameterize(mu_mach, log_var_mach)

        z_mach_all = z_mach
        z_stoch_all = z_stoch
        in_all = train_mp

        for (train_prof, train_mp, _) in train_data_iter:
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
        self.logger.experiment.add_figure('correlation_' + title, fig)
        return fig
