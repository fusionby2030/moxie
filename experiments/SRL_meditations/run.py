from dataset import *
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

EPOCHS = 50
global EPOCH, datacls
def main(): 
    datacls = ProfileSetModule(batch_size=124)
    datacls.setup()
    model = VAE_LLD(input_dim=2, latent_dim=5, conv_filter_sizes=[4, 8], transfer_hidden_dims=[10, 20, 20, 20, 20])

    all_pulse_sets = datacls.all_pulse_sets
    train_pulse_sets = datacls.train_pulse_sets
    train_iter = datacls.train_dataloader()
    val_iter = datacls.val_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for EPOCH in range(EPOCHS): 
        for n, batch in enumerate(train_iter): 
            
            
            
            inputs = batch[:, 0, :, :], batch[:, 1, :, :]
            # t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
            results = model.forward(inputs[0], inputs[1])
            # t_0_pred, t_1_pred, t_1_pred_from_trans, (mu_t, log_var_t), (mu_t_1, log_var_t_1), (A_t, o_t) = model.forward(t_0_batch, t_1_batch)
            losses = loss_function(inputs, results)
            loss, recon_loss, kld_loss = losses['loss'], losses['recon'], losses['kld']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'{EPOCH}: Loss {loss.item()}, Recon t0: {recon_loss.item()}, Unsup: {kld_loss.item()}')
        if EPOCH%10 == 0: # and EPOCH>0: 
            with torch.no_grad(): 
                for n, batch in enumerate(val_iter): 
                    if n>3: 
                        break
                    real = batch[:, 0, :, :], batch[:, 1, :, :]
                    t_0_pred, t_1_pred, t_1_pred_from_trans, *_ = model.forward(real[0], real[1])
                    preds = t_0_pred, t_1_pred, t_1_pred_from_trans
                    # plot_batch_results(real, preds, datacls)
                    

def loss_function(inputs, results):
    t_0_batch, t_1_batch = inputs 
    t_0_pred, t_1_pred, t_1_pred_from_trans, (mu_t, log_var_t), (mu_t_1, log_var_t_1), (A_t, o_t) = results
    recon_loss_t0 = F.mse_loss(t_0_batch, t_0_pred)
    recon_loss_t1 = F.mse_loss(t_1_batch, t_1_pred)
    
    kld_loss_t = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mu_t, torch.exp(0.5*log_var_t)),
                torch.distributions.normal.Normal(0, 1)
                ).mean(0).sum()
    kld_loss_t_1 = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mu_t_1, torch.exp(0.5*log_var_t_1)),
                torch.distributions.normal.Normal(0, 1)
                ).mean(0).sum()

    # KL Divergence z_t_1
    
    mean_1 = torch.matmul(A_t, mu_t.unsqueeze(2)).squeeze(2) + o_t
    cov_1 = torch.clip(torch.matmul(A_t, torch.diag(torch.exp(0.5*log_var_t_1))), min=1e-15, max=100000)
    cov_2 = torch.clip(torch.diag(torch.exp(0.5*log_var_t_1)), min=1e-15, max=100000)
    kld_loss_t_t_1 = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mean_1, cov_1),
                torch.distributions.normal.Normal(mu_t_1, cov_2)
                ).mean(0).sum()
    recon_loss = 20*recon_loss_t0 + 20*recon_loss_t1
    kld_loss = 0.0001*kld_loss_t + 0.0001*kld_loss_t_1 + 0.0005*kld_loss_t_t_1
    loss = recon_loss + kld_loss
    return {'loss': loss, 'recon': recon_loss, 'kld': kld_loss, 
            'recon_0': recon_loss_t0, 'recon_1': recon_loss_t1, 
            'kld_t': kld_loss_t, 'kld_t1': kld_loss_t_1,'kld_tt1': kld_loss_t_t_1,} 

def plot_batch_results_old(real, preds): 
    t_0_batch, t_1_batch = real 
    t_0_pred, t_1_pred, t_1_pred_from_trans = preds
    t_0_batch, t_1_batch = datacls.val_set.denormalize_profiles(t_0_batch), datacls.val_set.denormalize_profiles(t_1_batch)
    t_0_pred, t_1_pred, t_1_pred_from_trans = datacls.val_set.denormalize_profiles(t_0_pred), datacls.val_set.denormalize_profiles(t_1_pred), datacls.val_set.denormalize_profiles(t_1_pred_from_trans)
    fig, ax =  plt.subplots(3, 3, sharex=True)
    
    n_ax, t_ax, p_ax = ax[:, 0], ax[:, 1], ax[:, 2]
    for k_ax, k in zip([0, 1, 2], [-3, -2, -1]): 
        n_ax[k_ax].plot(t_0_batch[k, 0, :], color='blue', lw=5, label='x_t')
        n_ax[k_ax].plot(t_1_batch[k, 0, :], color='green', lw=5, label='x_t_1')
        

        t_ax[k_ax].plot(t_0_batch[k, 1, :], color='blue', lw=5)
        t_ax[k_ax].plot(t_1_batch[k, 1, :], color='green', lw=5)

        p_ax[k_ax].plot(t_0_batch[k, 1, :]*t_0_batch[k, 0, :], color='blue', lw=5)
        p_ax[k_ax].plot(t_1_batch[k, 1, :]*t_1_batch[k, 0, :], color='green', lw=5)

        n_ax[k_ax].plot(t_0_pred[k, 0, :], color='dodgerblue', ls='--',lw=5)
        n_ax[k_ax].plot(t_1_pred[k, 0, :], color='olivedrab', ls='--',lw=5)
        n_ax[k_ax].plot(t_1_pred_from_trans[k, 0, :], color='red', ls='--',lw=3)

        t_ax[k_ax].plot(t_0_pred[k, 1, :], color='dodgerblue', ls='--',lw=5, label=r'$\hat{x}_t$')
        t_ax[k_ax].plot(t_1_pred[k, 1, :], color='olivedrab', ls='--',lw=5, label=r'$\hat{x}_{t + 1}$')
        t_ax[k_ax].plot(t_1_pred_from_trans[k, 1, :], color='red', ls='--',lw=3, label=r'$\hat{x}_{zt -> zt + 1}$')

        p_ax[k_ax].plot(t_0_pred[k, 1, :]*t_0_pred[k, 0, :], color='dodgerblue',ls='--', lw=5)
        p_ax[k_ax].plot(t_1_pred[k, 1, :]*t_1_pred[k, 0, :], color='olivedrab',ls='--', lw=5)
        p_ax[k_ax].plot(t_1_pred_from_trans[k, 1, :]*t_1_pred_from_trans[k, 0, :], color='red',ls='--', lw=3)
        
    n_ax[0].legend()
    t_ax[0].legend()
    n_ax[0].set_title('Density')
    t_ax[0].set_title('Temperature')
    p_ax[0].set_title('Pressure')
    fig.suptitle(f'EPOCHS: {EPOCH}')
    
    plt.show()

def plot_latent_space(): 
    """
    Should plot a whole profile trajectory in latent space, 

    REQS:
        A pulse! Which we should have! 
        
    Then do like below 
    """
    pass 

def plot_final_animation(): 
    """ 
    This should make a profile from scratch...
    (maybe also) as well as the latent space movemet 
    Should look like this: 

    prof_t = something 
    while t < 150: 
        z_t = model.x2z(initial_prof)
        z_t_1 = model.zt2zt_1(z_t)
        prof_t_1 = model.z2x(z_t_1)
        # cache prof_t_1 
        # cache_z_t, z_t_1
        prof_t = prof_t_1



    """
    pass 

def plot_batch_results(real, preds, datacls):
    t_0_batch, t_1_batch = real 
    t_0_pred, t_1_pred, t_1_pred_from_trans = preds
    
    t_0_batch, t_1_batch = datacls.val_set.denormalize_profiles(t_0_batch), datacls.val_set.denormalize_profiles(t_1_batch)
    t_0_pred, t_1_pred, t_1_pred_from_trans= datacls.val_set.denormalize_profiles(t_0_pred), datacls.val_set.denormalize_profiles(t_1_pred), datacls.val_set.denormalize_profiles(t_1_pred_from_trans)

    fig, axs = plt.subplots(5, 4, sharey='col')
    n_ax1, n_ax2 = axs[:, 0], axs[:, 2]
    t_ax1, t_ax2 = axs[:, 1], axs[:, 3]
    
    for k_ax, k in zip([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]): 
        n_ax1[k_ax].plot(t_0_batch[k, 0, :], color='blue', lw=5, label='x_t')
        n_ax1[k_ax].plot(t_1_batch[k, 0, :], color='green', lw=5, label='x_t_1')

        n_ax1[k_ax].plot(t_0_pred[k, 0, :], color='dodgerblue', ls='--',lw=5)
        n_ax1[k_ax].plot(t_1_pred[k, 0, :], color='olivedrab', ls='--',lw=5)
        n_ax1[k_ax].plot(t_1_pred_from_trans[k, 0, :], color='red', ls='--',lw=3)

        n_ax2[k_ax].plot(t_0_batch[k*10, 0, :], color='blue', lw=5, label='x_t')
        n_ax2[k_ax].plot(t_1_batch[k*10, 0, :], color='green', lw=5, label='x_t_1')

        n_ax2[k_ax].plot(t_0_pred[k*10, 0, :], color='dodgerblue', ls='--',lw=5)
        n_ax2[k_ax].plot(t_1_pred[k*10, 0, :], color='olivedrab', ls='--',lw=5)
        n_ax2[k_ax].plot(t_1_pred_from_trans[k*10, 0, :], color='red', ls='--',lw=3)

        t_ax1[k_ax].plot(t_0_batch[k, 1, :], color='darkslateblue', lw=5, label='x_t')
        t_ax1[k_ax].plot(t_1_batch[k, 1, :], color='darkmagenta', lw=5, label='x_t_1')

        t_ax1[k_ax].plot(t_0_pred[k, 1, :], color='darkslateblue', ls='--',lw=5)
        t_ax1[k_ax].plot(t_1_pred[k, 1, :], color='pink', ls='--',lw=5)
        t_ax1[k_ax].plot(t_1_pred_from_trans[k, 1, :], color='orange', ls='--',lw=3)

        t_ax2[k_ax].plot(t_0_batch[k*10, 1, :], color='darkslateblue', lw=5, label='x_t')
        t_ax2[k_ax].plot(t_1_batch[k*10, 1, :], color='darkmagenta', lw=5, label='x_t_1')
        t_ax2[k_ax].plot(t_0_pred[k*10, 1, :], color='darkslateblue', ls='--',lw=5)
        t_ax2[k_ax].plot(t_1_pred[k*10, 1, :], color='pink', ls='--',lw=5)
        t_ax2[k_ax].plot(t_1_pred_from_trans[k*10, 1, :], color='orange', ls='--',lw=3)
    plt.show()

if __name__ == '__main__': 
    main()