from dataset import *
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

EPOCHS = 25

def main(): 
    global EPOCH, datacls, model 
    datacls = ProfileSetModule(batch_size=528)
    datacls.setup()
    model = VAE_LLD(input_dim=2, latent_dim=10, conv_filter_sizes=[8, 10, 12], transfer_hidden_dims=[20, 20, 30, 30, 30, 20])

    all_pulse_sets = datacls.all_pulse_sets
    train_pulse_sets = datacls.train_pulse_sets
    train_iter = datacls.train_dataloader()
    val_iter = datacls.val_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    plotting=False
    
    # plot_latent_space(model, datacls, )
    for EPOCH in range(EPOCHS): 
        for n, batch in enumerate(train_iter): 
            inputs = batch[:, 0, :, :], batch[:, 1, :, :]
            # t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
            results = model.forward(inputs[0], inputs[1])
            # t_0_pred, t_1_pred, t_1_pred_from_trans, (mu_t, log_var_t), (mu_t_1, log_var_t_1), (A_t, o_t) = model.forward(t_0_batch, t_1_batch)
            losses = loss_function(inputs, results)
            loss, recon_loss, kld_loss, physics_loss = losses['loss'], losses['recon'], losses['kld'], losses['physics']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'{EPOCH}: Loss {loss.item()}, Recon t0: {recon_loss.item()}, Unsup: {kld_loss.item()}, Phys: {physics_loss.item()}')
        if EPOCH%10 == 0 and plotting: # and EPOCH>0: 
            with torch.no_grad(): 
                for n, batch in enumerate(val_iter): 
                    if n>3: 
                        break
                    real = batch[:, 0, :, :], batch[:, 1, :, :]
                    t_0_pred, t_1_pred, t_1_pred_from_trans, *_ = model.forward(real[0], real[1])
                    preds = t_0_pred, t_1_pred, t_1_pred_from_trans
                    plot_batch_results(real, preds, datacls)
        scheduler.step()
    save_dict = {'state_dict': model.state_dict(), 
                'datacls': datacls}
    torch.save(save_dict, '/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/model_results/INITIAL_REVISED2.pth')
    
def loss_function(inputs, results):
    def static_pressure_stored_energy_approximation(profs_og):
        boltzmann_constant = 1.380e-23
        profs = torch.clone(profs_og)
        profs = datacls.train_set.denormalize_profiles(profs)
        profs[:, 0, :]*= 1e19
        return boltzmann_constant*torch.prod(profs, 1).sum(1)

    t_0_batch, t_1_batch = inputs 
    t_0_pred, t_1_pred, t_1_pred_from_trans, (mu_t, log_var_t), (mu_t_1, log_var_t_1), (A_t, o_t) = results
    recon_loss_t0 = F.mse_loss(t_0_batch, t_0_pred)
    recon_loss_t1 = F.mse_loss(t_1_batch, t_1_pred_from_trans)
    
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
    recon_loss = recon_loss_t0 + recon_loss_t1
    kld_loss = kld_loss_t + kld_loss_t_1 + kld_loss_t_t_1
    sp_x_t, sp_x_hat_t = static_pressure_stored_energy_approximation(t_0_batch), static_pressure_stored_energy_approximation(t_0_pred)
    sp_t_loss = F.mse_loss(sp_x_t, sp_x_hat_t)
    sp_x_t_1, sp_x_hat_t_1 = static_pressure_stored_energy_approximation(t_1_batch), static_pressure_stored_energy_approximation(t_1_pred_from_trans)
    sp_t_1_loss = F.mse_loss(sp_x_t_1, sp_x_hat_t_1)
    physics_loss = sp_t_loss + sp_t_1_loss 
    # physics_loss = torch.Tensor([0.0])
    loss = 100*recon_loss +  0.0005*kld_loss + physics_loss
    return {'loss': loss, 'recon': recon_loss, 'kld': kld_loss, 
            'recon_0': recon_loss_t0, 'recon_1': recon_loss_t1, 
            'kld_t': kld_loss_t, 'kld_t1': kld_loss_t_1,'kld_tt1': kld_loss_t_t_1, 'physics': physics_loss} 

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

def plot_latent_space(model, datacls): 
    """
    Should plot a whole profile trajectory in latent space, 

    REQS:
        A pulse! Which we should have! 
        
    prof_t = something 
    while t < 150: 
        z_t = model.x2z(initial_prof)
        z_t_1 = model.zt2zt_1(z_t)
        prof_t_1 = model.z2x(z_t_1)
        # cache prof_t_1 
        # cache_z_t, z_t_1
        prof_t = prof_t_1

    """

    t = 0
    test_dl = iter(datacls.test_dataloader())
    batch = next(test_dl)
    results_x, results_y, results_z = [], [], []
    results_profs = []
    t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
    ax = plt.figure().add_subplot(projection='3d')
    results_profs.append(t_0_batch[0])
    for t in range(10): 
        with torch.no_grad(): 
            z_t, *_ = model.x2z(t_0_batch)
            
            z_t_1, *_ = model.zt2zt_1(z_t)
            if t == 0: 
                results_x.append(z_t[0, 0])
                results_x.append(z_t_1[0, 0])
                
                results_y.append(z_t[0, 1])
                results_y.append(z_t_1[0, 1])

                results_z.append(z_t[0, 2])
                results_z.append(z_t_1[0, 2])
            else: 
                results_x.append(z_t_1[0, 0])
                results_y.append(z_t_1[0, 1])
                results_z.append(z_t_1[0, 2])
            prof_t_1 = model.z2x(z_t_1)
            t_0_batch = prof_t_1
            results_profs.append(prof_t_1[0])
        print(z_t[0], z_t_1[0])
        if t == 0: 
            ax.scatter(z_t[0, 0], z_t[0, 1], z_t[0, 2], color='red')
        ax.scatter(z_t[0, 0], z_t[0, 1], z_t[0, 2], color='green')
        ax.scatter(z_t_1[0, 0], z_t_1[0, 1], z_t_1[0, 2], color='orange')
        ax.plot([z_t[0, 0], z_t_1[0, 0]], [z_t[0, 1], z_t_1[0, 1]], [z_t[0, 2], z_t_1[0, 2]], color='dodgerblue')
    ax.plot(results_x, results_y, results_z)
    plt.show()
    prof_array = np.array(results_profs)
    print(prof_array.shape)
def plot_final_animation(prof_array): 
    """ 
    Takes list of profs from previous experiment (latent space)
    and plots them as an animation
    """

    t_0_batch, t_1_batch = datacls.test_set.denormalize_profiles(t_0_batch), datacls.test_set.denormalize_profiles(t_1_batch)
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