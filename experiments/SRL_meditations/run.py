from dataset import *
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 


def main(): 
    datacls = ProfileSetModule(batch_size=256)
    datacls.setup()
    model = ForwardVAE()

    all_pulse_sets = datacls.all_pulse_sets
    train_pulse_sets = datacls.train_pulse_sets
    train_iter = datacls.train_dataloader()
    val_iter = datacls.val_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for EPOCH in range(10): 
        for n, batch in enumerate(train_iter): 
            
            
            t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
            t_0_pred, t_1_pred, mu, log_var = model.forward(t_0_batch)

            recon_loss_t0 = F.mse_loss(t_0_batch, t_0_pred)
            recon_loss_t1 = F.mse_loss(t_1_batch, t_1_pred)
            
            unsup_kld_loss = torch.distributions.kl.kl_divergence(
                        torch.distributions.normal.Normal(mu, torch.exp(0.5*log_var)),
                        torch.distributions.normal.Normal(0, 1)
                        ).mean(0).sum()
            recon_loss = 500*recon_loss_t0 # + recon_loss_t1
            loss = recon_loss + 0.0001*unsup_kld_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'{EPOCH}: Loss {loss.item()}, Recon t0: {recon_loss_t0.item()}, Unsup: {unsup_kld_loss.item()}')

        with torch.no_grad(): 
            for n, batch in enumerate(val_iter): 
                t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
                t_0_pred, t_1_pred, mu, log_var = model.forward(t_0_batch)
                plt.plot(t_0_batch[0, 0, :], color='blue', lw=5)
                plt.plot(t_1_batch[0, 0, :], color='green', lw=5)

                plt.plot(t_0_pred[0, 0, :], color='dodgerblue', ls='--', lw=5)
                plt.plot(t_1_pred[0, 0, :], color='forestgreen', ls='--', lw=5)
                plt.show()
                break


if __name__ == '__main__': 
    main()