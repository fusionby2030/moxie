from load_data import ProfileDataModule
from model import VAE
from datasets import PROFILE 

datacls = ProfileDataModule(batch_size=256)
datacls.setup()
model = VAE()
N_mean, N_var, T_mean, T_var = datacls.return_normalizers()


def de_normalize_profiles(profiles, N_mean=None, N_var=None, T_mean=None, T_var=None): 
        def de_standardize(x, mu, var):
            x_normed = x*var + mu
            return x_normed

        profiles[:, 0] = de_standardize(profiles[:, 0], N_mean, N_var)
        profiles[:, 1] = de_standardize(profiles[:, 1], T_mean, T_var)
        return profiles 

import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import matplotlib.pyplot as plt 

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
for EPOCH in range(10): 
    for batch in datacls.train_dataloader(): 
        _prof, _ = batch 
        
        out_prof, mu, log_var = model.forward(_prof)
        recon_loss = F.mse_loss(_prof, out_prof)
        
        unsup_kld_loss = torch.distributions.kl.kl_divergence(
                    torch.distributions.normal.Normal(mu, torch.exp(0.5*log_var)),
                    torch.distributions.normal.Normal(0, 1)
                    ).mean(0).sum()
        loss = 100*recon_loss + 0.001*unsup_kld_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'{EPOCH}: Loss {loss.item()}')


    with torch.no_grad(): 
        for batch in datacls.val_dataloader(): 
            _prof, _ = batch 
            
            out_prof, _, _ = model.forward(_prof)

            denorm_in = de_normalize_profiles(_prof, N_mean, N_var, T_mean, T_var)
            denorm_out = de_normalize_profiles(out_prof, N_mean, N_var, T_mean, T_var)
            if EPOCH == 0: 
                plt.plot(denorm_in[0, 0, :], color='black', ls='--',)
            
            plt.plot(denorm_out[0, 0, :], label=str(EPOCH))
            break 
plt.legend()
plt.show()

