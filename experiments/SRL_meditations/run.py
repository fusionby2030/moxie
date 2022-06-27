# from dataset import *
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

from torch.utils.data import Dataset, DataLoader
import random

from torch_dataset import PULSE_DATASET 
from python_dataset import PULSE, PROFILE, MACHINEPARAMETER, ML_ENTRY 
from tqdm import tqdm 

import pickle 

EPOCHS = 50
save_loc = '/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/model_results/'


def main(model_name='STEP2_aux_cond'): 
    global EPOCH, model, train_set, val_set
    print(save_loc)
    model = VAE_LLD_MP(input_dim=2, latent_dim=20, out_length=75, 
                        conv_filter_sizes=[8, 10, 12], transfer_hidden_dims=[20, 20, 30], 
                        reg_hidden_dims= [40, 40, 40, 40, 40], mp_dim=14, act_dim=14)
    model.double()
    train_dl, val_dl, train_set, val_set = get_train_val_splits(batch_size=256) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
    plotting=False
    
    # plot_latent_space(model, datacls, )
    iter_epochs = tqdm(range(EPOCHS))
    for EPOCH in iter_epochs: 
        if EPOCH % 2 == 0: 
            iter_epochs.set_description('UNSUP')
        else: 
            iter_epochs.set_description('SUP')
        for n, batch in enumerate(train_dl): 
            profs_t0, profs_t1, mps_t0, mps_t1, mps_delta = batch 
            inputs = (profs_t0, profs_t1, mps_t0, mps_t1, mps_delta) 
            results = model.forward(profs_t0, profs_t1, mps_t0, mps_t1, mps_delta)
            # t_0_pred, t_1_pred, t_1_pred_from_trans, (mu_t, log_var_t), (mu_t_1, log_var_t_1), (A_t, o_t) = model.forward(t_0_batch, t_1_batch)
            losses = loss_function(inputs, results)
            loss, recon_loss, kld_loss, recon_prof, recon_mp = losses['loss'], losses['recon'], losses['kld'], losses['recon_prof'], losses['recon_mp']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_epochs.set_postfix_str(f'{n}/{len(train_dl)}: Loss {loss.item():{5}.{5}}, Recon: {recon_loss.item():{5}.{5}}, KLD: {kld_loss.item():{5}.{5}}, PROF_RECON: {recon_prof.item():{5}.{5}}, MPRECON: {recon_mp.item():{5}.{5}}')
        scheduler.step()
    save_dict = {'state_dict': model.state_dict()} #,  
                # 'train_set': train_set, 
                # 'val_set': val_set}
    torch.save(save_dict, save_loc + model_name + '.pth')
    data_dict = {'train_pulses': train_set.pulses, 'val_pulses': val_set.pulses, 'norms': train_set.norms} 
    with open(save_loc + model_name + '_data', 'wb') as file: 
        pickle.dump(data_dict, file)

PERSONAL_DATA_DIR = '/home/kitadam/ENR_Sven/moxie/data/raw/'
PERSONAL_DATA_DIR_PROC = '/home/kitadam/ENR_Sven/moxie/data/processed/'

PICKLED_AUG_PULSES = PERSONAL_DATA_DIR_PROC + 'AUG_PDB_PYTHON_PULSES.pickle'


def loss_function(inputs, results):
    def static_pressure_stored_energy_approximation(profs_og):
        boltzmann_constant = 1.380e-23
        profs = torch.clone(profs_og)
        profs = train_set.denormalize_profiles(profs)
        # profs[:, 0, :]*= 1e19
        return boltzmann_constant*torch.prod(profs, 1).sum(1)

    t_0_batch, t_1_batch, mp_t0, mp_t1, mps_delta = inputs 
    t_0_pred, t_1_pred, t_1_pred_from_trans, mp_hat_t_0, mp_hat_hat_t_1, (mu_t, log_var_t), (mu_t_1, log_var_t_1), (A_t, B_t, o_t), (mu_t_cond_prior, var_t_cond_prior) = results
    recon_loss_t0 = F.mse_loss(t_0_batch, t_0_pred)
    recon_loss_t1 = F.mse_loss(t_1_batch, t_1_pred_from_trans)
    recon_loss_mp_t0 = F.mse_loss(mp_t0, mp_hat_t_0)
    recon_loss_mp_t1 = F.mse_loss(mp_hat_hat_t_1,mp_t1 )
    kld_loss_t = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mu_t, torch.exp(0.5*log_var_t)),
                torch.distributions.normal.Normal(0, 1)
                ).mean(0).sum()
    kld_loss_t_1 = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mu_t_1, torch.exp(0.5*log_var_t_1)),
                torch.distributions.normal.Normal(0, 1)
                ).mean(0).sum()

    kld_loss_mp_cond = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mu_t, torch.exp(0.5*log_var_t)),
                torch.distributions.normal.Normal(mu_t_cond_prior, torch.exp(0.5*var_t_cond_prior))
                ).mean(0).sum()

    # KL Divergence z_t_1
    
    mean_1 = (torch.matmul(A_t, mu_t.unsqueeze(2)) + torch.matmul(B_t, mps_delta.unsqueeze(2))).squeeze(2) + o_t
    cov_1 = torch.clip(torch.matmul(A_t, torch.diag(torch.exp(0.5*log_var_t_1))), min=1e-15, max=100000)
    cov_2 = torch.clip(torch.diag(torch.exp(0.5*log_var_t_1)), min=1e-15, max=100000)
    kld_loss_t_t_1 = torch.distributions.kl.kl_divergence(
                torch.distributions.normal.Normal(mean_1, cov_1),
                torch.distributions.normal.Normal(mu_t_1, cov_2)
                ).mean(0).sum()
    
    # Sum all the losses! 
    recon_loss_prof = recon_loss_t0 + recon_loss_t1
    recon_loss_mp = recon_loss_mp_t1 + recon_loss_mp_t0
    recon_loss = 50*recon_loss_mp + 150*recon_loss_prof
    unsup_loss = kld_loss_t
    sup_loss = kld_loss_mp_cond

    kld_loss = kld_loss_t_1 + kld_loss_t_t_1 # + 0.0001*kld_loss_mp_cond
    if EPOCH % 2 == 0: 
        kld_loss += unsup_loss
    else: 
        kld_loss += sup_loss
    # sp_x_t, sp_x_hat_t = static_pressure_stored_energy_approximation(t_0_batch), static_pressure_stored_energy_approximation(t_0_pred)
    # sp_t_loss = F.mse_loss(sp_x_t, sp_x_hat_t)
    # sp_x_t_1, sp_x_hat_t_1 = static_pressure_stored_energy_approximation(t_1_batch), static_pressure_stored_energy_approximation(t_1_pred_from_trans)
    # sp_t_1_loss = F.mse_loss(sp_x_t_1, sp_x_hat_t_1)
    # physics_loss = sp_t_loss + sp_t_1_loss 
    physics_loss = torch.Tensor([0.0])
    loss = recon_loss +  0.001*kld_loss # + 0.001*physics_loss
    return {'loss': loss, 'recon': recon_loss, 'kld': kld_loss, 
            'recon_prof': recon_loss_prof, 'recon_mp': recon_loss_mp, 
            'recon_prof_0': recon_loss_t0, 'recon_prof_1': recon_loss_t1, 
            'recon_mp_1': recon_loss_mp_t1, 'recon_mp_0': recon_loss_mp_t0, 
            'kld_t': kld_loss_t, 'kld_t1': kld_loss_t_1,'kld_tt1': kld_loss_t_t_1, 'kld_mp_cond': kld_loss_mp_cond,
            'physics': physics_loss} 

# TODO: Move to UTISL 
def load_classes_from_pickle() -> List[PULSE]: 
    with open(PICKLED_AUG_PULSES, 'rb') as file: 
        AUG_PULSES = pickle.load(file)
    return AUG_PULSES   
def get_train_val_splits(batch_size=256): 
    AUG_PULSES = load_classes_from_pickle()
    random.shuffle(AUG_PULSES)

    train_count, val_count = int(0.9*len(AUG_PULSES)), int(0.1*len(AUG_PULSES))
    train_pulses = np.array(AUG_PULSES)[:train_count]
    test_pulses = np.array(AUG_PULSES)[train_count:]
    val_pulses = train_pulses[:val_count]
    train_pulses = train_pulses[val_count:]
    train_set = PULSE_DATASET(pulses=train_pulses)
    norms = train_set.norms
    val_set = PULSE_DATASET(pulses=val_pulses, norms=norms)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=batch_size)

    return train_dl, val_dl, train_set, val_set

if __name__ == '__main__': 
    main()