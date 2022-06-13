
from argparse import ArgumentParser
import sys, os, pathlib
import pickle
from moxie.models.PSI_model_ak1 import PSI_MODEL
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

# Overall Font
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['font.family'] = 'Arial'
from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')

def de_standardize(x, mu, var):
    return (x*var) + mu
def standardize(x, mu, var):
    return (x - mu) / var
def normalize_profiles(profiles, mu_T=None, var_T=None, mu_D=None, var_D=None):
    if mu_D is not None and var_D is not None and mu_T is not None and var_T is not None:
        profiles[:, 0] = standardize_simple(profiles[:, 0], mu_D, var_D)
        profiles[:, 1] = standardize_simple(profiles[:, 1], mu_T, var_T)
        return profiles
    else:
        profiles[:, 0], mu_D, var_D = standardize_simple(profiles[:, 0])
        profiles[:, 1], mu_T, var_T = standardize_simple(profiles[:, 1])
        return profiles, mu_D, var_D, mu_T, var_T
def standardize_simple(x, mu=None, var=None):
    if mu is not None and var is not None:
        x_normed = (x - mu ) / var
        return x_normed
    else:
        mu = x.mean(0, keepdim=True)[0]
        var = x.std(0, keepdim=True)[0]
        x_normed = (x - mu ) / var
        return x_normed, mu, var

def load_data(args, elm_style_choice='simple'):
    def convert_to_tensors(data):
        X, y, mask, ids, elms = data
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        elms, mask = torch.from_numpy(elms).float(), torch.from_numpy(mask) > 0

        if elm_style_choice in ['simple', 'mp_only']:
            if elm_style_choice == 'simple':
                ELM_data = torch.repeat_interleave(elms.unsqueeze(1), 20, 1).unsqueeze(1)
                X = torch.concat((X, ELM_data), 1)
            y = torch.column_stack((y, elms))
        data_tensor = X, y, mask, ids, elms
        return data_tensor
    with open(args.data_path, 'rb') as file:
        full_dict = pickle.load(file)
        # data = profs, mps, mask, ids, elms
        train_data = full_dict['train']['profiles'],  full_dict['train']['machine_parameters'],  full_dict['train']['profiles_mask'], full_dict['train']['timings'], full_dict['train']['elm_fractions']
        val_data =  full_dict['val']['profiles'],  full_dict['val']['machine_parameters'], full_dict['val']['profiles_mask'], full_dict['val']['timings'], full_dict['val']['elm_fractions']
        test_data = full_dict['test']['profiles'],  full_dict['test']['machine_parameters'], full_dict['test']['profiles_mask'], full_dict['test']['timings'], full_dict['test']['elm_fractions']

    train_data_tensor, val_data_tensor, test_data_tensor = convert_to_tensors(train_data), convert_to_tensors(val_data), convert_to_tensors(test_data)
    return (train_data, val_data, test_data), (train_data_tensor, val_data_tensor, test_data_tensor)

def get_latent_space(in_profiles, model):
    with torch.no_grad():
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = model.q_zy(in_profiles)
        z_stoch, z_mach = model.reparameterize(mu_stoch, log_var_stoch), model.reparameterize(mu_mach, log_var_mach)
        z = torch.cat((z_stoch, z_mach), 1)
    return z_mach, z_stoch, z

def reduced_find_separatrix(ne, te, x, plot_result=False):
    # This just moves until we find te < 100
    idx_r, idx_l = 1, 0
    while te[idx_r] > 100:
        idx_l, idx_r = idx_l +1, idx_r +1
    weights_r, weights_l = get_weights(te, idx_l, idx_r)
    tesep_estimation = weights_l*te[idx_l] + weights_r*te[idx_r]
    nesep_estimation = weights_l*ne[idx_l] + weights_r*ne[idx_r]
    rsep_estimation = weights_l*x[idx_l] + weights_r*x[idx_r]
    return tesep_estimation, nesep_estimation, rsep_estimation

def get_weights(te, idx_l, idx_r, query=100):
    # Gets weighst as usual
    dist = te[idx_r] - query + query - te[idx_l]
    weights = (1-(te[idx_r] - query)/dist, 1-(query - te[idx_l])/dist)
    return weights

def get_nesep_for_samples(sample_profs):
    # For all the genearted profiles, calculate the neseps and reseps
    sample_rmids = np.tile(np.linspace(3.75, 3.95, 20), (len(sample_profs), 1))
    sample_rseps_pred = np.zeros(len(sample_rmids))
    sample_neseps_pred = np.zeros(len(sample_rmids))
    for n, (profs, rmids) in enumerate(zip(sample_profs, sample_rmids)):
        ne, te, x = profs[0].numpy(), profs[1].numpy(), rmids
        logical_mask = np.logical_and(te > 0, ne > 0)
        ne, te, x = ne[logical_mask], te[logical_mask], x[logical_mask]
        try:
            estimations = reduced_find_separatrix(ne, te, x)
        except Exception as e:
            continue
        else:
            sample_neseps_pred[n] = estimations[1]
            sample_rseps_pred[n] = estimations[2]
    return sample_neseps_pred, sample_rseps_pred

def get_preds_from_conditional(mps, profs):
    with torch.no_grad():
        cond_mu_sample, cond_var_sample =  model.p_zmachx(mps)
        mu_stoch_sample, log_var_stoch_sample, mu_mach_sample, log_var_mach_sample = model.q_zy(profs)

        z_stoch_sample =  mu_stoch_sample# model.reparameterize(mu_stoch_sample, log_var_stoch_sample)
        z_mach_sample = cond_mu_sample# model.reparameterize(cond_mu_sample, cond_var_sample)

        z_conditional_sample = torch.cat((z_stoch_sample, z_mach_sample), 1)
        out_profs_cond_sample = model.p_yhatz(z_conditional_sample)
    out_profs_cond_sample[:, 0] = de_standardize(out_profs_cond_sample[:, 0], D_mu, D_var)
    out_profs_cond_sample[:, 1] = de_standardize(out_profs_cond_sample[:, 1], T_mu, T_var)

    return out_profs_cond_sample

def get_preds_from_latent_space(z_mach, z_stoch):
    with torch.no_grad():
        z_conditional = torch.cat((z_stoch, z_mach), 1)
        out_profs = model.p_yhatz(z_conditional)
        out_mps = model.q_hatxzmach(z_mach)

    out_profs[:, 0] = de_standardize(out_profs[:, 0], D_mu, D_var)
    out_profs[:, 1] = de_standardize(out_profs[:, 1], T_mu, T_var)
    out_mps[:, :-1] = de_standardize(out_mps[:, :-1], MP_mu, MP_var)
    return out_profs, out_mps

def repeated_sample_to_get_uncerts(z_mach, z_stoch, sample_1=-1, sample_size=20):
    sample_1_z_mach, sample_1_z_stoch = torch.tile(z_mach[sample_1], (sample_size, 1)), torch.tile(z_stoch, (sample_size, 1)) + torch.normal(0, 1, (sample_size, model_hyperparams['stoch_latent_dim']))
    sample_1_profs, sample_1_mps = get_preds_from_latent_space(sample_1_z_mach, sample_1_z_stoch)
    sample_1_neseps, sample_1_rseps = get_nesep_for_samples(sample_1_profs)
    keep_ids = sample_1_neseps != 0 # catch failures and leave them alone
    sample_1_profs, sample_1_mps, sample_1_neseps, sample_1_rseps = sample_1_profs[keep_ids], sample_1_mps[keep_ids], sample_1_neseps[keep_ids], sample_1_rseps[keep_ids]

    return  (torch.mean(sample_1_profs, 0), torch.std(sample_1_profs, 0)), (torch.mean(sample_1_mps, 0), torch.std(sample_1_mps, 0)), (sample_1_neseps.mean(), sample_1_neseps.std()), (sample_1_rseps, sample_1_rseps.std())
def plot_conditional_generate(data):
    profs, mps, mask, ids, elms = data

    image_res = 512
    sample_size = image_res ** 2 # 2D
    r1_c, r2_c = -4e6, 0
    a, b = sample_size, 2
    range_current = torch.linspace(start=r1_c, end=r2_c, steps=image_res)
    r1, r2 = 0, 1
    range_elm = torch.linspace(start=r1, end=r2, steps=image_res)

    range_xy = torch.cartesian_prod(range_current, range_elm)
    range_imagecoord = torch.linspace(0, image_res-1, steps=image_res, dtype=torch.int32)  # so we can easily go back
    range_imagecoord = torch.cartesian_prod(range_imagecoord, range_imagecoord)

    conditional_mps =  torch.tile(mps.mean(0), (sample_size, 1))
    conditional_mps[:, 8] = range_xy[:, 0]
    conditional_mps[:, -1] = range_xy[:, 1]

    profs = torch.tile(profs.mean(0), (sample_size, 1, 1))
    mps_norm = torch.clone(conditional_mps)
    mps_norm[:, :-1] = standardize(mps_norm[:, :-1], MP_mu, MP_var)
    profs_norm = torch.clone(profs)
    profs_norm = normalize_profiles(profs_norm,  T_mu, T_var, D_mu, D_var)
    out_profs_cond_sample = get_preds_from_conditional(mps_norm, profs_norm)

    inferred_y = out_profs_cond_sample[:, 0, 0]
    image_array = np.zeros((image_res, image_res))

    for i in range(range_imagecoord.shape[0]):
        _x, _y = range_imagecoord[i]
        _y = image_res - 1 - _y  # (0, 0) for img are on top left so reverse
        image_array[_y, _x] = inferred_y[i]
    cmap = mpl.cm.viridis
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=0, vmax=1e20)
    centi = 1/2.54
    #cmap = mpl.cm.viridis

    fig, ls_ax = plt.subplots(1, 1, figsize=(40*centi, 20*centi), constrained_layout=True)

    """ LATENT SPACE PLOT """
    cax = ls_ax.imshow(image_array,  cmap=cmap, norm=norm, interpolation='spline36', extent=[r1, r2, r1, r2]) #

    fig.colorbar(cax, ax=ls_ax, label='Inferred $n_{e, ped}$', location='left')
    ls_ax.set_xlabel('Conditional $I_P$ ')
    ls_ax.set_ylabel('Conditional ELM % ')
    x_label_list = np.linspace(-4, 0, 5)
    y_label_list = np.linspace(0, 1, 5)
    ls_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ls_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ls_ax.set_xticklabels(x_label_list)
    ls_ax.set_yticklabels(y_label_list)


    plt.show()
def plot_latent_space_nesep(data):
    # Latent space plot
    profs, mps, mask, ids, elms = data
    mps_norm, profs_norm = torch.clone(mps), torch.clone(profs)
    mps_norm[:, :-1] = standardize(mps_norm[:, :-1], MP_mu, MP_var)
    profs_norm = normalize_profiles(profs_norm,  T_mu, T_var, D_mu, D_var)
    # get mean of latent space for interpolation
    Z_MACH, Z_STOCH, Z = get_latent_space(profs_norm, model)

    image_res = 512
    sample_size = image_res ** 2 # 2D
    r1, r2 = -10, 10
    a, b = sample_size, 2
    ld_1, ld_2 = 1, 5

    z_mach_mean, z_stoch_mean = Z_MACH.mean(0), Z_STOCH.mean(0)
    z_mach_sample, z_stoch_sample = torch.tile(z_mach_mean, (sample_size, 1)), torch.tile(z_stoch_mean, (sample_size, 1))

    range_xy = torch.linspace(start=r1, end=r2, steps=image_res)
    range_xy = torch.cartesian_prod(range_xy, range_xy)
    range_imagecoord = torch.linspace(0, image_res-1, steps=image_res, dtype=torch.int32)  # so we can easily go back
    range_imagecoord = torch.cartesian_prod(range_imagecoord, range_imagecoord)

    z_mach_sample[:, ld_1] = range_xy[:, 0]
    z_mach_sample[:, ld_2] = range_xy[:, 1]

    sample_profs, sample_mps = get_preds_from_latent_space(z_mach_sample, z_stoch_sample)
    sample_neseps, sample_rseps = get_nesep_for_samples(sample_profs)

    x = np.linspace(3.75, 3.95, 20)

    image_array = np.zeros((image_res, image_res))

    for i in range(range_imagecoord.shape[0]):
        _x, _y = range_imagecoord[i]
        _y = image_res - 1 - _y  # (0, 0) for img are on top left so reverse
        image_array[_y, _x] = sample_neseps[i]

    sample_z4 = 2
    sample_z6 = 0
    data_sample = torch.tensor([sample_z4, sample_z6])
    min_i = -1
    min_dist = 100000
    for i in range(range_xy.shape[0]):
        a = data_sample.cpu().numpy()
        b = range_xy[i].cpu().numpy()
        dist = np.linalg.norm(a-b)
        if dist < min_dist:
            min_dist = dist
            min_i = i
    sample_1 = min_i
    sample_1_results = repeated_sample_to_get_uncerts(z_mach_sample, z_stoch_mean, sample_1)
    sample_1_profs, sample_1_mps, sample_1_neseps, sample_1_rseps = sample_1_results

    sample_z4 = 9
    sample_z6 = -9
    data_sample = torch.tensor([sample_z4, sample_z6])
    min_i = -1
    min_dist = 100000
    for i in range(range_xy.shape[0]):
        a = data_sample.cpu().numpy()
        b = range_xy[i].cpu().numpy()
        dist = np.linalg.norm(a-b)
        if dist < min_dist:
            min_dist = dist
            min_i = i
    sample_2 = min_i
    sample_2_results = repeated_sample_to_get_uncerts(z_mach_sample, z_stoch_mean, sample_2)
    sample_2_profs, sample_2_mps, sample_2_neseps, sample_2_rseps = sample_2_results

    rsep_approx = sample_1_rseps[0].mean()
    resep_error = sample_1_rseps[0].std()
    rsep_approx_2 = sample_2_rseps[0].mean()
    resep_error_2 = sample_2_rseps[0].std()

    cmap = mpl.cm.viridis
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=0, vmax=1e20)
    centi = 1/2.54
    #cmap = mpl.cm.viridis

    fig, (ls_ax, t_ax) = plt.subplots(1, 2, figsize=(40*centi, 20*centi), constrained_layout=True)

    """ LATENT SPACE PLOT """
    cax = ls_ax.imshow(image_array, extent=[r1, r2, r1, r2], cmap=cmap, norm=norm, interpolation='spline36')
    fig.colorbar(cax, ax=ls_ax, label='Inferred $n_{e, sep}$ %', location='left')
    ls_ax.scatter(z_mach_sample[:, ld_1][sample_1], z_mach_sample[:, ld_2][sample_1], c=sample_neseps[sample_1], edgecolor='black', cmap=cmap, norm=norm, s=500, marker='*', linewidths=3)
    ls_ax.scatter(z_mach_sample[:, ld_1][sample_2], z_mach_sample[:, ld_2][sample_2], c=sample_neseps[sample_2], edgecolor='black', cmap=cmap, norm=norm, s=500, marker='*', linewidths=3)
    ls_ax.set_xlabel('Latent Dimension ' + str(ld_1))
    ls_ax.set_ylabel('Latent Dimension ' + str(ld_2))


    """ PROFILES PLOT """
    n_ax = t_ax.twinx()

    n_ax.errorbar(x - rsep_approx_2, sample_2_profs[0][0]*1e-19, yerr=sample_2_profs[1][0]*1e-19, alpha=0.8, fmt='none', color='cadetblue')
    ne = n_ax.plot(x - rsep_approx_2, sample_2_profs[0][0]*1e-19, color='royalblue', lw=3)
    n_ax.scatter(x - rsep_approx_2, sample_2_profs[0][0]*1e-19, color='black', s=75)
    n_ax.set_ylim(0, 10)
    n_ax.set_ylabel('Generated $n_e$ [ $10^{19}$ m$^{-3}$ ]')
    n_ax.axvline(0, color='black', lw=1, zorder=10)

    n_ax.errorbar(0, sample_2_neseps[0]*1e-19, yerr=sample_2_neseps[1]*1e-19,  elinewidth=4, capsize=10.0, ls='--', ecolor='red', ms=25, color=cmap(norm(sample_2_neseps[0])), fmt='*', markeredgecolor='black', zorder=20)

    # plt.plot(sample_2_profs[0][1], color=cmap(norm(sample_1_neseps[0])), ls='--')
    t_ax.errorbar(x - rsep_approx_2, sample_2_profs[0][1], yerr=sample_2_profs[1][1], alpha=0.8,fmt='none',  color='tan')
    te = t_ax.plot(x - rsep_approx_2, sample_2_profs[0][1], color='orange', ls='--', lw=3)
    t_ax.scatter(x - rsep_approx_2, sample_2_profs[0][1], color='black', s=75, )

    """
    n_ax.errorbar(x - rsep_approx, sample_1_profs[0][0]*1e-19, yerr=sample_1_profs[1][0]*1e-19, alpha=0.8, fmt='none', color='cadetblue')
    ne = n_ax.plot(x - rsep_approx, sample_1_profs[0][0]*1e-19, color='royalblue', lw=3)
    n_ax.scatter(x - rsep_approx, sample_1_profs[0][0]*1e-19, color='black', s=75)
    n_ax.set_ylim(0, 8)
    n_ax.set_ylabel('Generated $n_e$ [ $10^{19}$ m$^{-3}$ ]')
    n_ax.axvline(0, color='black', lw=1, zorder=10)

    n_ax.errorbar(0, sample_1_neseps[0]*1e-19, yerr=sample_1_neseps[1]*1e-19,  elinewidth=4, capsize=10.0, ls='--', ecolor='red', ms=25, color=cmap(norm(sample_1_neseps[0])), fmt='*', markeredgecolor='black', zorder=20)

    # plt.plot(sample_1_profs[0][1], color=cmap(norm(sample_1_neseps[0])), ls='--')
    t_ax.errorbar(x - rsep_approx, sample_1_profs[0][1], yerr=sample_1_profs[1][1], alpha=0.8,fmt='none',  color='tan')
    te = t_ax.plot(x - rsep_approx, sample_1_profs[0][1], color='orange', ls='--', lw=3)
    t_ax.scatter(x - rsep_approx, sample_1_profs[0][1], color='black', s=75, )
    """
    t_ax.axhline(100, color='black', ls='--', lw=3)
    t_ax.annotate('$T_{e, sep} = 100$ [eV]', xy=(0.1, 0.1), xycoords='axes fraction')
    t_ax.set_ylim(0, 1500)
    n_ax.set_facecolor('gainsboro')

    t_ax.set_xlabel(r'$R_{mid} - R_{mid, sep}$ [m]')
    t_ax.set_ylabel('Generated $T_e$ [ eV ]')
    fig.suptitle('Generating profiles via representation learning')
    plt.show()
def main(args):
    global MP_mu, MP_var, D_mu, D_var, T_mu, T_var, model, model_hyperparams
    exp_dict = torch.load(f'./model_results/modelstatedict_{args.name}.pth')
    state_dict, (MP_mu, MP_var), (D_mu, D_var), (T_mu, T_var), model_hyperparams = exp_dict.values()
    model = PSI_MODEL(**model_hyperparams)
    model.load_state_dict(state_dict)

    (train_data, val_data, test_data), (train_data_tensor, val_data_tensor, test_data_tensor) = load_data(args, model_hyperparams['elm_style_choice'])

    # plot_conditional_generte(train_data_tensor)
    plot_latent_space_nesep(train_data_tensor)
if __name__ == '__main__':
    file_path = pathlib.Path(__file__).resolve()# .parent.parent
    # Path of experiment
    exp_path = file_path.parent
    # Path of moxie stuffs
    home_path = file_path.parent.parent.parent
    # Path to data
    dataset_path = home_path / 'data' / 'processed' / f'ML_READY_dict.pickle'# f'ML_READY_dict_{CURRENT_DATE}.pickle'
    parser = ArgumentParser()
    parser.add_argument("--name", '-n',  type=str, default=f'EXAMPLE_{CURRENT_DATE}')
    parser.add_argument('--data_path', type=str, default=dataset_path)
    args = parser.parse_args()

    main(args)
