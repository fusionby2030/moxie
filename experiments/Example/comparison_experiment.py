"""
A script to compare a bunch of similar pulses.


1.) Plot how the profiles look in reality.
Using a trained model>
2.)

"""

from moxie.models.DIVA_ak_1 import DIVAMODEL

import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# GLOBAL PARAMS

# MACHINE PARAM ORDER IN Y SETS
machine_param_order = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'ELON', 'POHM', 'IPLA', 'BVAC', 'NBI', 'ICRH', 'ELER']

# JET PEDESTAL DATABASE COLUMNS
relevant_columns = ['shot', 'nepedheight10^19(m^-3)','Tepedheight(keV)', 'B(T)','Ip(MA)', 'q95',  'R(m)', 'a(m)','gasflowrateofmainspecies10^22(e/s)',  'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'P_ICRH(MW)', 'P_NBI(MW)','plasmavolume(m^3)', 'averagetriangularity', 'divertorconfiguration', 'FLAG:Seeding', 'FLAG:Kicks', 'FLAG:RMP', 'FLAG:pellets',]

"""
Helper Functions
"""

def de_standardize(x, mu, var):
    return (x*var) + mu

def standardize(x, mu, var):
    return (x - mu) / var

def get_latent_space(in_profiles, model):
    with torch.no_grad():
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = model.q_zy(in_profiles)
        z_stoch, z_mach = model.reparameterize(mu_stoch, log_var_stoch), model.reparameterize(mu_mach, log_var_mach)
        z = torch.cat((z_stoch, z_mach), 1)
    return z_mach, z_stoch, z

def get_profile_data(data_path=''):
    if not data_path:
        data_path = '/home/adam/ENR_Sven/moxie/data/processed/pedestal_profiles_ML_READY_ak_09022022.pickle'

    with open(data_path, 'rb') as file:
        full_dict = pickle.load(file)
        train_X, train_y, train_mask, train_radii, train_ids = full_dict['train_dict']['padded']['profiles'],full_dict['train_dict']['padded']['controls'], full_dict['train_dict']['padded']['masks'], full_dict['train_dict']['padded']['radii'] , full_dict['train_dict']['padded']['pulse_time_ids']
        val_X, val_y, val_mask, val_radii, val_ids = full_dict['val_dict']['padded']['profiles'],full_dict['val_dict']['padded']['controls'], full_dict['val_dict']['padded']['masks'], full_dict['val_dict']['padded']['radii'], full_dict['val_dict']['padded']['pulse_time_ids']
        test_X, test_y, test_mask, test_radii, test_ids = full_dict['test_dict']['padded']['profiles'],full_dict['test_dict']['padded']['controls'], full_dict['test_dict']['padded']['masks'], full_dict['test_dict']['padded']['radii'], full_dict['test_dict']['padded']['pulse_time_ids']

    train_pulse_order = [int(x.split('/')[0]) for x in train_ids]
    val_pulse_order = [int(x.split('/')[0]) for x in val_ids]
    test_pulse_order = [int(x.split('/')[0]) for x in test_ids]
    return (train_X, train_y, train_mask, train_radii, train_ids, train_pulse_order), (val_X, val_y, val_mask, val_radii, val_ids, val_pulse_order), (test_X, test_y, test_mask, test_radii, test_ids, test_pulse_order)

def get_jet_pedestal_database(data_path=''):
    if not data_path:
        data_path = '/home/adam/ENR_Sven/moxie/data/processed/jet-pedestal-database.csv'
    jet_pedestal_database = pd.read_csv(data_path)
    jet_pedestal_database = jet_pedestal_database[(jet_pedestal_database['FLAG:HRTSdatavalidated'] > 0)]

    return jet_pedestal_database

def get_indexes_of_relevant_pulses(pulse_order, options=[]):
    if not options:
        raise ValueError('No Pulses given to compare, check that a list is being passed ot get_indexes_of_relevant_pulses')
    pulse_1_idxs, pulse_2_idxs = [], []
    for n, idx in enumerate(pulse_order):
        if idx == options[0]:
            pulse_1_idxs.append(n)
        elif idx == options[1]:
            pulse_2_idxs.append(n)
    return pulse_1_idxs, pulse_2_idxs

def plot_profile_comparisons(mp_dim, latex_name, PULSE_1, PULSE_2, annotation='', T_LIM=None):
    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True, figsize=(10, 5))

    PSI_LIM = (0.8,1.05)

    profs, radii, masks, ID, mps = PULSE_1
    profs_d = profs[:, 0, :]
    profs_t = profs[:, 1, :]

    if mp_dim == 4 or mp_dim == 5:
        avg_mp = np.mean((mps[:, 4] + mps[:, 5]) / 2)
    else:
        avg_mp = np.mean(mps[:, mp_dim])

    for n, (d_prof, t_prof, rad, mask) in enumerate(zip(profs_d, profs_t, radii, masks)):
        if n == 0:
            label = str(ID) + '\n' +  latex_name + '= {:.3}'.format(avg_mp)
        else:
            label = None
        axs[0].scatter(rad[mask], d_prof[mask], c='blue', alpha=0.5)
        axs[1].scatter(rad[mask], t_prof[mask], c='blue', label=label, alpha=0.5)

    axs[0].plot(np.mean(radii, 0), np.mean(profs_d, 0), c='black', ls='--')
    axs[1].plot(np.mean(radii, 0), np.mean(profs_t, 0), c='black', ls='--')

    profs, radii, masks, ID, mps = PULSE_2
    profs_d = profs[:, 0, :]
    profs_t = profs[:, 1, :]

    if mp_dim == 4 or mp_dim == 5:
        avg_mp = np.mean((mps[:, 4] + mps[:, 5]) / 2)
    else:
        avg_mp = np.mean(mps[:, mp_dim])

    for n, (d_prof, t_prof, rad, mask) in enumerate(zip(profs_d, profs_t, radii, masks)):
        if n == 0:
            label = str(ID) + '\n' +  latex_name + '= {:.3}'.format(avg_mp)
        else:
            label = None

        axs[0].scatter(rad[mask], d_prof[mask], c='salmon', alpha=0.5)
        axs[1].scatter(rad[mask], t_prof[mask], c='salmon', label=label, alpha=0.5)

    axs[0].plot(np.mean(radii, 0), np.mean(profs_d, 0), c='black', ls='--')
    axs[1].plot(np.mean(radii, 0), np.mean(profs_t, 0), c='black', ls='--')
    axs[0].set_xlim(PSI_LIM)
    axs[1].set_ylim(T_LIM)
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('$\psi_N$')
    axs[0].set_xlabel('$\psi_N$')
    axs[0].set_ylabel('$n_e$ [m$^{-3}$]')
    axs[1].set_ylabel('$T_e$ [eV]')
    axs[0].annotate(**annotation)
    plt.show()

def plot_latent_space(z, mps, mp_dim=-1, ls_dims=[1,3,7], MP_NAME='Something'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    if mp_dim == 4 or mp_dim == 5:
        ALL_NORM = matplotlib.colors.Normalize(vmin=((mps[:, 4] + mps[:, 5]) / 2.0).min(), vmax=((mps[:, 4] + mps[:, 5]) / 2.0).max(), clip=False)
        ALL_MAPPER = cm.ScalarMappable(norm=ALL_NORM, cmap=cm.seismic)
        c_vals = ALL_NORM((mps[:, 4] + mps[:, 5]) / 2.0)

        im = ax.scatter(z[:, ls_dims[0]], z[:, ls_dims[1]], z[:, ls_dims[2]], c=c_vals, alpha=0.5, cmap=cm.seismic)
        MP_NAME = 'TRIANGULARITY'
    else:
        ALL_NORM = matplotlib.colors.Normalize(vmin=(mps[:, mp_dim]).min(), vmax=(mps[:, mp_dim]).max(), clip=False)
        ALL_MAPPER = cm.ScalarMappable(norm=ALL_NORM, cmap=cm.seismic)
        c_vals = ALL_NORM(mps[:, mp_dim])

        im = ax.scatter(z[:, ls_dims[0]], z[:, ls_dims[1]], z[:, ls_dims[2]], c=c_vals, alpha=0.5, cmap=cm.seismic)

    ax.set_xlabel('Z: {}'.format(ls_dims[0]))
    ax.set_ylabel('Z: {}'.format(ls_dims[1]))
    ax.set_zlabel('Z: {}'.format(ls_dims[2]))
    LS_XLIM, LS_YLIM, LS_ZLIM = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    cax = fig.add_axes([0.15, .87, 0.75, 0.031])
    fig.colorbar(ALL_MAPPER, label=MP_NAME, cax=cax, orientation='horizontal', pad=0.6)
    # RETURN norm, mapper
    plt.show()

def main(experiment_name="TRIANGULARITY"):

    model_hyperparams = {'in_ch': 2, 'out_length':19,
    'mach_latent_dim': 7, 'stoch_latent_dim': 3,
    'beta_stoch': 10e-3,
    'beta_mach':  250., 'alpha_mach': 1.0, 'alpha_prof': 1.0,
    'loss_type': 'semi-supervised'}


    experiment = experiment_dict[experiment_name]
    pulse_1_int = experiment['PULSE_LOW']["id"]
    pulse_2_int = experiment['PULSE_HIGH']["id"]
    mp_dim = experiment['mp_dim']

    LS_DIMS = experiment['ls_dims']
    PULSES_TO_COMPARE = [pulse_1_int, pulse_2_int]
    print(PULSES_TO_COMPARE)
    MP_NAME = machine_param_order[mp_dim]
    LATEX_MP_NAME = experiment['latex']
    ANNOTATION = experiment['annotation']


    train_data, _, _ = get_profile_data()
    train_X, train_y, train_mask, train_radii, train_ids, train_pulse_order = train_data

    pulse_1_idxs, pulse_2_idxs = get_indexes_of_relevant_pulses(options=PULSES_TO_COMPARE,pulse_order=train_pulse_order)

    PULSE_1 = train_X[pulse_1_idxs], train_radii[pulse_1_idxs], train_mask[pulse_1_idxs] > 0, PULSES_TO_COMPARE[0], train_y[pulse_1_idxs]
    PULSE_2 = train_X[pulse_2_idxs], train_radii[pulse_2_idxs], train_mask[pulse_2_idxs] > 0, PULSES_TO_COMPARE[1], train_y[pulse_2_idxs]
    # 1.) Plot profiles
    plot_profile_comparisons(mp_dim, LATEX_MP_NAME, PULSE_1, PULSE_2, ANNOTATION, T_LIM=experiment['profile_lims']['T_LIM'])


    # get the model and prepare data for passing (i.e., tensors)
    model = DIVAMODEL(**model_hyperparams)
    state_dict = torch.load('./samplemodelstatedict.pth')
    model.load_state_dict(state_dict['model'])
    MP_norm, MP_var= state_dict['MP_norms']
    D_norm, D_var= state_dict['D_norms']
    T_norm, T_var= state_dict['T_norms']

    train_mp_tensors = torch.tensor(train_y).float()
    train_mp_normalized = standardize(train_mp_tensors, MP_norm, MP_var)

    train_profiles = torch.tensor(train_X)

    train_profiles_normalized = torch.clone(train_profiles).float()
    train_profiles_normalized[:, 0] = standardize(train_profiles_normalized[:, 0], D_norm, D_var)
    train_profiles_normalized[:, 1] = standardize(train_profiles_normalized[:, 1], T_norm, T_var)

    Z_MACH_TRAINING, Z_STOCH_TRAINING, Z_TRAINING = get_latent_space(train_profiles_normalized, model)

    plot_latent_space(Z_MACH_TRAINING, train_mp_tensors, mp_dim=mp_dim, ls_dims=LS_DIMS, MP_NAME=LATEX_MP_NAME)


experiment_dict =    {
        "TRIANGULARITY":
            {
                "PULSE_LOW": {'id': 82127, 'value': 0.248164},
                "PULSE_HIGH": {'id': 82647, 'value': 0.377240},
                "annotation": {'text': '$q_{95} = 3.2-3.4$\n$I_{P} = 2$ [MA]\n$B_{T} = 2$ [T]\n$P_{abs} = 10$ [MW]\nV/H divertor', 'xy': (0.68,0.70), "xycoords": 'axes fraction', 'size': 'large'},
                "profile_lims": {'PSI_LIM': (0.8, 1.05), 'T_LIM': (0, 1200)},
                "mp_dim": 4,
                "ls_dims": [0, 1, 2],
                # "ls_dims": [1, 5, 7],
                "latex": '$\delta$'
            },
        "GASPUFF":
            {
                "PULSE_LOW": {'id': 82130, 'value': 1.12e22},
                "PULSE_HIGH": {'id': 81982, 'value': 6.95e22, "subset": [43]},# Because there are two sets of this, so do subset[0]:
                "annotation": {'text': '$q_{95} = 3.2-3.4$\n$I_{P} = 2$ [MA]\n$B_{T} = 2$ [T]\n$P_{abs} = 10$ [MW]\nV/H divertor\n$\delta=0.27$', 'xy': (0.68,0.70), "xycoords": 'axes fraction', 'size': 'large'},
                "profile_lims": {'PSI_LIM': (0.8, 1.05), 'T_LIM': (-20, 1400)},
                "latex": '$\Gamma$',
                "ls_dims": [3, 6, 7],
                "mp_dim": -1,
            },
        "NBI POWER":
            {
                "PULSE_LOW": {'id': 83249, 'value': 26.5},
                "PULSE_HIGH": {'id': 83551, 'value': 17.3},
                "annotation": {'text': '$q_{95} = 3$\n$I_{P} = 2.5$ [MA]\n$B_{T} = 2$ [T]\n$\Gamma = 2-2.5$ [e/s]\nV/H divertor\n$\delta=0.28$', 'xy': (0.65,0.70), "xycoords": 'axes fraction', 'size': 'large'},
                "profile_lims": {'PSI_LIM': (0.8, 1.05), 'T_LIM': (-20, 1400)},
                "latex": '$\Gamma$',
                "ls_dims": [1, 3, 7],
                "mp_dim": -3,
            }
    }
if __name__== '__main__':
    main()
