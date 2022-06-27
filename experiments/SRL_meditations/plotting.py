from torch_dataset import *
from python_dataset import PULSE, PROFILE, MACHINEPARAMETER, ML_ENTRY 
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import matplotlib.gridspec as gridspec
t = 0
save_loc = '/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/model_results/'

def denormalize_profiles(profiles, norms): 
    def de_standardize(x, mu, var): 
        if isinstance(x, torch.Tensor): 
            mu, var = torch.from_numpy(mu), torch.from_numpy(var)
        return x*var + mu
    N_mean, N_var, T_mean, T_var, _, _ = norms
    profiles[:, 0, :] = de_standardize(profiles[:, 0, :], N_mean, N_var)
    profiles[:, 1, :] = de_standardize(profiles[:, 1, :], T_mean, T_var)
    return profiles

def normalize_profiles(profiles, norms): 
    def standardize(x, mu, var):
        if mu is None and var is None:
            mu = x.mean(0, keepdim=True)[0]
            var = x.std(0, keepdim=True)[0]
        x_normed = (x - mu ) / var
        return x_normed, mu, var
    N_mean, N_var, T_mean, T_var, _, _ = norms
    profiles[:, 0, :], _, _ = standardize(profiles[:, 0, :], N_mean, N_var)
    profiles[:, 1, :], _, _ = standardize(profiles[:, 1, :], T_mean, T_var)
    return profiles

def normalize_mps(mps, norms): 
    def standardize(x, mu, var):
        if mu is None and var is None:
            mu = x.mean(0, keepdim=True)[0]
            var = x.std(0, keepdim=True)[0]
        x_normed = (x - mu ) / var
        return x_normed, mu, var
    _, _, _, _, MP_mu, MP_var = norms
    mps, _, _ = standardize(mps, MP_mu, MP_var)
    return mps

def denormalize_mps(mps, norms): 
    def destandardize(x, mu, var):
        if mu is None and var is None:
            mu = x.mean(0, keepdim=True)[0]
            var = x.std(0, keepdim=True)[0]
        x_normed = x*var + mu
        return x_normed, mu, var
    _, _, _, _, MP_mu, MP_var = norms
    mps, _, _ = destandardize(mps, MP_mu, MP_var)
    return mps


def main(model_name='STEP2_aux_cond'): 
    global EPOCH, datacls, model, t
    # model = VAE_LLD(input_dim=2, latent_dim=10, conv_filter_sizes=[8, 10, 12], transfer_hidden_dims=[10, 20, 30])
    model = VAE_LLD_MP(input_dim=2, latent_dim=20, out_length=75, 
                        conv_filter_sizes=[8, 10, 12], transfer_hidden_dims=[20, 20, 30], 
                        reg_hidden_dims= [40, 40, 40, 40, 40], mp_dim=14, act_dim=14)
    save_dict = torch.load(save_loc + model_name + '.pth')
    state_dict = save_dict['state_dict']
    with open(save_loc + model_name + '_data', 'rb') as file: 
        data_dict = pickle.load(file)
    train_set, val_set, norms = data_dict['train_pulses'], data_dict['val_pulses'], data_dict['norms']
    model.load_state_dict(state_dict)
    model.double()

    plot_condtional(model, val_set, norms)
    plot_condtional(model, train_set, norms)
    # plot_comparison(model, val_set, norms)
def plot_condtional(model, pulses: List[PULSE], norms): 
    """
    Loop through a pulse
    1.) Try to check just from machine parameter conditional priors at each time step to get profiles 
    2.) Go from t_0 and try to get all 
    """
    
    for n, pulse in enumerate(pulses):
        
        all_profs, all_mps = pulse.get_ML_ready_array()
        t = np.arange(len(all_profs))
        t0_profs, t1_profs = all_profs[:-1, :, :], all_profs[1:, :, :]
        t0_mps, t1_mps = all_mps[:-1, :], all_mps[1:, :]
        mp_diff = t1_mps - t0_mps 
        # Normalize everything! 
        t0_mps_norm, mp_diff_norm = torch.from_numpy(t0_mps).double(), torch.from_numpy(mp_diff).double()
        t0_mps_norm, mp_diff_norm = normalize_mps(t0_mps_norm, norms), normalize_mps(mp_diff_norm, norms)
        with torch.no_grad(): 
            z_t_cond_prior, mu_t_cond_prior, var_t_cond_prior = model.mp2z(t0_mps_norm)
            z_hat_t_1, A_t, B_t, o_t = model.zt2zt_1(z_t_cond_prior, mp_diff=mp_diff_norm)

            # Profiles expected
            x_hat_hat_t_1 = model.z2x(z_hat_t_1)
            # Machine parameters expected 
            mp_hat_hat_t_1 = model.z2mp(z_hat_t_1) 

        mp_pred = torch.clone(mp_hat_hat_t_1)
        mp_pred = denormalize_mps(mp_pred, norms)
        fig = plt.figure() 
        subfigs = fig.subfigures(2, 1, wspace=0.01)
        axs = subfigs[0].subplots(2, 7, sharex=True)
        # Now we can try to plot the machine parameters... 
        # fig, axs = plt.subplots(2, 7)
        bars = []
        for dim, (ax, label) in enumerate(zip(axs.ravel(), pulse.control_param_labels)): 
            
            ax.plot(t0_mps[:, dim], color='blue', label='real')
            ax.plot(mp_pred[:, dim], color='red', label='pred')
            ax.set_xlabel(label)
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            bars.append(ax.axvline(t[0], color='black'))
            if dim == 0: 
                ax.legend(frameon=False)

        fig.suptitle(str(pulse.pulse_id))

        prof_pred = torch.clone(x_hat_hat_t_1)
        prof_pred = denormalize_profiles(prof_pred, norms)
        n_ax = subfigs[1].subplots()
        t_ax = n_ax.twinx()

        xdata, ndata, tdata = [], [], []
        n_ln, = n_ax.plot([], [], 'ro', label='Real $n_e$')
        t_ln, = t_ax.plot([], [], 'go', label='Real $T_e$')
        n_p_ln,  =  n_ax.plot([], [], 'm^', label='Pred $n_e$')
        t_p_ln, = t_ax.plot([], [], 'b^', label='Pred $T_e$')
        n_ax.legend(frameon=False, loc='center right')
        t_ax.legend(frameon=False, loc='upper right')
        min_n, max_n = all_profs[:, 0, :].min(), all_profs[:, 0, :].max()
        min_t, max_t = all_profs[:, 1, :].min(), all_profs[:, 1, :].max()
        n_ax.set_ylim(min_n, max_n)
        t_ax.set_ylim(min_t, max_t)
        n_ax.set_xlim(0, len(all_profs[0, 0, :]))
        t_ax.set_xlim(0, len(all_profs[0, 0, :]))
        time_template = 'time = %.3fs'
        time_text = n_ax.text(0.05, 0.9, '', transform=n_ax.transAxes)
        n_ax.set_title(f'{pulse.device} SHOT {pulse.pulse_id}, #Slices: {len(all_profs)}')
        n_ax.set_ylabel('$n_e (10^{19} m^{-3})$', color='red', fontsize='x-large')
        t_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
        def animate(i): 
            ndata = all_profs[i+1, 0, :]
            tdata = all_profs[i+1, 1, :]

            np_data = prof_pred[i, 0, :]
            tp_data = prof_pred[i, 1, :]

            xdata = range(len(all_profs[i, 0, :])) # radii[i] 
            n_ln.set_data(xdata, ndata)
            t_ln.set_data(xdata, tdata)

            n_p_ln.set_data(xdata, np_data)
            t_p_ln.set_data(xdata, tp_data)
            time_text.set_text(time_template % (t[i]*0.001))
            for bar in bars:
                bar.set_data(t[i], [0, 1e6]) 

            return n_ln, t_ln,n_p_ln, t_p_ln,time_text, *bars, # time_text, *bars
        
        ani = FuncAnimation(fig, animate, len(all_profs) - 1, interval=0.001, repeat_delay=1e3, blit=True)
        if pulse.pulse_id == 37723: 
            writer_video = FFMpegWriter(fps=150)
            ani.save('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/conditional_generation_1_37723.mp4', dpi=300, writer=writer_video)
        plt.show()

    pass 
def plot_comparison(model, pulses, norms):    
    for pulse in pulses: 
        profs, mps = pulse.get_ML_ready_array()
        profs = torch.from_numpy(profs).double()
        mps = torch.from_numpy(mps).double()
        # print(profs[0])
        min_n, max_n = profs[:, 0, :].min(), profs[:, 0, :].max()
        min_t, max_t = profs[:, 1, :].min(), profs[:, 1, :].max()
        profs = normalize_profiles(profs, norms)
        mps = normalize_mps(mps, norms)
        # print(profs[0])

        with torch.no_grad(): 
            z_t, *_ = model.x2z(profs)
            z_t_1, *_ = model.zt2zt_1(z_t)
            prof_t_1 = model.z2x(z_t_1)
            mp_t_1 = model.z2mp(z_t_1)

        prof_t_1 = denormalize_profiles(prof_t_1, norms)
        mps_t1 = denormalize_mps(mp_t_1, norms)
        profs = denormalize_profiles(profs, norms)
        break 
    fig, n_ax = plt.subplots()
    t_ax = n_ax.twinx()
    xdata, ndata, tdata = [], [], []
    n_ln, = n_ax.plot([], [], 'ro')
    t_ln, = t_ax.plot([], [], 'go')
    n_p_ln,  =  n_ax.plot([], [], 'mo')
    t_p_ln, = t_ax.plot([], [], 'bo')
    n_ax.set_ylim(min_n, max_n)
    t_ax.set_ylim(min_t, max_t)
    n_ax.set_xlim(0, len(profs[0, 0, :]))
    t_ax.set_xlim(0, len(profs[0, 0, :]))
    n_ax.set_title(f'{pulse.device} SHOT {pulse.pulse_id}, #Slices: {len(profs)}')
    n_ax.set_ylabel('$n_e (10^{19} m^{-3})$', color='red', fontsize='x-large')
    t_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
    # n_ax.set_xlabel(r'$\Psi_N$')
    # n_ax.axvline(1.0, ls='--', color='grey')

    def animate(i): 
        ndata = profs[i+1, 0, :]
        tdata = profs[i+1, 1, :]

        np_data = prof_t_1[i, 0, :]
        tp_data = prof_t_1[i, 1, :]

        xdata = range(len(profs[i, 0, :])) # radii[i] 
        n_ln.set_data(xdata, ndata)
        t_ln.set_data(xdata, tdata)

        n_p_ln.set_data(xdata, np_data)
        t_p_ln.set_data(xdata, tp_data)

        # for bar, mp_idx in zip(bars, mp_idxs): 
        #     bar.set_data(t[i], [0, 1e6])
        # time_text.set_text(time_template % (t[i]))
        # if not isinstance(lines[0], int): 
        #     for line, mp_idx in zip(lines, mp_idxs): line.set_data(t[i], self.mapped_mps[i, mp_idx]) 
        #     return n_ln, t_ln, time_text, *bars, *lines
        # else: 
        return n_ln, t_ln,n_p_ln, t_p_ln, # time_text, *bars
    ani = FuncAnimation(fig, animate, len(profs) - 1, interval=5, repeat_delay=1e3, blit=True)
    plt.show()

if __name__ == '__main__': 
    main()