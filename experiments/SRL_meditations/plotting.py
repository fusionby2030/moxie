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
    model = VAE_LLD_MP(input_dim=2, latent_dim=13, out_length=75, 
                        conv_filter_sizes=[8, 10, 12], transfer_hidden_dims=[20, 20, 30], 
                        reg_hidden_dims= [40, 40, 40, 40, 40], mp_dim=14)
    save_dict = torch.load(save_loc + model_name + '.pth')
    state_dict = save_dict['state_dict']
    with open(save_loc + model_name + '_data', 'rb') as file: 
        data_dict = pickle.load(file)
    train_set, val_set, norms = data_dict['train_pulses'], data_dict['val_pulses'], data_dict['norms']
    model.load_state_dict(state_dict)
    model.double()

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

        # Now we can try to plot the machine parameters... 
        fig, axs = plt.subplots(2, 7)
        for dim, (ax, label) in enumerate(zip(axs.ravel(), pulse.control_param_labels)): 
            ax.plot(t0_mps[:, dim], color='blue')
            ax.plot(mp_pred[:, dim], color='red')
            ax.set_title(label)
        fig.suptitle(str(pulse.pulse_id))
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
def plot_condtional_old_2(model, pulses, norms): 
    dt = 0.001
    
    for n, pulse in enumerate(pulses):
        if n != 0: 
            continue 
        results_profs = []
        profs, mps = pulse.get_ML_ready_array()
        print(mps.shape)
        min_n, max_n = profs[:, 0, :].min(), profs[:, 0, :].max()
        min_t, max_t = profs[:, 1, :].min(), profs[:, 1, :].max()
        profs = torch.from_numpy(profs).double()
        results_profs.append(profs[0])

        t_0_profs = torch.clone(profs[0:10, :, :])
        t_0_profs = normalize_profiles(t_0_profs, norms)
        
        # t = np.arange(0, len(profs)*dt, step = dt)
        t = np.arange(0, 1, step = dt)
        for time in tqdm(t): 
            with torch.no_grad(): 
                z_t, *_ = model.x2z(t_0_profs)
                z_t_1, *_ = model.zt2zt_1(z_t)
                t_1_profs = model.z2x(z_t_1)
            t_0_profs = torch.clone(t_1_profs) 
            t_1_profs = denormalize_profiles(t_1_profs, norms)
            results_profs.append(t_1_profs[0])
        break 
    fig, n_ax = plt.subplots()
    t_ax = n_ax.twinx()
    xdata, ndata, tdata = [], [], []
    n_ln, = n_ax.plot([], [], 'ro', label='real')
    t_ln, = t_ax.plot([], [], 'go', label='real')
    n_p_ln,  =  n_ax.plot([], [], 'mo', label='Cond.')
    t_p_ln, = t_ax.plot([], [], 'bo', label='Cond.')
    n_ax.set_ylim(min_n, max_n)
    t_ax.set_ylim(min_t, max_t)
    n_ax.set_xlim(0, len(profs[0, 0, :]))
    t_ax.set_xlim(0, len(profs[0, 0, :]))
    n_ax.set_title(f'{pulse.device} SHOT {pulse.pulse_id}, #Slices: {len(profs)}')
    n_ax.set_ylabel('$n_e (10^{19} m^{-3})$', color='red', fontsize='x-large')
    t_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
    time_template = 'time = %.3fs'
    time_text = n_ax.text(0.05, 0.9, '', transform=n_ax.transAxes)
    n_ax.legend()
    # n_ax.set_xlabel(r'$\Psi_N$')
    # n_ax.axvline(1.0, ls='--', color='grey')

    def animate(i): 
        ndata = profs[i, 0, :]
        tdata = profs[i, 1, :]

        np_data = results_profs[i][0, :]
        tp_data = results_profs[i][1, :]

        xdata = range(len(profs[i, 0, :])) # radii[i] 
        n_ln.set_data(xdata, ndata)
        t_ln.set_data(xdata, tdata)

        n_p_ln.set_data(xdata, np_data)
        t_p_ln.set_data(xdata, tp_data)

        time_text.set_text(time_template % (t[i]))

        # for bar, mp_idx in zip(bars, mp_idxs): 
        #     bar.set_data(t[i], [0, 1e6])
        # time_text.set_text(time_template % (t[i]))
        # if not isinstance(lines[0], int): 
        #     for line, mp_idx in zip(lines, mp_idxs): line.set_data(t[i], self.mapped_mps[i, mp_idx]) 
        #     return n_ln, t_ln, time_text, *bars, *lines
        # else: 
        return n_ln, t_ln,n_p_ln, t_p_ln, time_text,#  *bars
    ani = FuncAnimation(fig, animate, len(results_profs) - 1, interval=100, repeat_delay=1e3, blit=True)
    plt.show()
    pass 
def plot_comparison_old(model, datacls):
    print('Comparison Plotting...') 
    
    
    rel_pulse = all_pulses[0] 

    fig, n_ax = plt.subplots()
    t_ax = n_ax.twinx()
    # p_ax = n_ax.twinx()
    profs = np.array([prof.get_ML_ready_array() for prof in rel_pulse.profiles])

    pred_profs = get_comparisons(model, datacls, profs)
    radii = np.array([prof.radius[-60:-10] for prof in rel_pulse.profiles])
    t = np.array([prof.time_stamp for prof in rel_pulse.profiles])

    n_ln, = n_ax.plot([], [], 'ro')
    t_ln, = t_ax.plot([], [], 'go')

    pn_ln, = n_ax.plot([], [], color='salmon')
    pt_ln, = t_ax.plot([], [], color='cyan')

    min_n, max_n = profs[:, 0, :].min(), profs[:, 0, :].max()
    min_t, max_t = profs[:, 1, :].min(), profs[:, 1, :].max()
    time_template = 'time = %.3fs'
    time_text = n_ax.text(0.05, 0.9, '', transform=n_ax.transAxes)
    
    n_ax.set_xlim(radii.min(), radii.max())
    t_ax.set_xlim(radii.min(), radii.max())
    n_ax.set_ylim(min_n, max_n)
    t_ax.set_ylim(min_t, max_t)
    n_ax.set_ylabel('$n_e (10^{19} m^{-3})$', color='red', fontsize='x-large')
    t_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
    n_ax.set_xlabel(r'$\Psi_N$')
    n_ax.axvline(1.0, ls='--', color='grey')
    # p_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
    n_ax.set_title(f'#Slices: {len(profs)}')
    fig.suptitle(f'{rel_pulse.device} SHOT {rel_pulse.pulse_id}')
    def animate(i): 
        ndata = profs[i, 0, :]
        tdata = profs[i, 1, :]
        pndata = pred_profs[i, 0, :]
        ptdata = pred_profs[i, 1, :]
        xdata = radii[i] # range(len(profs[i, 0, :]))
        n_ln.set_data(xdata, ndata)
        t_ln.set_data(xdata, tdata)
        pn_ln.set_data(xdata, pndata)
        pt_ln.set_data(xdata, ptdata)
        time_text.set_text(time_template % (t[i]))
        return n_ln, t_ln, pn_ln, pt_ln, time_text
    ani = FuncAnimation(fig, animate, len(t), interval=(t[1] - t[0])*20000, repeat_delay=1e3, blit=True)
    # writer_video = FFMpegWriter(fps=(t[1] - t[0])*20000)
    # ani.save('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/comparison_video_fast.mp4', dpi=300, writer=writer_video)
    
    plt.show()



def plot_conditionally_old(model, datacls): 
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
    test_dl = iter(datacls.test_dataloader())
    batch = next(test_dl)
    results_profs = []
    results_latent, results_latent_zt = [], []
    t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
    results_profs.append(t_0_batch[0])
    dt = 0.001
    t_stop = 1.5 # 1.0
    t = np.arange(0, t_stop, dt)
    for time in t: 
        with torch.no_grad(): 
            z_t, *_ = model.x2z(t_0_batch)
            z_t_1, *_ = model.zt2zt_1(z_t)
            if time == 0: 
                results_latent.append(z_t[0])
                results_latent.append(z_t_1[0])
            else: 
                results_latent.append(z_t_1[0])
                results_latent_zt.append(z_t[0])
            prof_t_1 = model.z2x(z_t_1)
            t_0_batch = prof_t_1
            results_profs.append(prof_t_1[0])

    ax = plt.figure().add_subplot(projection='3d')
    results_latent = torch.vstack(results_latent)
    results_latent_zt = torch.vstack(results_latent_zt)
    
    ax.plot(results_latent[:, 0], results_latent[:, 1], results_latent[:, 2])
    ax.scatter(results_latent[:, 0], results_latent[:, 1], results_latent[:, 2], color='green', label=r'$z_{t1}$\n(prev->enc->trans)')
    ax.scatter(results_latent_zt[:, 0], results_latent_zt[:, 1], results_latent_zt[:, 2], color='gold', label=r'$z_t$ (prev->enc)')
    ax.scatter(results_latent[0, 0], results_latent[0, 1], results_latent[0, 2], color='red', marker='*', s=150)
    ax.scatter(results_latent[-1, 0], results_latent[-1, 1], results_latent[-1, 2], color='black', marker='*', s=150)
    ax.legend()
    # plt.savefig('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/full_pulse_train')
    plt.show()
    
    prof_array = torch.stack(results_profs, 0)
    plot_final_animation(prof_array, (dt, t))

def plot_final_animation(prof_array, time_tuple): 
    """ 
    Takes list of profs from previous experiment (latent space)
    and plots them as an animation
    """
    dt, t = time_tuple
    
    prof_de_norm = datacls.test_set.denormalize_profiles(prof_array)

    
    fig, (n_ax, t_ax) = plt.subplots(2, 1, sharex=True)
    xdata, ndata, tdata = [], [], []
    n_ln, = n_ax.plot([], [], 'ro')
    t_ln, = t_ax.plot([], [], 'go')
    min_n, max_n = prof_de_norm[:, 0, :].min(), prof_de_norm[:, 0, :].max()
    min_t, max_t = prof_de_norm[:, 1, :].min(), prof_de_norm[:, 1, :].max()
    time_template = 'time = %.3fs'
    time_text = n_ax.text(0.05, 0.9, '', transform=n_ax.transAxes)
    n_ax.set_xlim(0, 51)
    t_ax.set_xlim(0, 51)
    n_ax.set_ylim(min_n, max_n)
    t_ax.set_ylim(min_t, max_t)

    
    
    def animate(i): 
        ndata = prof_de_norm[i, 0, :]
        tdata = prof_de_norm[i, 1, :]
        xdata = range(len(prof_de_norm[i, 0, :]))
        n_ln.set_data(xdata, ndata)
        t_ln.set_data(xdata, tdata)
        time_text.set_text(time_template % (i*dt))
        return n_ln, t_ln, time_text
    ani = FuncAnimation(fig, animate, len(t), interval=dt*20000, repeat_delay=1e3, blit=True)
    # writer_video = FFMpegWriter(fps=dt*20000)
    # ani.save('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/full_pulse_train_vid_fast.mp4', dpi=300, writer=writer_video)
    # Plot everything as an animation! 
    plt.show()

if __name__ == '__main__': 
    main()
    