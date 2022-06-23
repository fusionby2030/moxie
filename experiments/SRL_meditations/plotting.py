from dataset import *
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
t = 0
def main(): 
    global EPOCH, datacls, model, t
    model = VAE_LLD(input_dim=2, latent_dim=10, conv_filter_sizes=[8, 10, 12], transfer_hidden_dims=[10, 20, 30])
    save_dict = torch.load('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/model_results/INITIAL_REVISED1.pth')
    state_dict, datacls = save_dict['state_dict'], save_dict['datacls']
    model.load_state_dict(state_dict)
    plot_comparison(model, datacls)
    # plot_conditionally(model, datacls)

def get_comparisons(model, datacls, real_profs): 
    profs = torch.from_numpy(real_profs)
    norm_profs = datacls.train_set.normalize_profiles(profs)
    with torch.no_grad(): 
        z_t, *_ = model.x2z(norm_profs)
        z_t_1, *_ = model.zt2zt_1(z_t)
        prof_t_1 = model.z2x(z_t_1)
    prof_t_1 = datacls.train_set.denormalize_profiles(profs)
    pred_profs = torch.cat((profs[0:1, :, :], prof_t_1), 0)
    return pred_profs
def plot_comparison(model, datacls):
    print('Comparison Plotting...') 
    with open(PICKLED_PULSES_FILELOC, 'rb') as file: 
        all_pulses = pickle.load(file)
    
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


def plot_conditionally(model, datacls): 
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