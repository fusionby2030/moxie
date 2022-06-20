from dataset import *
from model import *
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

t = 0
def main(): 
    global EPOCH, datacls, model, t
    model = VAE_LLD(input_dim=2, latent_dim=5, conv_filter_sizes=[4, 8], transfer_hidden_dims=[10, 20, 20, 20, 20])
    save_dict = torch.load('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/model_results/INITIAL.pth')
    state_dict, datacls = save_dict['state_dict'], save_dict['datacls']
    model.load_state_dict(state_dict)
    plot_conditionally(model, datacls)

def plot_comparison(model, datacls): 
    pass 


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
    results_x, results_y, results_z = [], [], []
    results_profs = []
    t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
    ax = plt.figure().add_subplot(projection='3d')
    results_profs.append(t_0_batch[0])
    dt = 0.002
    t_stop = 0.1
    t = np.arange(0, t_stop, dt)
    for time in t: 
        with torch.no_grad(): 
            z_t, *_ = model.x2z(t_0_batch)
            z_t_1, *_ = model.zt2zt_1(z_t)
            if time == 0: 
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
        if time == 0: 
            ax.scatter(z_t[0, 0], z_t[0, 1], z_t[0, 2], color='red')
        ax.scatter(z_t[0, 0], z_t[0, 1], z_t[0, 2], color='green')
        ax.scatter(z_t_1[0, 0], z_t_1[0, 1], z_t_1[0, 2], color='orange')
        ax.plot([z_t[0, 0], z_t_1[0, 0]], [z_t[0, 1], z_t_1[0, 1]], [z_t[0, 2], z_t_1[0, 2]], color='dodgerblue')
    ax.plot(results_x, results_y, results_z)
    # plt.savefig('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/cool_ls_graph')
    plt.show()
    
    prof_array = torch.stack(results_profs, 0)
    plot_final_animation(prof_array, (dt, t))

def plot_final_animation(prof_array, time_tuple): 
    """ 
    Takes list of profs from previous experiment (latent space)
    and plots them as an animation
    """
    dt, t = time_tuple
    from matplotlib.animation import FuncAnimation, PillowWriter
    prof_de_norm = datacls.test_set.denormalize_profiles(prof_array)

    
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')
    min_n, max_n = prof_de_norm[:, 0, :].min(), prof_de_norm[:, 0, :].max()
    time_template = 'time = %.3fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    ax.set_xlim(0, 51)
    ax.set_ylim(min_n, max_n)

    def init(): 
        ax.set_xlim(0, 51)
        ax.set_ylim(min_n, max_n)
        return ln, 
    def update(frame): 
        ydata = frame[0, :]
        xdata = range(len(frame[0, :]))
        ln.set_data(xdata, ydata)
        return ln, 
    def animate(i): 
        ydata = prof_de_norm[i, 0, :]
        xdata = range(len(prof_de_norm[i, 0, :]))
        ln.set_data(xdata, ydata)
        time_text.set_text(time_template % (i*dt))
        return ln, time_text
    ani = FuncAnimation(fig, animate, len(t), interval=dt*20000, repeat_delay=1e3, blit=True)
    # ani = FuncAnimation(fig, update, frames=prof_de_norm, init_func=init, blit=True)
    plt.show()
    # ani.save('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/cool_expansion.gif', dpi=300, writer=PillowWriter())
    # Plot everything as an animation! 
    
if __name__ == '__main__': 
    main()