import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SMALL_SIZE = 40
MEDIUM_SIZE = 45
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['keymap.quit'].append(' ')


def plot_sample_profiles_from_batch(results, normalization=1.0, plot_params: dict = {'title': 'Representation Comparison of VAE', 'ylabel': '$n_e (m^{-3})$', 'xlabel': 'R (m)'}):
    """ Plot profiles from test_set

    """

    real_profiles, generated_profiles = results[1].squeeze(), results[0].squeeze()
    # Calculate Loss  between the two

    for i in range(0, len(real_profiles)-2, 10):
        fig, axs = plt.subplots(1, 1, figsize=(18, 18))
        axs.plot(real_profiles[i]*normalization, label='Real Profile', lw=4, c='black')
        axs.plot(generated_profiles[i]*normalization, label='Generated Profile', lw=4, c='forestgreen')

        axs.set(**plot_params)
        axs.legend()

        to_continue = input('Continue plotting? (y/n)')
        if to_continue == 'y':
            plt.show()
            continue
        else:
            plt.close()
            return False
