import pickle 
import numpy as np
import pandas as pd
import pathlib 
import os, sys
import torch 
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
from torch.nn import functional as F


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


""" 

Produce the various experiments against known physical variations. 

- Triangularity:  Pulse 82127 (low d and trian) -> Pulse 82647 (high d and trian) 
- Gamma: 
- PTOT 
- IP 
- DC 
- ISOTOPE 


"""
 


"""


General Idea: 
    Grab relevant model outputs of specific pulses and test between them based on variations in latent space. 
    
1.) The interpolation between two pulses should follow general physical laws. 
    i.) The profile reconstruction should be continous between two points, highlighting basic changes in profile
        (e.g., going from High current to Low current should show a decrease in density)
    ii.) The machine parameter reconstruction should not vary the machine parameters held constant except for those that are different between the two profiles compared. 

RELEVANT PLOTS: 

    real profile comparison (ne and te) + mean of pulse windows 
    latent space all with relevant dimensions (return xlim, ylim, zlim)
    latent space of relevant profiles + latent space mean of relevant profiles 
    interpolation path in latent space between latent space means 
    interpolation of profiles between latent space means
        Also gives the predicted output of relevant machine parameter via coloring 
    interpolation of machine parameters between latent space means (13 plots?)
    reconstructed profile comparison with real + mean of reconstructed 
"""

def de_standardize(x, mu, var): 
    return (x*var) + mu

def standardize(x, mu, var): 
    return (x - mu) / var


def real_profile_comparison(PULSE_1, PULSE_2, de_normed=False, norms_d=None, norms_t=None): 
    # PULSE_I = (profs, radii, masks, id)
    profs, radii, masks, ID = PULSE_1 
    if de_normed and norms_d is not None and norms_t is not None: 
        profs_d = de_standardize(profs[:, 0, :], *norms_d)
        profs_t = de_standardize(profs[:, 1, :], *norms_t)
    else: 
        profs_d = profs[:, 0, :]
        profs_t = profs[:, 1, :]
    
    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True, figsize=(10, 5))
    for n, (d_prof, t_prof, rad, mask) in enumerate(zip(profs_d, profs_t, radii, masks)): 
        if n == 0: 
            label = ID 
        else: 
            label = None
        axs[0].scatter(rad[mask], d_prof[mask], c='salmon')
        axs[1].scatter(rad[mask], t_prof[mask], c='salmon', label=label)

    profs, radii, masks, ID = PULSE_2
    if de_normed and norms_d is not None and norms_t is not None: 
        profs_d = de_standardize(profs[:, 0, :], *norms_d)
        profs_t = de_standardize(profs[:, 1, :], *norms_t)
    else: 
        profs_d = profs[:, 0, :]
        profs_t = profs[:, 1, :]

    for n, (d_prof, t_prof, rad, mask) in enumerate(zip(profs_d, profs_t, radii, masks)): 
        if n == 0: 
            label = ID 
        else: 
            label = None
        
        axs[0].scatter(rad[mask], d_prof[mask], c='blue')
        axs[1].scatter(rad[mask], t_prof[mask], c='blue', label=ID)
    
    axs[1].legend()
    plt.show()

"""


NEEDS IN TERMS OF CONFIG: 

    RELEVANT LATENT DIMS TO PLOT
    NAME OF MP BEING VARIED 
    PULSE NUMBER (to find the indexes)

"""

"""

THINGS THAT WILL BE REUSED: 

    train_X, ..., ..., ... 
    train_X_norm, train_y_norm, .... 

    train_Z_MACH, train_Z_STOCH, train_Z 
    D_NORM, D_VAR 
    T_NORM, T_VAR 
    MP_NORM, MP_VAR 
    

"""
