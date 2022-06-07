print("""\

                                       ._ o o
                                       \_`-)|_
                                    ,""       \
                                  ,"  ## |   ಠ ಠ.
                                ," ##   ,-\__    `.
                              ,"       /     `--._;)
                            ,"     ## /
                          ,"   ##    /

                            BE UNGOVERNABLE
                    """)

import sys, os, pathlib

from moxie.models.DIVA_ak_2 import DIVAMODEL
from moxie.data.profile_lightning_module import PLDATAMODULE_AK
from moxie.experiments.DIVA_EXP_AK_1 import EXAMPLE_DIVA_EXP_AK

import torch
import pytorch_lightning as pl

# Find the curent path of the file
file_path = pathlib.Path(__file__).resolve()# .parent.parent
# Path of experiment
exp_path = file_path.parent
# Path of moxie stuffs
home_path = file_path.parent.parent.parent
# Path to data
dataset_path = home_path / 'data' / 'processed' / 'pedestal_profiles_ML_READY_ak_19042022.pickle'
# print('\n# Path to Dataset Exists? {}'.format(dataset_path.exists()))
# print(dataset_path.resolve())



# SEED EVERYTHING!

pl.utilities.seed.seed_everything(42)
# generator=torch.Generator().manual_seed(42)
# torch.manual_seed(42)

# TODO: move to a config file
# dataset_choice = 'ALL', 'ALL_NO_VARIATIONS', 'SANDBOX_ALL', 'SANDBOX_NO_VARIATIONS'
STATIC_PARAMS = {'data_dir':dataset_path, 'num_workers': 4, 'pin_memory': False, 'dataset_choice': 'SANDBOX_NO_VARIATIONS'}

HYPERPARAMS = {'LR': 0.003, 'weight_decay': 0.0, 'batch_size':256, 'scheduler_step': 10000}

model_hyperparams = {'in_ch': 2, 'out_length':19,
                    'mach_latent_dim': 13, 'stoch_latent_dim': 3,
                    'beta_stoch': 1.1, 'beta_mach_unsup':  0.0001,'beta_mach_sup':  1.0,
                    'alpha_mach': 150.0, 'alpha_prof': 500.0,  # 212, 306, 26, 295, 263, 336] 	[485, 499, 352, 432]
                    'start_sup_time': 1000,
                    'physics': True, 'gamma_stored_energy': 20.0, 'gamma_bpol': 1.0, 'gamma_beta': 1.0, 
                    'mp_hdims_aux': [256, 128, 64], 'mp_hdims_cond':[256, 128, 64], # 'mp_hdims_aux': [263, 469, 284], 'mp_hdims_cond':[20, 136, 54, 205, 154, 408], # 'mp_hdims_cond': [397, 369, 29, 113, 284], 'mp_hdims_aux': [122, 398, 463, 354, 399], 
                    'hidden_dims': [2, 4], 'loss_type': 'semi-supervised',}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}

datacls = PLDATAMODULE_AK(**params)
# TODO: Grab Sizes of the input/output datasets


model = DIVAMODEL(**model_hyperparams)

trainer_params = {'max_epochs': 50, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value'}
if model_hyperparams['physics']:
    model_name =  'SCHEDULER_PHYSICS_{}MD_{}SD_{}BMUN_{}BMSUP_{}BS_{}AM_{}AP_{}EP'.format(model_hyperparams['mach_latent_dim'], model_hyperparams['stoch_latent_dim'], int(model_hyperparams['beta_mach_unsup']), model_hyperparams['beta_mach_sup'], model_hyperparams['beta_stoch'], int(model_hyperparams['alpha_mach']), int(model_hyperparams['alpha_prof']), trainer_params['max_epochs'])# 'VAE_7MD_3SD_500BM_50AM_10AP'
    if model_hyperparams['gamma_stored_energy'] > 0.0:
        model_name += '_{}GAMMA'.format(model_hyperparams['gamma_stored_energy'])
else:
    model_name =  'SCHEDULER_SECULAR_{}MD_{}SD_{}BMUN_{}BMSUP_{}BS_{}AM_{}AP_{}EP'.format(model_hyperparams['mach_latent_dim'], model_hyperparams['stoch_latent_dim'],                   int(model_hyperparams['beta_mach_unsup']), model_hyperparams['beta_mach_sup'], model_hyperparams['beta_stoch'], int(model_hyperparams['alpha_mach']), int(model_hyperparams['alpha_prof']),trainer_params['max_epochs'])# 'VAE_7MD_3SD_500BM_50AM_10AP'
    
# model_name = 'no_physics_decent'
# model_name = 'DIVA_W_PHYSICS_1'

print(model_name)

logger = pl.loggers.TensorBoardLogger(exp_path / "tb_logs", name=model_name)

experiment = EXAMPLE_DIVA_EXP_AK(model, params)
runner = pl.Trainer(logger=logger, **trainer_params)

runner.fit(experiment, datamodule=datacls)
runner.test(experiment, datamodule=datacls)

model = runner.model.model

state = {'model': model.state_dict(),
        'MP_norms': runner.datamodule.get_machine_norms(),
        'D_norms': runner.datamodule.get_density_norms(),
        'T_norms': runner.datamodule.get_temperature_norms(),
        'hyparams': model_hyperparams,
        }
model_pth_name = 'modelstatedict_' + model_name + '.pth'
torch.save(state, exp_path / 'model_results' / model_pth_name)
