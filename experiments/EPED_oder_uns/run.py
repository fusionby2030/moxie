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

from moxie.models.DIVA_ak_1 import DIVAMODEL
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
dataset_path = home_path / 'data' / 'processed' / 'pedestal_profiles_ML_READY_ak_09022022.pickle'
# print('\n# Path to Dataset Exists? {}'.format(dataset_path.exists()))
# print(dataset_path.resolve())



# SEED EVERYTHING!

pl.utilities.seed.seed_everything(42)
# generator=torch.Generator().manual_seed(42)
# torch.manual_seed(42)

# TODO: move to a config file
STATIC_PARAMS = {'data_dir':dataset_path, 'num_workers': 3, 'pin_memory': False, 'dataset_choice': 'padded'}

HYPERPARAMS = {'LR': 0.003, 'weight_decay': 0.0, 'batch_size': 512}

model_hyperparams = {'in_ch': 2, 'out_length':19,
                    'mach_latent_dim': 9, 'stoch_latent_dim': 3,
                    'beta_stoch': 6.021, 'beta_mach_unsup':  0.245,'beta_mach_sup':  0.0,
                    'alpha_mach': 1595.0, 'alpha_prof': 1623.0,
                    'encoder_end_dense_size': 128, 
                    'hidden_dims': [2, 4], 'mp_hdims_cond': [70, 85], 'mp_hdims_aux': [70, 128, 240, 360], 
                    'physics': True, 'gamma_stored_energy': 0.36,
                    'start_sup_time': 50, 
                    'loss_type': 'semi-supervised'}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}

datacls = PLDATAMODULE_AK(**params)
# TODO: Grab Sizes of the input/output datasets


model = DIVAMODEL(**model_hyperparams)

trainer_params = {'max_epochs': 50, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value'}
if model_hyperparams['physics']:
    model_name =  'REDUCED_PHYSICS_{}MD_{}SD_{}BMUN_{}BMSUP_{}BS_{}AM_{}AP_{}ENCDENSE_{}EP'.format(model_hyperparams['mach_latent_dim'], model_hyperparams['stoch_latent_dim'], int(model_hyperparams['beta_mach_unsup']), model_hyperparams['beta_mach_sup'], model_hyperparams['beta_stoch'], int(model_hyperparams['alpha_mach']), int(model_hyperparams['alpha_prof']), model_hyperparams['encoder_end_dense_size'], trainer_params['max_epochs'])# 'VAE_7MD_3SD_500BM_50AM_10AP'
    if model_hyperparams['gamma_stored_energy'] > 0.0:
        model_name += '_{}GAMMA'.format(model_hyperparams['gamma_stored_energy'])
else:
    model_name =  'REDUCED_SECULAR_{}MD_{}SD_{}BMUN_{}BMSUP_{}BS_{}AM_{}AP_{}ENCDENSE_{}EP'.format(model_hyperparams['mach_latent_dim'], model_hyperparams['stoch_latent_dim'], int(model_hyperparams['beta_mach_unsup']), model_hyperparams['beta_mach_sup'], model_hyperparams['beta_stoch'], int(model_hyperparams['alpha_mach']), int(model_hyperparams['alpha_prof']), model_hyperparams['encoder_end_dense_size'], trainer_params['max_epochs'])# 'VAE_7MD_3SD_500BM_50AM_10AP'
# model_name = 'DIVA_W_PHYSICS_1'
# model_name =  'SEARCH_RESULTS{}MD_{}SD_{}BMUN_{}BMSUP_{}BS_{}AM_{}AP_{}ENCDENSE_{}EP'.format(model_hyperparams['mach_latent_dim'], model_hyperparams['stoch_latent_dim'], int(model_hyperparams['beta_mach_unsup']), model_hyperparams['beta_mach_sup'], model_hyperparams['beta_stoch'], int(model_hyperparams['alpha_mach']), int(model_hyperparams['alpha_prof']), model_hyperparams['encoder_end_dense_size'], trainer_params['max_epochs'])# 'VAE_7MD_3SD_500BM_50AM_10AP'
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






# Do something with it
