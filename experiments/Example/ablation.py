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
from moxie.experiments.DIVA_EXP_AK import EXAMPLE_DIVA_EXP_AK

import pytorch_lightning as pl 
import numpy as np 

# Find the curent path of the file 
file_path = pathlib.Path(__file__).resolve()# .parent.parent
# Path of experiment 
exp_path = file_path.parent
# Path of moxie stuffs 
home_path = file_path.parent.parent.parent 
# Path to data 
dataset_path = home_path / 'data' / 'processed' / 'pedestal_profiles_ML_READY_ak_09022022.pickle'
print('\n# Path to Dataset Exists? {}'.format(dataset_path.exists()))
print(dataset_path.resolve())


def get_param_dict(old_dict, param_to_vary, lower_bound, upper_bound, n_samples):
    study = []
    
    study_variations = np.linspace(lower_bound, upper_bound, n_samples)
    for n in study_variations: 
        new_dict = old_dict.copy()
        new_dict.update([(param_to_vary, n)])
        study.append(new_dict)

    return study 

# SEED EVERYTHING! 



# TODO: move to a config file
STATIC_PARAMS = {'data_dir':dataset_path, 'num_workers': 4, 'pin_memory': False, 'dataset_choice': 'padded'}

HYPERPARAMS = {'LR': 0.0025, 'weight_decay': 0.0, 'batch_size': 512}

model_hyperparams = {'in_ch': 2, 'out_length':19, 
                    'mach_latent_dim': 8, 'beta_stoch': 0.00211535, 
                    'beta_mach':  530., 'alpha_mach': 10.0, 'alpha_prof': 1.0, 
                    'loss_type': 'semi-supervised'}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}

study = get_param_dict(params, 'LR', lower_bound=0.001, upper_bound=0.01, n_samples=10)

for param in study: 
    pl.utilities.seed.seed_everything(42)
    # TODO: Grab Sizes of the input/output datasets 

    datacls = PLDATAMODULE_AK(**param)

    model = DIVAMODEL(**model_hyperparams)

    trainer_params = {'max_epochs': 50, 'profiler': 'simple', 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value'}

    logger = pl.loggers.TensorBoardLogger(exp_path / "tb_logs", name='LR={:.4}'.format(param['LR']))

    experiment = EXAMPLE_DIVA_EXP_AK(model, param)
    runner = pl.Trainer(logger=logger, **trainer_params)

    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)
# Do something with it
