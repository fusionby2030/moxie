
import sys, os, pathlib

from moxie.models.PSI_model_ak1 import PSI_MODEL
from moxie.data.profile_lightning_module_psi import PLDATAMODULE_AK
from moxie.experiments.DIVA_EXP_AK_1 import EXAMPLE_DIVA_EXP_AK

import torch
import pytorch_lightning as pl

from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')

# Find the curent path of the file
file_path = pathlib.Path(__file__).resolve()# .parent.parent
# Path of experiment
exp_path = file_path.parent
# Path of moxie stuffs
home_path = file_path.parent.parent.parent
# Path to data
dataset_path = home_path / 'data' / 'processed' / f'ML_READY_dict_{CURRENT_DATE}.pickle'


# SEED EVERYTHING!

pl.utilities.seed.seed_everything(42)
# generator=torch.Generator().manual_seed(42)
# torch.manual_seed(42)

# TODO: move to a config file
STATIC_PARAMS = {'data_dir':dataset_path, 'num_workers': 4, 'pin_memory': False, 'elm_style_choice': 'simple'}

HYPERPARAMS = {'LR': 0.003, 'weight_decay': 0.0, 'batch_size':512, 'scheduler_step': 0}

# 'semi-supervised-start', 'semi-supervsied-cutoff', 'supervised'
model_hyperparams = {'out_length':20, 'elm_style_choice': 'simple',
                    'mach_latent_dim': 9, 'stoch_latent_dim': 3, # 0.0273
                    'beta_stoch': 1.0, 'beta_mach_unsup': 0.01,'beta_mach_sup':  0.00,
                    'alpha_mach': 100, 'alpha_prof': 100.0,  
                    'start_sup_time': 1000, 'scheduler_step': 5000,
                    'physics': True, 'gamma_stored_energy': 50.0, 'gamma_bpol': 100.0, 'gamma_beta': 10.0,
                    'mp_hdims_aux': [40, 40, 40, 40, 40, 40, 32, 16], 'mp_hdims_cond':[40, 40, 40, 40, 40, 40, 32, 16], 
                    'hidden_dims': [3, 3, 6], 'loss_type': 'semi-supervised-cutoff',}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}

datacls = PLDATAMODULE_AK(**params)
# TODO: Grab Sizes of the input/output datasets


model = PSI_MODEL(**model_hyperparams)
model_name='PSI_v2_physics'
trainer_params = {'max_epochs': 50, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value'}

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




