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

# This is a really crude way to get to the moxie package. Be wary! 
import sys, os, pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from moxie.models.DIVA_ak_1 import DIVAMODEL
from moxie.data.profile_lightning_module import PLDATAMODULE_AK
from moxie.experiments.DIVA_EXP_AK import EXAMPLE_DIVA_EXP_AK

exp_path = os.path.dirname(os.path.realpath(__file__))

import pytorch_lightning as pl 

# generator=torch.Generator().manual_seed(42)
# torch.manual_seed(42)

pl.utilities.seed.seed_everything(42)
# desired_path = '/home/kitadam/ENR_Sven/moxie_revisited/data/processed/raw_padded_fitted_datasets.pickle'  
dir_path = pathlib.Path(__file__).parent.parent.parent
desired_path = dir_path / 'data' / 'processed' / 'raw_padded_fitted_datasets.pickle'
print('\n# Path to Dataset Exists? {}'.format(desired_path.exists()))
print(desired_path.resolve())

STATIC_PARAMS = {'data_dir':desired_path, 'num_workers': 1, 'pin_memory': False, 'dataset_choice': 'padded'}

HYPERPARAMS = {'LR': 0.002, 'weight_decay': 0.0, 'batch_size': 512}

model_hyperparams = {'in_ch': 2, 'out_length':24, 
                    'mach_latent_dim': 10, 'beta_stoch': 0.0025, 
                    'beta_mach':  500., 'alpha_mach': 1.0, 'alpha_prof': 25.0, 
                    'loss_type': 'semi-supervised'}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}

datacls = PLDATAMODULE_AK(**params)
# TODO: Grab Sizes of the input/output datasets 


model = DIVAMODEL(**model_hyperparams)

trainer_params = {'max_epochs': 150, 'profiler': 'simple', 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value'}

logger = pl.loggers.TensorBoardLogger(exp_path + "/tb_logs", name='Example')

experiment = EXAMPLE_DIVA_EXP_AK(model, params)
runner = pl.Trainer(logger=logger, **trainer_params)

runner.fit(experiment, datamodule=datacls)
runner.test(experiment, datamodule=datacls)
# Do something with it
