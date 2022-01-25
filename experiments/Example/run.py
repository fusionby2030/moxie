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
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from moxie.models.DIVA_ak_1 import DIVAMODEL
from moxie.data.profile_lightning_module import PLDATAMODULE_AK
from moxie.experiments.DIVA_EXP_AK import EXAMPLE_DIVA_EXP_AK
import pathlib 

import pytorch_lightning as pl 


desired_path = '/home/kitadam/ENR_Sven/moxie_revisited/data/processed/raw_padded_fitted_datasets.pickle'  


STATIC_PARAMS = {'data_dir':desired_path, 'num_workers': 1, 'pin_memory': False, 'dataset_choice': 'padded'}

HYPERPARAMS = {'LR': 0.003, 'weight_decay': 0.0, 'batch_size': 512}

model_hyperparams = {'in_ch': 2, 'out_length':24, 'mach_latent_dim': 10, 'beta_stoch': 0.001, 'beta_mach':  300., 'alpha_mach': 25.0, 'alpha_prof': 1.0, 'loss_type': 'semi-supervised'}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}

datacls = PLDATAMODULE_AK(**params)
# TODO: Grab Sizes of the input/output datasets 


model = DIVAMODEL()

trainer_params = {'max_epochs': 15, 'profiler': 'simple'}

logger = pl.loggers.TensorBoardLogger("tb_logs", name='Example')

experiment = EXAMPLE_DIVA_EXP_AK(model, params)
runner = pl.Trainer(logger=logger, **trainer_params)

runner.fit(experiment, datamodule=datacls)
runner.test(experiment, datamodule=datacls)
# Do something with it
