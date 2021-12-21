import pytorch_lightning as pl
# from experiment import VAExperiment, DualVAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

generator=torch.Generator().manual_seed(42)
torch.manual_seed(42)
logger = TensorBoardLogger("tb_logs", name="DIVA_Working")

STATIC_PARAMS = {'data_dir': '/home/adam/ENR_Sven/moxie/data/processed/profile_database_v1_psi22.hdf5',
                'num_workers': 4}
HYPERPARAMS = {'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}

# from models.VAE import VanillaVAE
from models import BetaGammaVAE, VisualizeBetaVAE, DualVAE, DualEncoderVAE, DIVA_v1

model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':5,
                        'beta_stoch':   0.000139164, 'beta_mach': 84840,
                        'alpha_mach': 1., 'alpha_prof': 1.0,
                    'loss_type': 'supervised'}
"""model_hyperparams = {'in_ch': 1, 'out_dim':63, 'hidden_dims': [2, 4],
                    'stoch_latent_dim':4, 'mach_latent_dim':13,
                    'num_conv_blocks': 3, 'num_trans_conv_blocks': 1,
                    'alpha': 1.0, 'beta_mach': 0.000005, 'beta_stoch': 0.00008, 'gamma': 0.000000}"""

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}
model = DIVA_v1(**model_hyperparams)
trainer_params = {'max_epochs': 1000,  'gpus': 1 if str(device).startswith('cuda') else 0, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm':"value", 'profiler':"advanced"}

from experiments import DualVAExperiment, BasicExperiment, DIVA_EXP

experiment = DIVA_EXP(model, params)


# early_stop_callback = EarlyStopping(monitor="hp/recon", min_delta=0.001, patience=15, verbose=True, mode="min")
# callbacks=[early_stop_callback]
runner = pl.Trainer(logger=logger, **trainer_params)


datacls = DataModuleClass(**params)

runner.fit(experiment, datamodule=datacls)
runner.test(experiment, datamodule=datacls)

"""
runner = pl.Trainer(logger=logger, **trainer_params)

datacls = DataModuleClass(**params)

lr_finder = runner.tuner.lr_find(experiment, datamodule=datacls, max_lr=0.1)


new_lr = lr_finder.suggestion()

params['LR'] = new_lr

model = VisualizeBetaVAE(**model_hyperparams)

experiment = VAExperiment(model, params)"""
