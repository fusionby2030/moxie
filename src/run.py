import pytorch_lightning as pl
from experiment import VAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger


generator=torch.Generator().manual_seed(42)
torch.manual_seed(42)
logger = TensorBoardLogger("tb_logs", name="my_model")

STATIC_PARAMS = {'data_dir': '/home/kitadam/ENR_Sven/moxie/data/processed/pedestal_profile_dataset_v3.hdf5'}
HYPERPARAMS = {'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}

# from models.VAE import VanillaVAE
from models import BetaGammaVAE, VisualizeBetaVAE

model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':5, 'hidden_dims': [4, 8], 'beta': 5, 'gamma': 700000000.0, 'loss_type': 'G'}
# model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':5, 'hidden_dims': [2, 4, 8], 'beta':  50, 'loss_type': 'G', 'gamma': 7000000000.0}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}
model = VisualizeBetaVAE(**model_hyperparams)
trainer_params = {'max_epochs': 1000}

experiment = VAExperiment(model, params)
"""

runner = pl.Trainer(logger=logger, **trainer_params)

datacls = DataModuleClass(**params)

lr_finder = runner.tuner.lr_find(experiment, datamodule=datacls, max_lr=0.1)


new_lr = lr_finder.suggestion()

params['LR'] = new_lr

model = VisualizeBetaVAE(**model_hyperparams)

experiment = VAExperiment(model, params)
"""

runner = pl.Trainer(logger=logger, **trainer_params)

datacls = DataModuleClass(**params)

runner.fit(experiment, datamodule=datacls)
runner.test(experiment, datamodule=datacls)
