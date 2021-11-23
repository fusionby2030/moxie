import pytorch_lightning as pl
from experiment import VAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

generator=torch.Generator().manual_seed(42)
torch.manual_seed(42)
logger = TensorBoardLogger("tb_logs", name="my_model")

STATIC_PARAMS = {'data_dir': '/home/kitadam/ENR_Sven/moxie/data/processed/pedestal_profile_dataset_v3.hdf5', 'num_workers': 4}
HYPERPARAMS = {'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}

# from models.VAE import VanillaVAE
from models import BetaGammaVAE, VisualizeBetaVAE

model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':10, 'hidden_dims': [4, 8], 'beta': 5, 'gamma': 300000000000.0, 'loss_type': 'G'}
# model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':10, 'hidden_dims': [2, 4], 'beta': 5, 'gamma': 300000000000.0, 'loss_type': 'G'}

params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}
model = VisualizeBetaVAE(**model_hyperparams)
trainer_params = {'max_epochs': 10, 'gpus': 1 if str(device).startswith('cuda') else 0}


experiment = VAExperiment(model, params)




runner = pl.Trainer(logger=logger, **trainer_params)

runner.logger._log_graph = True

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
