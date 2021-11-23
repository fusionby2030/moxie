import pytorch_lightning as pl
from experiment import VAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

from models import BetaGammaVAE, VisualizeBetaVAE

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback = EarlyStopping(monitor="hp/final_loss", min_delta=0.001, patience=15, verbose=True, mode="min")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

generator=torch.Generator().manual_seed(42)
STATIC_PARAMS = {'data_dir': '/home/kitadam/ENR_Sven/moxie/data/processed/pedestal_profile_dataset_v3.hdf5', 'num_workers': 4}
HYPERPARAMS = {'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}
# model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':4, 'hidden_dims': [4, 8], 'beta': 5, 'gamma': 300000000000.0, 'loss_type': 'G'}
model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':4, 'hidden_dims': [4, 8], 'beta': 5, 'gamma': 3000000000000.0, 'loss_type': 'G'}
trainer_params = {'max_epochs': 300, 'gpus': 1 if str(device).startswith('cuda') else 0}
for ls in [4,5,8, 10, 12]:
    torch.manual_seed(42)

    logger = TensorBoardLogger("tb_logs", name="{}-search".format(ls))

    model_hyperparams['latent_dim'] = ls
    # model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':10, 'hidden_dims': [2, 4], 'beta': 5, 'gamma': 300000000000.0, 'loss_type': 'G'}

    params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}
    model = VisualizeBetaVAE(**model_hyperparams)



    experiment = VAExperiment(model, params)

    runner = pl.Trainer(logger=logger, callbacks=[early_stop_callback], **trainer_params)


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
