import pytorch_lightning as pl
from experiment import VAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

from sklearn.model_selection import train_test_split


from pytorch_lightning.loggers import TensorBoardLogger


generator=torch.Generator().manual_seed(42)

logger = TensorBoardLogger("tb_logs", name="my_model")
params = {'LR': 0.0002, 'weight_decay': 0.000, 'batch_size': 512}

# from models.VAE import VanillaVAE
from models.ConVae import CNNVAE
model = CNNVAE(in_dim=63, latent_dim=12, hidden_dims = [2, 4, 8, 16])

experiment = VAExperiment(model, params)

trainer_params = {'max_epochs': 50}
runner = pl.Trainer(logger=logger, **trainer_params)

datacls = DataModuleClass(**params, conv=True)
runner.fit(experiment, datacls)

datacls.setup(stage="test")
print(runner.test(experiment, datamodule=datacls))
# runner.predict(experiment, datacls)
