import pytorch_lightning as pl
from experiment import VAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

from sklearn.model_selection import train_test_split


from pytorch_lightning.loggers import TensorBoardLogger


generator=torch.Generator().manual_seed(42)

logger = TensorBoardLogger("tb_logs", name="my_model")
params = {'LR': 0.0005, 'weight_decay': 0.00001, 'batch_size': 512, 'hidden_dims': [128, 256, 512, 1024]}

from models.VAE import VanillaVAE
model = VanillaVAE(in_dim=63, latent_dim=5)

experiment = VAExperiment(model, params)

trainer_params = {'max_epochs': 15}
runner = pl.Trainer(logger=logger, **trainer_params)

datacls = DataModuleClass(**params)
runner.fit(experiment, datacls)

datacls.setup(stage="test")
print(runner.test(experiment, datamodule=datacls))
# runner.predict(experiment, datacls)
