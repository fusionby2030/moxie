"""


Model Idea:

- Different encoders for each machinen parameter.


Search runs

- Want lowest recon error on validation set!
- Also want highest correlation of input parameters
- Vary beta_stoch and beta_machine
- We want something that balances beta_stoch and beta_machine
- could sum up the abs(corrleation) for all, then return that, wanting it to be the max!

Random search (1000 Runs)
- 500 Epochs
- Beta_stoch > Loguniform(-5, 0)
- Beta_machine > Loguniform(-5, 0)

Results?
- TB Logs are not so big...

"""


import pytorch_lightning as pl
from experiment import VAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

from models import BetaGammaVAE, VisualizeBetaVAE

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback = EarlyStopping(monitor="hp/final_loss", min_delta=0.001, patience=15, verbose=True, mode="min")


import numpy as np 
from sklearn.model_selection import ParameterSampler 
from scipy.stats.distributions import loguniform


rng = np.random.RandomState(42)

param_grid = {'beta_mach': loguniform(0.000000001, 0.1), 'beta_stoch':loguniform(0.000000001, 0.1)} 

param_list = list(ParameterSampler(param_grid, n_iter=1000, random_state=4))

for updated_space in param_list:
    print(updated_space)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    generator=torch.Generator().manual_seed(42)

    torch.manual_seed(42)
    logger = TensorBoardLogger("tb_logs", name="DualEncoderVAE")

    STATIC_PARAMS = {'data_dir': '/home/adam/ENR_Sven/moxie/data/processed/profile_database_v1_psi22.hdf5',
                    'num_workers': 4}
    HYPERPARAMS = {'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}

    # from models.VAE import VanillaVAE
    from models import BetaGammaVAE, VisualizeBetaVAE, DualVAE, DualEncoderVAE

    # model_hyperparams = {'in_ch': 1, 'out_dim':63, 'latent_dim':10, 'hidden_dims': [4, 8], 'beta': 5, 'gamma': 300000000000.0, 'loss_type': 'G'}
    model_hyperparams = {'in_ch': 1, 'out_dim':63, 'hidden_dims': [2, 8],
                        'stoch_latent_dim':4, 'mach_latent_dim':13,
                        'num_conv_blocks': 3, 'num_trans_conv_blocks': 1,
                        'alpha': 1.0, 'beta_mach': 0.0000008, 'beta_stoch': 0.00008, 'gamma': 0.000000}
    model_hyperparams.update(updated_space)
    params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}
    model = DualEncoderVAE(**model_hyperparams)
    trainer_params = {'max_epochs': 500, 'gpus': 1 if str(device).startswith('cuda') else 0}

    from experiments import DualVAExperiment

    experiment = DualVAExperiment(model, params)


    # early_stop_callback = EarlyStopping(monitor="hp/recon", min_delta=0.001, patience=15, verbose=True, mode="min")
    # callbacks=[early_stop_callback]
    runner = pl.Trainer(logger=logger, **trainer_params)


    datacls = DataModuleClass(**params)

    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)

