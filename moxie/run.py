import pytorch_lightning as pl
from data.profile_dataset import DS, DataModuleClass
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from experiments import DIVA_EXP
from models import DIVA_v1, DIVA_v2
from pathlib import Path
import argparse
import os

def train_model(data_dir='/home/adam/ENR_Sven/moxie/data/processed/profile_database_v1_psi22.hdf5',
                cpus_per_trial=8, gpus_per_trial=1, name='STANDALONE', num_epochs=350, pin_memory=False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    generator=torch.Generator().manual_seed(42)
    torch.manual_seed(42)
    logger = TensorBoardLogger("tb_logs", name=name)

    STATIC_PARAMS = {'data_dir':data_dir,
                    'num_workers': cpus_per_trial,
                    'pin_memory': pin_memory}
    HYPERPARAMS = {'LR': 0.0001174, 'weight_decay': 0.0, 'batch_size': 512}

    # from models.VAE import VanillaVAE

    # 14 |  0.000193631 |       13930
    # 25, 'beta_stoch': 0.0013071927935371294, 'beta_mach': 240, 'loss_type': 'supervised', 'alpha_mach': 8.787106145338122, 'alpha_prof': 4.407600476836661}
    #  {'LR': 0.004131252321597796, 'mach_latent_dim': 39, 'beta_stoch': 0.09321524399430335, 'beta_mach': 60, 'alpha_prof': 172.01358801832427, 'alpha_mach': 19.22641567358799, 'loss_type': 'supervised'}
    #    17 |           15 |           6 | supervised      |   175.367    |     121.05   | 0.00019844  | 0.00270345 |  0.731794
    model_hyperparams = {'in_ch': 2, 'out_dim':63,
                            'mach_latent_dim': 25, 'beta_stoch': .00463757, 'beta_mach':  1000.,
                            'alpha_mach': 10, 'alpha_prof': 1.,
                        'loss_type': 'supervised'}

    params = {**STATIC_PARAMS, **HYPERPARAMS, **model_hyperparams}
    model = DIVA_v2(**model_hyperparams)
    trainer_params = {'max_epochs': num_epochs,  'gpus': gpus_per_trial if str(device).startswith('cuda') else 0,
                    'gradient_clip_val': 0.5, 'gradient_clip_algorithm':"value",
                    'profiler':"simple"}


    experiment = DIVA_EXP(model, params)
    runner = pl.Trainer(logger=logger, **trainer_params)


    datacls = DataModuleClass(**params)

    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search for hyperparams using raytune and HPC.')
    parser.add_argument('-gpu', '--gpus_per_trial', default=0, help='# GPUs per trial')
    parser.add_argument('-cpu', '--cpus_per_trial', default=8, help='# CPUs per trial')
    parser.add_argument('-name', '--experiment_name', default='STANDALONE', help='What is the name of the experiment? i.e., how will it be logged under')
    parser.add_argument('-ep', '--num_epochs', default=2000, help='# Epochs to train on')
    parser.add_argument('-pm', '--pin_memory', default=False, help='# Epochs to train on', type=bool)
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"
    # os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = '8'

    dir_path = Path(__file__).parent
    desired_path = dir_path.parent
    desired_path = desired_path / 'data' / 'processed' / 'profile_database_v1_psi22.hdf5'
    print('\n# Path to Dataset Exists? {}'.format(desired_path.exists()))
    print(desired_path.resolve())
    train_model(data_dir=desired_path.resolve(), cpus_per_trial=int(args.cpus_per_trial), gpus_per_trial=int(args.gpus_per_trial), name=args.experiment_name, num_epochs=args.num_epochs, pin_memory=args.pin_memory)
