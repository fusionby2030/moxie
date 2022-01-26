from ray import tune 
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from moxie.models.DIVA_ak_1 import DIVAMODEL
from moxie.data.profile_lightning_module import PLDATAMODULE_AK
from moxie.experiments.DIVA_EXP_AK import EXAMPLE_DIVA_EXP_AK

from pathlib import Path
import argparse
def train_model_on_tune(search_space, num_epochs, num_gpus, num_cpus, data_dir='./data/processed/profile_database_v1_psi22.hdf5', pin_memory=False):
    model_params = search_space
    experiment_params ={'LR': 0.001, 'weight_decay': 0.0, 'batch_size': 512}
    if 'LR' in search_space.keys():
        experiment_params['LR'] = search_space['LR']

    data_params = {'data_dir': data_dir,
                    'num_workers': num_cpus,
                    'pin_memory': pin_memory}

    trainer_params = {
        'max_epochs': num_epochs,
        'gpus': 1 if num_gpus > 0.0 else 0,
        'logger': TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version='.'),
        'gradient_clip_val': 0.5,
        'gradient_clip_algorithm':"value",
        'progress_bar_refresh_rate':0,
        'callbacks': [
        TuneReportCallback(
            metrics={
                "loss": "ReconLoss/Valid",
                "KLD_mach": "KLD_mach/Valid",
                "KLD_stoch": "KLD_stoch/Valid",
                'loss_mp': 'ReconLossMP/Valid'
            },
            on="validation_end")
        ]
        }

    generator=torch.Generator().manual_seed(42)
    torch.manual_seed(42)

    datacls = PLDATAMODULE_AK(**data_params)

    model = DIVAMODEL(**model_params)

    experiment = EXAMPLE_DIVA_EXP_AK(model, experiment_params)


    runner = pl.Trainer(**trainer_params)

    runner.fit(experiment, datamodule=datacls)
    # runner.test(experiment, datamodule=datacls)

def tune_asha(num_samples=1, num_epochs=150, gpus_per_trial=0, cpus_per_trial=5, data_dir='/scratch/project_2005083/moxie/data/processed/profile_database_v1_psi22.hdf5', pin_memory=False):
    search_space = {
        # 'LR': tune.loguniform(0.00001, 0.01),
        'mach_latent_dim': tune.grid_search([10, 15, 20, 30, 40, 50]),
        'beta_stoch': tune.grid_search([0.0001, 0.001, 0.01, 0.1, 1., 10. ,100.]),
        'beta_mach': tune.grid_search([1., 10., 100., 1000.]),
        'alpha_prof': tune.grid_search([0., 1., 10., 100., 1000.]),
        'alpha_mach': tune.grid_search([0., 1., 10., 100., 1000.]),
        # 'loss_type': tune.choice(['supervised', 'semi-supervised'])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=50,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["mach_latent_dim", "beta_stoch", "beta_mach", "loss_type", "alpha_mach", "alpha_prof"],
        metric_columns=["loss", "loss_mp"],
        max_report_frequency=100)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_fn_with_parameters = tune.with_parameters(train_model_on_tune,
                                                num_epochs=num_epochs,
                                                num_gpus=gpus_per_trial,
                                                num_cpus=cpus_per_trial,
                                                data_dir=data_dir,
                                                pin_memory=pin_memory)

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        log_to_file=False,
        raise_on_failed_trial=False,
        local_dir= dir_path + '/ray_results',
        name="tune_DIVA_beta",
        fail_fast=True)

    print("Best hyperparameters found were: ", analysis.best_config)

    df = analysis.results_df
    df.to_csv(dir_path + '/DIVA_TE_gridsearch.csv')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search for hyperparams using raytune and HPC.')
    parser.add_argument('-gpu', '--gpus_per_trial', default=0, help='# GPUs per trial')
    parser.add_argument('-cpu', '--cpus_per_trial', default=4, help='# CPUs per trial')
    parser.add_argument('-ep', '--num_epochs', default=200, help='# Epochs to train on')
    parser.add_argument('-pm', '--pin_memory', default=False, help='# Epochs to train on', type=bool)

    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = '8'
    # desired_path = '/home/kitadam/ENR_Sven/moxie_revisited/data/processed/raw_padded_fitted_datasets.pickle'  
    dir_path = Path(__file__).parent.parent.parent
    desired_path = dir_path / 'data' / 'processed' / 'raw_padded_fitted_datasets.pickle'
    print('\n# Path to Dataset Exists? {}'.format(desired_path.exists()))
    print(desired_path.resolve())


    tune_asha(cpus_per_trial=int(args.cpus_per_trial), gpus_per_trial=float(args.gpus_per_trial),  num_epochs=int(args.num_epochs), data_dir=desired_path.resolve(), pin_memory=args.pin_memory)