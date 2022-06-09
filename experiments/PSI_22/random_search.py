from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import sys, os

from moxie.models.PSI_model_ak1 import PSI_MODEL
from moxie.data.profile_lightning_module_psi import PLDATAMODULE_AK
from moxie.experiments.PSI_LIGHTNING_EXP import PSI_EXP

from pathlib import Path
import argparse
"""
'mp_hdims_aux': tune.choice([[tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],
                                    [tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],
                                    [tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],
                                    [tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],]),
        'mp_hdims_cond': tune.choice([[tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],
                                    [tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],
                                    [tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],
                                    [tune.randint(16, 500), tune.randint(16, 500), tune.randint(16, 500)],])"""
def train_model_on_tune(search_space, num_epochs, num_gpus, num_cpus, data_dir='', pin_memory=False):
    model_params = search_space
    experiment_params ={'LR': 0.003, 'weight_decay': 0.0, 'batch_size': 512}
    if 'LR' in search_space.keys():
        experiment_params['LR'] = search_space['LR']
        experiment_params['physics'] = search_space['physics']
        experiment_params['start_sup_time'] = search_space['start_sup_time']
        experiment_params['scheduler_step'] = search_space['scheduler_step']

    data_params = {'data_dir': data_dir,
                    'num_workers': num_cpus,
                    'pin_memory': pin_memory,
                    'dataset_choice': search_space['dataset_choice']}

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

    pl.utilities.seed.seed_everything(42)
    # generator=torch.Generator().manual_seed(42)
    # torch.manual_seed(42)

    datacls = PLDATAMODULE_AK(**data_params)

    model = PSI_MODEL(**model_params)

    experiment = PSI_EXP(model, experiment_params)


    runner = pl.Trainer(**trainer_params)

    runner.fit(experiment, datamodule=datacls)
    # runner.test(experiment, datamodule=datacls)

def tune_asha(num_samples=500, num_epochs=50, gpus_per_trial=0, cpus_per_trial=5, data_dir='', pin_memory=False, experiment_name='NEW_CONSTRAINTS'):
    search_space = {
        'LR': tune.loguniform(0.00001, 0.01),
        'mach_latent_dim': tune.choice([6, 7, 8, 9, 10, 11, 12]),
        'stoch_latent_dim': 3, # tune.choice([3, 4, 5]),
        'beta_stoch': 5.0, # tune.qloguniform(0.005,10, 0.001),
        'beta_mach_unsup': tune.qloguniform(0.0001, 0.1, 0.0001),  # tune.qloguniform(0.005, 10, 0.001),
        'beta_mach_sup':  10.0, # tune.choice([0.0, 1.0, tune.qloguniform(0.01, 2.01, 0.01)]),
        "alpha_prof": tune.randint(100, 1000),
        "alpha_mach": 200., # tune.randint(100, 500),
        "start_sup_time": 500., # tune.randint(0, 2000),
        'physics': False, # tune.choice([True, False]),
        'gamma_stored_energy': 0.0, # tune.qloguniform(0.01, 2, 0.01),
        'encoder_end_dense_size': 128, # 128,
        'dataset_choice': 'SANDBOX_NO_VARIATIONS',
        'scheduler_step': 0.0,
        'mp_hdims_aux': [40, 40, 40, 40, 40, 40, 32, 16], 'mp_hdims_cond':[40, 40, 40, 40, 40, 40, 32, 16],
        'hidden_dims': [2, 4],
        'elm_style_choice': 'simple',
        'loss_type': 'semi-supervised-cutoff',}


    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=35,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["beta_mach_unsup", "alpha_prof", "mach_latent_dim"],
        metric_columns=["loss", "loss_mp", 'KLD_stoch', 'KLD_mach'],
        max_report_frequency=10,
        max_progress_rows=30,
        sort_by_metric=True)

    # dir_path = os.path.dirname(os.path.realpath(__file__))
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
        local_dir= exp_path  / 'ray_results',
        name=experiment_name,
        fail_fast=False)

    print("Best hyperparameters found were: ", analysis.best_config)

    df = analysis.results_df
    experiment_name += '.csv'
    df.to_csv(exp_path / experiment_name )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search for hyperparams using raytune and HPC.')
    parser.add_argument('-gpu', '--gpus_per_trial', default=0, help='# GPUs per trial')
    parser.add_argument('-cpu', '--cpus_per_trial', default=2, help='# CPUs per trial')
    parser.add_argument('-ep', '--num_epochs', default=50, help='# Epochs to train on')
    parser.add_argument('-pm', '--pin_memory', default=False, help='If using GPUs, should try to set this to True', type=bool)
    parser.add_argument('-en', '--experiment_name', default="DEFAULT", help='Name Experiments')
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = '8'
    # desired_path = '/home/kitadam/ENR_Sven/moxie_revisited/data/processed/raw_padded_fitted_datasets.pickle'
    # pedestal_profiles_ML_READY_ak_09022022
    file_path = Path(__file__).resolve()
    exp_path = file_path.parent
    home_path = exp_path.parent.parent
    dataset_path = home_path / 'data' / 'processed' / f'ML_READY_dict.pickle'# f'ML_READY_dict_{CURRENT_DATE}.pickle'
    # desired_path = home_path / 'data' / 'processed' / 'cleaned_ml_ready_dict_180622.pickle'

    tune_asha(cpus_per_trial=int(args.cpus_per_trial), gpus_per_trial=float(args.gpus_per_trial),  num_epochs=int(args.num_epochs), data_dir=dataset_path.resolve(), pin_memory=args.pin_memory, experiment_name=args.experiment_name)
