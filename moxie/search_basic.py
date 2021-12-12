import pytorch_lightning as pl
# from experiment import VAExperiment, DualVAExperiment
from data.profile_dataset import DS, DataModuleClass
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import VisualizeBetaVAE
from experiments import BasicExperiment

from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import loguniform

from models import BetaGammaVAE, VisualizeBetaVAE, DualVAE, DualEncoderVAE
from experiments import DualVAExperiment, BasicExperiment
import os

def train_model(param_dict):
    model_params = param_dict['MODEL_PARAMS']
    trainer_params = param_dict['TRAINER_PARAMS']
    experiment_params = param_dict['EXPERIMENT_PARAMS']
    data_params = param_dict['DATA_PARAMS']

    generator=torch.Generator().manual_seed(42)
    torch.manual_seed(42)

    datacls = DataModuleClass(**data_params)

    logger = TensorBoardLogger("tb_logs", name="Search_4_Beta")

    model = VisualizeBetaVAE(**model_params)

    experiment = BasicExperiment(model, experiment_params)

    runner = pl.Trainer(logger=logger, **trainer_params)




    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.suggest.basic_variant import BasicVariantGenerator

def train_model_on_tune(search_space, num_epochs, num_gpus, num_cpus):
    model_params = search_space
    # 1 if str(device).startswith('cuda') else 0
    experiment_params ={'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}
    data_params = {'data_dir': '/home/kitadam/ENR_Sven/moxie/data/processed/profile_database_v1_psi22.hdf5',
                    'num_workers': num_cpus}

    trainer_params = {
        'max_epochs': num_epochs,
        'gpus': num_gpus,
        'logger': TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version='.'),
        'progress_bar_refresh_rate':0,
        'callbacks': [
        TuneReportCallback(
            metrics={
                "loss": "ReconLoss/valid",
                "KLD_loss": "KLD/valid"
            },
            on="validation_end")
        ]
        }

    generator=torch.Generator().manual_seed(42)
    torch.manual_seed(42)

    datacls = DataModuleClass(**data_params)

    model = VisualizeBetaVAE(**model_params)

    experiment = BasicExperiment(model, experiment_params)


    runner = pl.Trainer(**trainer_params)

    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)


def train_model_on_tune_checkpoint(search_space, num_epochs, num_gpus, num_cpus, checkpoint_dir=None):
    model_params = search_space
    # 1 if str(device).startswith('cuda') else 0
    experiment_params ={'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}
    data_params = {'data_dir': '/home/kitadam/ENR_Sven/moxie/data/processed/profile_database_v1_psi22.hdf5',
                    'num_workers': num_cpus}

    trainer_params = {
        'max_epochs': num_epochs,
        'gpus': num_gpus,
        'logger': TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version='.'),
        'progress_bar_refresh_rate':0,
        'callbacks': [
        TuneReportCheckpointCallback(
            metrics={
                "loss": "ReconLoss/valid",
                "KLD_loss": "KLD/valid"
            },
            filename="checkpoint",
            on="validation_end")
        ]
        }

    if checkpoint_dir:
        trainer_params['resume_from_checkpoint'] = os.path.join(checkpoint_dir, "checkpoint")

    generator=torch.Generator().manual_seed(42)
    torch.manual_seed(42)

    datacls = DataModuleClass(**data_params)

    model = VisualizeBetaVAE(**model_params)

    experiment = BasicExperiment(model, experiment_params)


    runner = pl.Trainer(**trainer_params)

    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)


def tune_random(num_samples=100, num_epochs=300, gpus_per_trial=0, cpus_per_trial=2):
    search_space = {
        'latent_dim': tune.choice([3, 4, 5, 6, 7, 8]),
        'beta': tune.loguniform(1e-7, 1),
        'num_conv_blocks': tune.choice([1, 2, 3]),
        'num_trans_conv_blocks': tune.choice([1, 2, 3]),
        'channel_1_size': tune.choice([2, 3, 4, 5, 6, 7, 8]),
        'channel_2_size': tune.choice([2, 3, 4, 5, 6, 7, 8])
    }


    search_alg = BasicVariantGenerator(
        points_to_evaluate=[
            {'latent_dim': 4, 'num_conv_blocks': 1, 'num_trans_conv_blocks': 1},
            {'num_trans_conv_blocks': 1, 'beta': 0.000005},
            {'latent_dim': 5}
        ]
    )
    reporter = CLIReporter(
        parameter_columns=["latent_dim", "beta", "num_conv_blocks", "num_trans_conv_blocks"],
        metric_columns=["loss", "KLD_loss", "training_iteration"])


    train_fn_with_parameters = tune.with_parameters(train_model_on_tune,
                                                num_epochs=num_epochs,
                                                num_gpus=gpus_per_trial,
                                                num_cpus=cpus_per_trial)

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=num_samples,
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tune_random_trial")

    print("Best hyperparameters found were: ", analysis.best_config)

def tune_pbt(num_samples=10, num_epochs=300, gpus_per_trial=0, cpus_per_trial=2):
    search_space = {
        'latent_dim': tune.choice([3, 4, 5, 6, 7, 8]),
        'beta': 0.000001,
        'num_conv_blocks': tune.choice([1, 2, 3]),
        'num_trans_conv_blocks': tune.choice([1, 2, 3]),
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "beta": tune.loguniform(1e-6, 1),
        })

    reporter = CLIReporter(
        parameter_columns=["latent_dim", "beta", "num_conv_blocks", "num_trans_conv_blocks"],
        metric_columns=["loss", "KLD_loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_model_on_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            num_cpus=cpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_pbt_v2")

    print("Best hyperparameters found were: ", analysis.best_config)


def tune_asha(num_samples=10, num_epochs=200, gpus_per_trial=0, cpus_per_trial=2):

    search_space = {
        'latent_dim': tune.choice([3, 4, 5, 6, 7, 8]),
        'beta': tune.loguniform(1e-6, 1),
        'num_conv_blocks': tune.choice([1, 2, 3]),
        'num_trans_conv_blocks': tune.choice([1, 2, 3]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["latent_dim", "beta", "num_conv_blocks", "num_trans_conv_blocks"],
        metric_columns=["loss", "KLD_loss", "training_iteration"])


    train_fn_with_parameters = tune.with_parameters(train_model_on_tune,
                                                num_epochs=num_epochs,
                                                num_gpus=gpus_per_trial,
                                                num_cpus=cpus_per_trial)

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_asha_trial")

    print("Best hyperparameters found were: ", analysis.best_config)



if __name__ == '__main__':
    tune_random()
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATA_PARAMS = {'data_dir': '/home/kitadam/ENR_Sven/moxie/data/processed/profile_database_v1_psi22.hdf5',
                    'num_workers': 4}
    EXPERIMENT_PARAMS = {'LR': 0.0001, 'weight_decay': 0.0, 'batch_size': 512}
    MODEL_PARAMS = {'in_ch': 1, 'out_dim':63, 'hidden_dims': [2, 4],
                        'latent_dim':4,
                        'beta': 0.0005, 'gamma': 300000000000.0, 'loss_type': 'B',
                        'num_conv_blocks': 2, 'num_trans_conv_blocks': 1,}
    TRAINER_PARAMS = {'max_epochs': 400, 'gpus': 1 if str(device).startswith('cuda') else 0}


    param_grid = {'beta': loguniform(0.000001, 1.0)}

    param_list = list(ParameterSampler(param_grid, n_iter=100, random_state=41))

    for param in param_list:
        print(param)
        MODEL_PARAMS.update(param)
        print(MODEL_PARAMS)
        all_params = {'DATA_PARAMS': {**DATA_PARAMS, **EXPERIMENT_PARAMS}, 'EXPERIMENT_PARAMS': EXPERIMENT_PARAMS, 'MODEL_PARAMS': MODEL_PARAMS, 'TRAINER_PARAMS': TRAINER_PARAMS}
        main(all_params)
    """
