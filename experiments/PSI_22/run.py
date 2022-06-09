from argparse import ArgumentParser
import sys, os, pathlib

from moxie.models.PSI_model_ak1 import PSI_MODEL
from moxie.data.profile_lightning_module_psi import PLDATAMODULE_AK
from moxie.experiments.PSI_LIGHTNING_EXP import PSI_EXP

import torch
import pytorch_lightning as pl

from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')


def main(model_hyperparams, args):
    pl.utilities.seed.seed_everything(42)

    STATIC_PARAMS = {'out_length': 20, 'elm_style_choice': 'simple', 'data_dir': args.data_path}
    model_hyperparams = {**model_hyperparams, **STATIC_PARAMS}


    TRAINING_HYPERPARAMS = {'LR': 0.003, 'weight_decay': 0.0, 'batch_size':512, 'scheduler_step': 0}

    # 'semi-supervised-start', 'semi-supervsied-cutoff', 'supervised'

    params = {**TRAINING_HYPERPARAMS, **model_hyperparams}

    datacls = PLDATAMODULE_AK(**params)
    # TODO: Grab Sizes of the input/output datasets

    model = PSI_MODEL(**model_hyperparams)
    trainer_params = {'max_epochs': 50, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value'}

    logger = pl.loggers.TensorBoardLogger(exp_path / "tb_logs", name=args.name)

    experiment = PSI_EXP(model, params)
    runner = pl.Trainer(logger=logger, **trainer_params)

    runner.fit(experiment, datamodule=datacls)
    runner.test(experiment, datamodule=datacls)

    model = runner.model.model
    state = {'model': model.state_dict(),
            'MP_norms': runner.datamodule.get_machine_norms(),
            'D_norms': runner.datamodule.get_density_norms(),
            'T_norms': runner.datamodule.get_temperature_norms(),
            'hparams': model_hyperparams,
            }
    model_pth_name = 'modelstatedict_' + args.name + '.pth'
    torch.save(state, exp_path / 'model_results' / model_pth_name)

# {'LR': 0.003, 'mach_latent_dim': 12, 'stoch_latent_dim': 3, 'beta_stoch': 5.0, 'beta_mach_unsup': 0.0843, 'beta_mach_sup': 10.0, 'alpha_prof': 959, 'alpha_mach': 200.0, 'start_sup_time': 500.0, 'physics': False, 'gamma_stored_energy': 0.0, 'encoder_end_dense_size': 128, 'dataset_choice': 'SANDBOX_NO_VARIATIONS', 'scheduler_step': 0.0, 'mp_hdims_aux': [128, 128, 64, 32], 'mp_hdims_cond': [128, 128, 64, 32], 'hidden_dims': [3, 6], 'elm_style_choice': 'simple', 'loss_type': 'semi-supervised-cutoff-increasing'}

if __name__ == '__main__':

    file_path = pathlib.Path(__file__).resolve()
    exp_path = file_path.parent
    home_path = file_path.parent.parent.parent

    dataset_path = home_path / 'data' / 'processed' / f'ML_READY_dict.pickle'# f'ML_READY_dict_{CURRENT_DATE}.pickle'
    parser = ArgumentParser()
    parser.add_argument("--name", '-n',  type=str, default=f'EXAMPLE_{CURRENT_DATE}')
    parser.add_argument('--data_path', type=str, default=dataset_path)
    args = parser.parse_args()
    model_hyperparams = {'mach_latent_dim': 9, 'stoch_latent_dim': 3, # 0.0273
                        'beta_stoch': 1.0, 'beta_mach_unsup': 0.001,'beta_mach_sup':  0.00,
                        'alpha_mach': 100, 'alpha_prof': 350.0,
                        'start_sup_time': 1000, 'scheduler_step': 5000,
                        'physics': True, 'gamma_stored_energy': 10.0, 'gamma_bpol': 100.0, 'gamma_beta': 10.0,
                        'mp_hdims_aux': [40, 40, 40, 40, 40, 40, 32, 16], 'mp_hdims_cond':[40, 40, 40, 40, 40, 40, 32, 16],
                        'hidden_dims': [2, 4], 'loss_type': 'semi-supervised-cutoff',}

    main(model_hyperparams, args)
