import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys, os, pathlib

from moxie.models.PSI_model_ak1 import PSI_MODEL

import torch
import pytorch_lightning as pl

from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')

def main(args):
    exp_dict = torch.load(f'./model_results/modelstatedict_{args.name}.pth')
    state_dict, (MP_mu, MP_var), (D_mu, D_var), (T_mu, T_var), model_hyperparams = exp_dict.values()
    model = PSI_MODEL(**model_hyperparams)
    model.load_state_dict(state_dict)

def plot_1():
    pass
    
if __name__ == '__main__':
    file_path = pathlib.Path(__file__).resolve()# .parent.parent
    # Path of experiment
    exp_path = file_path.parent
    # Path of moxie stuffs
    home_path = file_path.parent.parent.parent
    # Path to data
    dataset_path = home_path / 'data' / 'processed' / f'ML_READY_dict.pickle'# f'ML_READY_dict_{CURRENT_DATE}.pickle'
    parser = ArgumentParser()
    parser.add_argument("--name", '-n',  type=str, default=f'EXAMPLE_{CURRENT_DATE}')
    parser.add_argument('--data_path', type=str, default=dataset_path)
    args = parser.parse_args()

    main(args)
