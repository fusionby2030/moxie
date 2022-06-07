"""
author: adam.kit@helsinki.fi

Script assumes the following:

1. User  has the proccessed pulse dict '../../data/processed/processed_pulse_dict_{CURRENT_DATE}.pickle' (via ./Create_Proccessed_Dataset.py)
    RELEVANT KEYS: 'profiles': np.array(shape=(#TIMESLICES, 2, 20)),
                   'profiles_uncert': np.array(shape=(#TIMESLICES, 2, 20)),
                   'profiles_flags': np.array(shape=(#TIMESLICES, 20))},
                   'rmids_efit': np.array(shape=(#TIMESLICES, 20)),
                   'machine_parameters': np.array(shape=(#TIMESLICES, 13)),
                   'elm_fractions': np.array(shape=(#TIMESLICES,)),
                   'timings': np.array(shape=(#TIMESLICES,))
2. user has access to jet pedestal database and it is in csv format

Implement dataclasses? maybe later

This script outputs the following files:

1.) A ML ready dictionary to be passed to model training (ML_READY_DICT_{CURRENT_DATE}.pickle) with following structure
    {'train': {'profiles': np.array(shape(#Num Slices, 2, 20))}

"""


from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
PROCESSED_DATA_DIR = '../../data/processed/'
RAW_DATA_DIR = '../../data/raw/'

JET_PROC_FILE_LOC = PROCESSED_DATA_DIR + f'processed_pulse_dict_{CURRENT_DATE}.pickle'
JET_PDB_FILE_LOC = '/home/mn2596/JETPEDESTAL_ANALYSIS/moxie_profile_project/final_data/processed/jet-pedestal-database.csv'

JET_PDB_MP_COLS = [ 'Ip(MA)', 'B(T)', 'R(m)', 'a(m)', 'averagetriangularity', 'Meff', 'P_NBI(MW)', 'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)', 'q95', 'gasflowrateofmainspecies10^22(e/s)']
JET_PDB_PROF_COLS = ['Tepedheight(keV)', 'nepedheight10^19(m^-3)', 'nesep']
JET_PDB_TIME_COLS = ['FLAG:HRTSdatavalidated', 't1', 't2']

DATASET_MP_NAME_LIST = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'ELON', 'POHM', 'IPLA', 'BVAC', 'NBI', 'ICRH', 'ELER']

def load_proc_data() -> (dict, pd.DataFrame):
    with open(JET_PROC_FILE_LOC, 'rb') as file:
        JET_PROC_DICT = pickle.load(file)
    JET_PDB = pd.read_csv(JET_PDB_FILE_LOC)
    JET_PDB = JET_PDB[(JET_PDB['FLAG:HRTSdatavalidated'] > 0) & (JET_PDB['shot'] > 80000) & (~(JET_PDB['shot'].isin([95008, 95009, 95012])))]
    return JET_PROC_DICT, JET_PDB

def get_train_val_test_idx(PULSE_DF, random_state=42):
    all_pulse_numbers = list(set(PULSE_DF['shot'].astype(int).to_list()))
    TEST_SIZE = 0.1
    VAL_SIZE = 0.2
    TRIAN_SIZE = 0.7
    train_idx, test_idx = train_test_split(all_pulse_numbers, test_size=TEST_SIZE, train_size=None, random_state=random_state)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SIZE, train_size=None, random_state=random_state)
    return train_idx, val_idx, test_idx

def make_subset_from_processed(processed_dict, pulse_numbers, dataset_name):
    set_dict_list = {'profiles': [],'profiles_uncert': [], 'rmids_efit': [], 'machine_parameters': [], 'elm_fractions': [], 'timings': []}
    set_dict = {}
    total_removed = 0
    for pulse_num in pulse_numbers:
        pulse_dict = processed_dict[pulse_num]
        remove_idx_where_elm_is_nan = ~np.isnan(pulse_dict['elm_fractions'])
        remove_idx_where_mp_is_nan = ~np.isnan(pulse_dict['machine_parameters']).any(axis=1)
        remove_idx = np.logical_and(remove_idx_where_mp_is_nan, remove_idx_where_elm_is_nan)
        total_removed += (np.invert(remove_idx).sum())
        for key, pulse_value in pulse_dict.items():
            if key == 'fails':
                continue
            elif key == 'timings':
                set_dict_list[key].append(np.array([f'{pulse_num}/{time}' for time in pulse_value])[remove_idx])
            else:
                set_dict_list[key].append(pulse_value[remove_idx])

    for key, set_list in set_dict_list.items():
        set_dict[key] = np.concatenate(set_list, 0)
        if key == 'profiles_uncert':
            dte_too_big_idx = ~(set_dict[key][:, 1, :] > 3000)
            dne_too_big_idx = ~(set_dict[key][:, 0, :] > 1e20)
            mask = np.logical_and(dne_too_big_idx, dte_too_big_idx)
            HRTS_points_removed = np.invert(mask).sum()
            set_dict['profiles_mask'] = mask
    print('{}: Total slices: {}, '.format(dataset_name, len(set_dict[key])), f'Total slices removed: {total_removed} HRTS Points masked: {HRTS_points_removed}')
    return set_dict

def main():
    JET_PROC_DICT, JET_PDB = load_proc_data()

    train_idx, val_idx, test_idx = get_train_val_test_idx(JET_PDB)
    total_slice_num = len(train_idx) + len(val_idx)+ len(test_idx)
    print(f'Total pulses: {total_slice_num}: ' + '# Train: {}, # Valid: {}, # Test: {}'.format(len(train_idx), len(val_idx), len(test_idx)))

    train_dict = make_subset_from_processed(JET_PROC_DICT, train_idx, 'train')
    val_dict = make_subset_from_processed(JET_PROC_DICT, val_idx, 'val')
    test_dict = make_subset_from_processed(JET_PROC_DICT, test_idx, 'test')

    full_dict = {'train': train_dict, 'val': val_dict, 'test': test_dict}

    return full_dict


if __name__ == '__main__':
    print('Creating ML ready python dictionary \n')
    ML_READY_DICT = main()

    with open(PROCESSED_DATA_DIR + f'ML_READY_dict_{CURRENT_DATE}.pickle', 'wb') as file:
        pickle.dump(ML_READY_DICT, file)
    print('\nSaved ML ready dict to: {}'.format(PROCESSED_DATA_DIR + f'ML_READY_dict_{CURRENT_DATE}.pickle'))
