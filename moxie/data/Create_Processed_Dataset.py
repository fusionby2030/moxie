"""
author: adam.kit@helsinki.fi

Script assumes the following: 

1. user has pulled the raw dataset from HEIMDALL, and stored it as '../../data/raw/JET_RAW_DATA_DDMMYY.pickle'
2. user has access to jet pedestal database and it is in csv format


This script generates the following files: 

1. A binarized python dictionary which for each pulse stores all the time slices information, i.e., the profiles, machine parameters, and ELM timings with accessible structure: 
    {123456: {'profiles': np.array(shape=(#TIMESLICES, 2, 20)), 
              'profiles_uncert': np.array(shape=(#TIMESLICES, 2, 20)), 
              'profiles_flags': np.array(shape=(#TIMESLICES, 20))}, 
              'rmids_efit': np.array(shape=(#TIMESLICES, 20)), 
              'machine_parameters': np.array(shape=(#TIMESLICES, 13)), 
              'elm_fractions': np.array(shape=(#TIMESLICES,)), 
              'timings': np.array(shape=(#TIMESLICES,))
    123457: {...}, 123458: {...}, ... }
    
    NB: dictionary key is a python int. 
    NB: file is stored in processed under the name 'processed_pulse_dict_DDMMYY.pickle'
    NOTE: This can be used then to do splitting for train-val-test, as mentioned below. 
"""
from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm 

PROCESSED_DATA_DIR = '../../data/processed/' 
RAW_DATA_DIR = '../../data/raw/'

JET_RAW_DATA_LOC = RAW_DATA_DIR + 'JET_RAW_DATA_030622.pickle'
JET_PDB_FILE_LOC = RAW_DATA_DIR + 'jet-pedestal-database.csv'

JET_PDB_MP_COLS = [ 'Ip(MA)', 'B(T)', 'R(m)', 'a(m)', 'averagetriangularity', 'Meff', 'P_NBI(MW)', 'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)', 'q95', 'gasflowrateofmainspecies10^22(e/s)']
JET_PDB_PROF_COLS = ['Tepedheight(keV)', 'nepedheight10^19(m^-3)', 'nesep']
JET_PDB_TIME_COLS = ['FLAG:HRTSdatavalidated', 't1', 't2']

DATASET_MP_NAME_LIST = ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'ELON', 'POHM', 'IPLA', 'BVAC', 'NBI', 'ICRH', 'ELER']

def load_raw_data() -> (dict, pd.DataFrame): 
    with open(JET_RAW_DATA_LOC, 'rb') as file: 
        JET_RAW_DICT = pickle.load(file)
    JET_PDB = pd.read_csv(JET_PDB_FILE_LOC) 
    JET_PDB = JET_PDB[(JET_PDB['FLAG:HRTSdatavalidated'] > 0)]
    return JET_RAW_DICT, JET_PDB

def get_subset_of_jetpdb(pdb: pd.DataFrame) -> pd.DataFrame: 
    SUBSET_PDB = pdb[(pdb['shot'] > 80000) & (pdb['Atomicnumberofseededimpurity'].isin([0, 7])) & (pdb['FLAG:DEUTERIUM'] == 1.0) & (pdb['FLAG:Kicks'] == 0.0) & (pdb['FLAG:RMP'] == 0.0) & (pdb['FLAG:pellets'] == 0.0)]
    return SUBSET_PDB

def get_profile_window_times(pulse_profs: dict, t1: float, t2: float) -> (np.array, np.array):
    sample_times = pulse_profs['NE']['time']
    
    window_mask = np.logical_and(sample_times >= t1, sample_times <= t2)
    window_times = sample_times[window_mask]
    return window_times, window_mask
def get_profile_xaxis(pulse_profs: dict, mask: np.array) -> (np.array, np.array, np.array, np.array): 
    RMID, RHO, PSI = pulse_profs['RMID'], pulse_profs['RHO'], pulse_profs['PSI']
    np.testing.assert_array_equal(RMID['radius'], RHO['radius'])
    return RMID['radius'], RMID['values'][mask], RHO['values'][mask], PSI['values'][mask]


def cut_profiles(pulse_profs: dict, time_window_mask: np.array, rmid: np.array) -> (np.array, np.array,np.array,np.array,np.array,np.array,np.array):
    # Gather the last 20 values from HRTS measurement
    n_profs, t_profs = pulse_profs['NE']['values'][time_window_mask], pulse_profs['TE']['values'][time_window_mask]
    dne_profs, dte_profs = pulse_profs['DNE']['values'][time_window_mask], pulse_profs['DTE']['values'][time_window_mask]
    prof_fails = pulse_profs['FAIL']['values'][time_window_mask]    
    n_cut, t_cut = n_profs[:, -20:], t_profs[:, -20:]
    dn_cut, dt_cut = dne_profs[:, -20:], dte_profs[:, -20:]
    fail_cut = prof_fails[:, -20:]
    together_cut, d_together_cut = np.stack((n_cut, t_cut), 1), np.stack((dn_cut, dt_cut), 1)
    rmid_cut = rmid[:, -20:]
    return n_cut, t_cut, together_cut, rmid_cut, dn_cut, dt_cut, d_together_cut, fail_cut

def cut_controls(pulse_mps: dict, t1: float, t2: float, time_windows: np.array, pulse_num: int) -> np.array: 
    def sample_input(mp_loc: dict, key: str, t1, t2, window_times, index): 
        mp_vals, mp_times = mp_loc[key]['values'], mp_loc[key]['time']
        sampled_vals = np.zeros_like(window_times)
        
        delta_T = 0.05 
        # final_mp_vals = average_machine_with_times(window_times, mp_val, mp_time, key, index)
        if len(mp_vals) == 0: 
            print(f'MP_VALS LEN =  0, {key}, {pulse_num}', mp_vals, mp_times)
        if type(mp_times) == str: 
            # NO ICHR WAS USED! 
            return sampled_vals
        if (mp_vals.sum() == 0.0 or mp_vals[np.logical_and(mp_times > t1, mp_times < t2)].sum() == 0.0) and key=='ICRH': 
            return sampled_vals
        for slice_num, time in enumerate(window_times): 
            window_t2, window_t1 = time + delta_T, time - delta_T
            aggregation_idx = np.logical_and(mp_times >= window_t1, mp_times <= window_t2)
            aggregation_vals = mp_vals[aggregation_idx]
            if len(aggregation_vals) == 0: 
                if key in ['ICRH'] and ((mp_times < time).sum() == 0):
                    continue
            aggregation_mean = aggregation_vals.mean()
            if aggregation_mean < 0 and key not in ['IPLA', 'BVAC']: 
                aggregation_mean = np.nan
            sampled_vals[slice_num] = aggregation_mean
        return sampled_vals

    return np.array([sample_input(pulse_mps, key, t1, t2, time_windows, pulse_num) for key in DATASET_MP_NAME_LIST]).T

def add_to_dict(data_dict: dict, pulse_num: int, **kwargs): 
    if pulse_num not in data_dict.keys(): 
        data_dict[pulse_num] = kwargs
    else: 
        old_length = len(data_dict[pulse_num]['machine_parameters'])
        for key, value in kwargs.items():             
            if key in ['machine_parameters', 'rmids_efit', 'profiles_flags']: 
                data_dict[pulse_num][key] = np.vstack((data_dict[pulse_num][key], value))
            else: 
                data_dict[pulse_num][key] = np.concatenate((data_dict[pulse_num][key], value), 0)
        assert len(data_dict[pulse_num]['machine_parameters']) != old_length
            
def calculate_elm_percent(time_last_elm, time_next_elm, hrts_time): 
    return (hrts_time - time_last_elm) / (time_next_elm - time_last_elm)

def get_elm_timings(pulse_elms, time_windows): 
    elm_timings = np.empty(len(time_windows))
    elm_timings[:] = np.nan
    if pulse_elms[0] != 'NO ELM TIMINGS':
        no_elm_idx = np.where(pulse_elms == 'NO ELM TIMINGS')
        if len(no_elm_idx[0]) != 0: 
            pulse_elms = np.delete(pulse_elms, no_elm_idx[0])
            pulse_elms = pulse_elms.astype(float)
        for et, time in enumerate(time_windows):
            try: 
                diff = pulse_elms - time
            except Exception as e: 
                print(e)
                print(pulse_elms, time_windows)
            try:
                time_last_elm = pulse_elms[diff < 0][-1]
                time_next_elm = pulse_elms[diff > 0][0]
            except IndexError as e:
                # print('Outside of ELM timings', time, pulse_elm_timings_frass)
                continue
            else: 
                slice_elm_percent = calculate_elm_percent(time_next_elm, time_last_elm, time)
                elm_timings[et] = slice_elm_percent
    else: 
        pass 
    return elm_timings
    
def main() -> dict: 
    JET_RAW_DICT, JET_PDB = load_raw_data()
    JET_PROC_DICT = {}
    # IMPLEMENT: Get train-val-test split
    pbar = tqdm(JET_PDB.iterrows())
    total_slices, total_nan, total_elm_lost = 0, 0, 0
    for index, row in pbar: 
        pulse_num, t1, t2 = int(row['shot']), row['t1'], row['t2']
        if  pulse_num < 80000 or pulse_num in [95008, 95009, 95012]: # The last three pulses come from NBI failures...
            continue
        pulse_mps, pulse_profs, pulse_elms = JET_RAW_DICT[str(pulse_num)]['inputs'], JET_RAW_DICT[str(pulse_num)]['outputs'],JET_RAW_DICT[str(pulse_num)]['elms'] 
        time_windows, time_window_mask = get_profile_window_times(pulse_profs, t1, t2)
        pbar.set_postfix({'#': pulse_num, 't1': t1, 't2': t2, '#slices': len(time_windows)})
        HRTS_RADIUS, RMID, RHO, PSI = get_profile_xaxis(pulse_profs, time_window_mask)
        N_CUT, T_CUT, TOGETHER_CUT, RMID_CUT, DN_CUT, DT_CUT, D_TOGETHER_CUT, FAIL_CUT = cut_profiles(pulse_profs, time_window_mask, RMID)
        
        CONTROLS = cut_controls(pulse_mps, t1, t2, time_windows, pulse_num)      
        ELM_FRACS = get_elm_timings(pulse_elms, time_windows)
        
        add_to_dict(JET_PROC_DICT, pulse_num, machine_parameters=CONTROLS, profiles=TOGETHER_CUT, profiles_uncert=D_TOGETHER_CUT, elm_fractions=ELM_FRACS, rmids_efit=RMID_CUT, fails=FAIL_CUT, timings=time_windows)
        
        total_slices += len(time_windows)
        total_nan += np.isnan(CONTROLS[:, -2]).sum() + np.isnan(CONTROLS[:, -6]).sum()
        total_elm_lost += np.isnan(ELM_FRACS).sum()
    total_pulses = len(JET_PROC_DICT)
    print(f'{total_slices} slices collected from {total_pulses} pulses\nThere are {total_nan} slices with unusable machine parameters\n{total_elm_lost} slices do not have ELM timings')
    for key, value in JET_PROC_DICT[pulse_num].items(): 
        print(key, value.shape)
    return JET_PROC_DICT

if __name__ == '__main__': 
    JET_PROC_DICT = main()
    
    with open(PROCESSED_DATA_DIR + f'processed_pulse_dict_{CURRENT_DATE}.pickle', 'wb') as file: 
        pickle.dump(JET_PROC_DICT, file)
