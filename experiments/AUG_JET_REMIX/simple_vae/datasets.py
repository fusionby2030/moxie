"""
1. Create datasets from raw data
    - Use python dataclasses
2. 
"""

from typing import Dict, List, Union
from dataclasses import dataclass
import numpy as np 
import pickle 
import pandas as pd 
from tqdm import tqdm

PERSONAL_DATA_DIR = '/home/kitadam/ENR_Sven/moxie/data/raw/'
PERSONAL_DATA_DIR_PROC = '/home/kitadam/ENR_Sven/moxie/data/processed/'

AUG_RAW_FILE_LOC = PERSONAL_DATA_DIR + 'AUG_pdb_pulses_raw_data.pickle'
AUG_PDB_FILE_LOC = PERSONAL_DATA_DIR + 'AUG_PDB_CSV.csv'

JET_RAW_FILE_LOC = PERSONAL_DATA_DIR + 'JET_pdb_pulses_raw_data.pickle' # 'JET_pdb_pulses_raw_with_bo.pickle' #'JET_pdb_pulses_raw_data.pickle'
JET_PDB_FILE_LOC = PERSONAL_DATA_DIR + 'jet-pedestal-database.csv'


@dataclass 
class PROFILE: 
    device: str 
    pulse_id: int 
    time_stamp: float 
    ne: Union[List[float], np.array]
    te: Union[List[float], np.array]
    dne: Union[List[float], np.array]
    dte: Union[List[float], np.array]
    radius: Union[List[float], np.array]
    elm_frac: float = None
    def get_ML_ready_array(self, from_SOL: bool = True) -> np.array:
        if from_SOL: 
            if self.device == 'AUG':
                beg_idx = -50
            elif self.device == 'JET': 
                beg_idx = -20
        else: 
            beg_idx = 0
        return np.vstack((self.ne[beg_idx:], self.te[beg_idx:]))

def load_raw_data(): 
    cols = ['shot', 'time', 'Teped(eV)', 'neped(10^19m^-3)', 'peped(Pa)', 'tewidth(psin)', 'newidth(psin)', 'pewidth(psin)', 'tepos(psin)', 'nepos(psin)', 'pepos(psin)', 'beta_n', 'Ip(MA)', 'Bt(T)', 'Rgeo(m)', 'aminor(m)', 'elongation', 'delta_upp', 'delta_low', 'P_NBI(MW)', 'P_ICRF(MW)', 'P_ECRH(MW)', 'P_TOT(MW)', 'P_RAD(MW)', 'Volume(m^3)', 'q95', 'Rmag(m)', 'Zmag(m)', 'Something(-)']
    with open(AUG_RAW_FILE_LOC, 'rb') as file: 
        AUG_DATA = pickle.load(file)
    AUG_PDB = pd.read_csv(AUG_PDB_FILE_LOC, header=3, usecols=cols)
    with open(JET_RAW_FILE_LOC, 'rb') as file: 
        JET_DATA = pickle.load(file)
    JET_PDB = pd.read_csv(JET_PDB_FILE_LOC) 


    return AUG_DATA, AUG_PDB, JET_DATA, JET_PDB

def save_proccessed_classes(list_of_classes): 
    with open('./profile_data_classses.pickle', 'wb') as file: 
        pickle.dump(list_of_classes, file)

def create_dataclasses_from_raw(): 
    AUG_DATA, AUG_PDB, JET_DATA, JET_PDB = load_raw_data()

    pbar = tqdm(AUG_PDB.iterrows())
    total_slices = 0
    total_pulses = 0
    unique_pulses = []

    all_profs = []
    for index, row in pbar:
        pulse_num, t1 = int(row['shot']), row['time']
        pbar.set_postfix({'#': pulse_num, 't1': t1})
        if not AUG_DATA.get(pulse_num): 
            continue
        
        profile_data, mp_data = AUG_DATA[pulse_num]['ida_profiles'],AUG_DATA[pulse_num]['machine_params'] 
        if profile_data is None or mp_data is None: 
            continue
        sample_time, sample_rad,  sample_ne, sample_te, sample_dte, sample_dne = profile_data['time'], profile_data['radius'], profile_data['ne'], profile_data['Te'], profile_data['ne_unc'], profile_data['Te_unc']    
        sample_window = np.logical_and(sample_time > t1 - 0.1, sample_time < t1 + 0.1)
                
        cut_rad, cut_time, cut_ne, cut_te, cut_dne, cut_dte = sample_rad.T[sample_window], sample_time[sample_window], sample_ne.T[sample_window], sample_te.T[sample_window], sample_dne.T[sample_window], sample_dte.T[sample_window]
        pbar.set_postfix({'#': pulse_num, 't1': t1, '#Slices': sample_window.sum()})
        total_slices += sample_window.sum()
        total_pulses += 1
        unique_pulses.append(pulse_num)
        sample_pulse_profs = [PROFILE(device='AUG', pulse_id=pulse_num, time_stamp=cut_time[idx], ne=cut_ne[idx], te=cut_te[idx], radius=cut_rad[idx], dne=cut_dne[idx], dte=cut_dte[idx]) for idx, _ in enumerate(cut_time)]
        
        all_profs.extend(sample_pulse_profs)
    print('Aug', total_pulses, total_slices, len(set(unique_pulses)))
    del sample_pulse_profs, sample_window, sample_time, sample_rad,  sample_ne, sample_te, sample_dte, sample_dne, cut_rad, cut_time, cut_ne, cut_te, cut_dne, cut_dte
    pbar = tqdm(JET_PDB.iterrows())

    for index, row in pbar: 
        pulse_num, t1, t2 = int(row['shot']), row['t1'], row['t2']
        if not JET_DATA.get(str(pulse_num)): 
            continue
        
        pulse_profs = JET_DATA[str(pulse_num)]['outputs']

        sample_time, sample_rad, sample_ne, sample_te, sample_dte, sample_dne = pulse_profs['NE']['time'],  pulse_profs['NE']['radius'], pulse_profs['NE']['values'], pulse_profs['TE']['values'], pulse_profs['DNE']['values'], pulse_profs['DTE']['values']
        sample_window = np.logical_and(sample_time >= t1, sample_time <= t2)
        pbar.set_postfix({'#': pulse_num, 't1': t1, '#Slices': sample_window.sum()})
        cut_rad, cut_time, cut_ne, cut_te, cut_dne, cut_dte = sample_rad, sample_time[sample_window], sample_ne[sample_window], sample_te[sample_window], sample_dne[sample_window], sample_dte[sample_window]
        
        total_slices += sample_window.sum()
        total_pulses += 1

        sample_pulse_profs = [PROFILE(device='JET', pulse_id=pulse_num, time_stamp=cut_time[idx], ne=cut_ne[idx], te=cut_te[idx], radius=cut_rad, dne=cut_dne[idx], dte=cut_dte[idx]) for idx, _ in enumerate(cut_time)]
        all_profs.extend(sample_pulse_profs)
    
    save_proccessed_classes(all_profs)
    print('Together', total_pulses, len(all_profs))
    return all_profs

if __name__ == '__main__': 
    list_profs = create_dataclasses_from_raw()