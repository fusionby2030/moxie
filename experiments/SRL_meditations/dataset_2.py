from typing import Union, List 
import numpy as np 
import pickle 
import pandas as pd 
from dataclasses import dataclass 
from tqdm import tqdm 
import matplotlib.pyplot as plt 

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
                beg_idx = -60
                end_idx = -10
            elif self.device == 'JET': 
                beg_idx = -20
                end_idx = -1
        else: 
            beg_idx = 0
            end_idx = -1
        return np.vstack((1e-19*self.ne[beg_idx:end_idx], self.te[beg_idx:end_idx]))
    def __repr__(self): 
        return (f'{self.__class__.__name__}'f'SHOT: {self.pulse_id}, t={self.time_stamp}')

@dataclass 
class PULSE: 
    device: str 
    pulse_id: int 
    profiles: List[PROFILE]
    def get_time_obervables(self, window_size=2):
        # return a list of profile pairs, one after the other 
        current_set = []
        full_set = []
        it = iter(self.profiles)
        while True: 
            try: 
                while len(current_set) < window_size: 
                    current_set.append(next(it))
                full_set.append(current_set)
                current_set = [current_set[1]]
            except StopIteration: 
                return full_set    

def load_raw_data() -> Union[dict, pd.DataFrame]:
    cols = ['shot', 'time', 'Teped(eV)', 'neped(10^19m^-3)', 'peped(Pa)', 'tewidth(psin)', 'newidth(psin)', 'pewidth(psin)', 'tepos(psin)', 'nepos(psin)', 'pepos(psin)', 'beta_n', 'Ip(MA)', 'Bt(T)', 'Rgeo(m)', 'aminor(m)', 'elongation', 'delta_upp', 'delta_low', 'P_NBI(MW)', 'P_ICRF(MW)', 'P_ECRH(MW)', 'P_TOT(MW)', 'P_RAD(MW)', 'Volume(m^3)', 'q95', 'Rmag(m)', 'Zmag(m)', 'Something(-)']

    with open(AUG_RAW_FILE_LOC, 'rb') as file: 
        AUG_DATA = pickle.load(file)
    AUG_PDB = pd.read_csv(AUG_PDB_FILE_LOC, header=3, usecols=cols)
    with open(JET_RAW_FILE_LOC, 'rb') as file: 
        JET_DATA = pickle.load(file)
    JET_PDB = pd.read_csv(JET_PDB_FILE_LOC) 

    return AUG_DATA, AUG_PDB, JET_DATA, JET_PDB

def save_proccessed_classes(list_of_classes, file_name:str): 
    with open(PERSONAL_DATA_DIR_PROC + file_name, 'wb') as file: 
        pickle.dump(list_of_classes, file)


def create_dataclasses_from_raw(plotting: bool = False): 
    AUG_DATA, AUG_PDB, JET_DATA, JET_PDB = load_raw_data()
    pbar = tqdm(AUG_PDB.iterrows())
    all_profs, all_pulses = [], []
    total_slices = 0
    set_pulses = set() 
    print(len(set(AUG_PDB['shot'].to_list())))
    for index, row in pbar:
        pulse_num, t1 = int(row['shot']), row['time']
        pbar.set_postfix({'#': pulse_num, 't1': t1})
        if not AUG_DATA.get(pulse_num): 
            continue
        if pulse_num in set_pulses: 
            continue
        profile_data, mp_data = AUG_DATA[pulse_num]['ida_profiles'],AUG_DATA[pulse_num]['machine_params']
        if profile_data is None or mp_data is None: 
            continue
        set_pulses.add(pulse_num)
        sample_time, sample_rad,  sample_ne, sample_te, sample_dte, sample_dne = profile_data['time'], profile_data['radius'], profile_data['ne'], profile_data['Te'], profile_data['ne_unc'], profile_data['Te_unc']    
        # Cut bases on current threshold
        current_threshold = 0.6e6 
        current_times, current_data = mp_data['IpiFP']['time'], mp_data['IpiFP']['data']
        
        if np.mean(current_data) <= current_threshold: 
            plotting=True
        
        sample_window = current_data >= current_threshold # np.logical_and(current_data >= 0.8e6, current_data <= 5e6)
        t_ipthresh_1, t_ipthresh_2 = min(current_times[sample_window]), max(current_times[sample_window]) 
        sample_window = np.logical_and(sample_time >= t_ipthresh_1, sample_time <= t_ipthresh_2)
        cut_rad, cut_time, cut_ne, cut_te, cut_dne, cut_dte = sample_rad.T[sample_window], sample_time[sample_window], sample_ne.T[sample_window], sample_te.T[sample_window], sample_dne.T[sample_window], sample_dte.T[sample_window]
        pbar.set_postfix({'#': pulse_num, 't1': t1, '#Slices': len(cut_time)})
        total_slices += len(cut_time)
        sample_pulse_profs = [PROFILE(device='AUG', pulse_id=pulse_num, time_stamp=cut_time[idx], ne=cut_ne[idx], te=cut_te[idx], radius=cut_rad[idx], dne=cut_dne[idx], dte=cut_dte[idx]) for idx, _ in enumerate(cut_time)]
        sample_pulse = PULSE(device='AUG', pulse_id=pulse_num, profiles=sample_pulse_profs)

        all_pulses.append(sample_pulse)
        all_profs.extend(sample_pulse_profs)
        if plotting: 
            fig, axs = plt.subplots()
            axs.plot(cut_rad[0], cut_ne[0], label=f'$t$ = {cut_time[0]}')
            axs.plot(cut_rad[100], cut_ne[100], color='dodgerblue', label=f'$t$ = {cut_time[100]}')
            axs.plot(cut_rad[-1], cut_ne[-1], color='cadetblue', label=f'$t$ = {cut_time[-1]}')
            t_ax = axs.twinx()
            
            t_ax.plot(cut_rad[0], cut_te[0], color='orange')
            t_ax.plot(cut_rad[100], cut_te[100], color='salmon')
            t_ax.plot(cut_rad[-1], cut_te[-1], color='peru')
            axs.legend()
            plt.show()
            plt.axhline(np.mean(current_data))
            plt.plot(current_times, current_data)
            plt.axhline(0.8e6)
            plt.show()
        plotting = False
    print(total_slices, len(set_pulses))
    save_proccessed_classes(all_pulses, file_name='AUG_ALL_PULSES.pickle')
    save_proccessed_classes(all_profs, file_name='AUG_ALL_PROFILES.pickle')

if __name__ == '__main__': 
    create_dataclasses_from_raw()