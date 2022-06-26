"""
Dataclass structuring file. 

Author: adam.kit@helsinki.fi
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd 
import pickle
from tqdm import tqdm 

PERSONAL_DATA_DIR = '/home/kitadam/ENR_Sven/moxie/data/raw/'
PERSONAL_DATA_DIR_PROC = '/home/kitadam/ENR_Sven/moxie/data/processed/'

PICKLED_AUG_PULSES = PERSONAL_DATA_DIR_PROC + 'AUG_PDB_PYTHON_PULSES.pickle'

AUG_RAW_FILE_LOC = PERSONAL_DATA_DIR + 'AUG_pdb_pulses_raw_data.pickle'
AUG_PDB_FILE_LOC = PERSONAL_DATA_DIR + 'AUG_PDB_CSV.csv'

JET_RAW_FILE_LOC = PERSONAL_DATA_DIR + 'JET_pdb_pulses_raw_data.pickle' # 'JET_pdb_pulses_raw_with_bo.pickle' #'JET_pdb_pulses_raw_data.pickle'
JET_PDB_FILE_LOC = PERSONAL_DATA_DIR + 'jet-pedestal-database.csv'
REL_AUG_MP_COLS = ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'P_OH', 'PICR_TOT', 'PECR_TOT', 'P_TOT', 'SHINE_TH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']

@dataclass 
class PROFILE: 
    """ 
    This is the HRTS/IDA profile for a given time step. 
    """
    time_stamp: float 
    ne: Union[List[float], np.array]
    te: Union[List[float], np.array]
    dne: Union[List[float], np.array]
    dte: Union[List[float], np.array]
    radius: Union[List[float], np.array]
    SOL_IDX: int = field(init=False, default=None)
    def __post_init__(self):
        self.SOL_IDX = np.searchsorted(self.radius, 1.15, side='left')

    def get_density_and_temperature_array(self,  from_SOL: bool=True) -> np.array:
        if from_SOL:     
            if self.SOL_IDX is None: 
                self.SOL_IDX = np.searchsorted(self.radius, 1.15, side='left')
            beg_idx = self.SOL_IDX - 75 # TODO: Should have some variable to get number of points! 
            return np.vstack((self.ne[beg_idx:self.SOL_IDX], self.te[beg_idx:self.SOL_IDX]))
        else: 
            return np.vstack((self.ne, self.te))
    def get_radius_array(self, from_SOL: bool=True) -> np.array:
        beg_idx = self.SOL_IDX - 75 # TODO: Should have some variable to get number of points! 
        return self.radius[beg_idx:self.SOL_IDX]

    def __repr__(self) -> str:
        pass

@dataclass 
class MACHINEPARAMETER: 
    """
    This is the machine parameter value for an entire pulse. 
    """
    name: str 
    data: Union[np.array, List]
    times: Union[np.array, List]

    def __repr__(self) -> str:
        pass

@dataclass 
class ML_ENTRY:
    """
    This is a combination of the machine parameters + profiles for a given time step 
    """
    time_stamp: float 
    profile: PROFILE
    mps: Union[List[float], np.array]
    mps_labels: List[str]

    def get_ml_stuff(self) -> Tuple[np.array, np.array]: 
        return self.profile.get_density_and_temperature_array(), self.mps
    def __repr__(self) -> str:
        pass


@dataclass 
class PULSE: 
    device: str 
    pulse_id: str 
    profiles_raw: List[PROFILE]
    mps_raw: List[MACHINEPARAMETER]
    mps_raw_labels: List[str]
    ml_entries: List[ML_ENTRY] = None
    control_param_labels: List[str] = None
    def plot_raw_time_evolution(self, **kwargs): 
        # TODO: Add able to plot the predicted outputs! 
        import matplotlib.pyplot as plt 
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        profs = np.array([prof.get_density_and_temperature_array() for prof in self.profiles_raw])
        radii = np.array([prof.get_radius_array() for prof in self.profiles_raw])
        t = np.array([prof.time_stamp for prof in self.profiles_raw])
        fig, n_ax = plt.subplots()
        t_ax = n_ax.twinx()
        if kwargs.get('machine_params'): 
            axs, bars, lines, mp_idxs = [], [], [], []
            offset = 0.0
            for axs_idx, param_name  in enumerate(kwargs['machine_params']): 
                mp_idx = self.mps_raw_labels.index(param_name)
                times, vals = self.mps_raw[mp_idx].times, self.mps_raw[mp_idx].data
                axs.append(n_ax.inset_axes([0.75, 0.8 - offset, 0.15, 0.15]))
                offset += 0.2
                axs[axs_idx].plot(times, vals)
                bars.append(axs[axs_idx].axvline(t[0], color='gold'))
                mp_idxs.append(mp_idx)
                if self.ml_entries is not None: 
                    ln, = axs[axs_idx].plot(t[0], self.mapped_mps[0, mp_idx], 'or')
                    lines.append(ln)
                else: 
                    lines = [0]*len(bars)
                axs[axs_idx].axvline(t[0], color='black', ls='--')
                axs[axs_idx].axvline(t[-1], color='black', ls='--')
                axs[axs_idx].set_title(param_name)
            
        xdata, ndata, tdata = [], [], []
        n_ln, = n_ax.plot([], [], 'ro')
        t_ln, = t_ax.plot([], [], 'go')
        
        min_n, max_n = profs[:, 0, :].min(), profs[:, 0, :].max()
        min_t, max_t = profs[:, 1, :].min(), profs[:, 1, :].max()
        
        time_template = 'time = %.3fs'
        time_text = n_ax.text(0.05, 0.9, '', transform=n_ax.transAxes)
        
        n_ax.set_xlim(radii.min(), radii.max())
        t_ax.set_xlim(radii.min(), radii.max())
        n_ax.set_ylim(min_n, max_n)
        t_ax.set_ylim(min_t, max_t)
        n_ax.set_ylabel('$n_e (10^{19} m^{-3})$', color='red', fontsize='x-large')
        t_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
        n_ax.set_xlabel(r'$\Psi_N$')
        n_ax.axvline(1.0, ls='--', color='grey')
        # p_ax.set_ylabel('$T_e$', color='green', fontsize='x-large')
        n_ax.set_title(f'{self.device} SHOT {self.pulse_id}, #Slices: {len(profs)}')
        def animate(i): 
            ndata = profs[i, 0, :]
            tdata = profs[i, 1, :]
            xdata = radii[i] # range(len(profs[i, 0, :]))
            n_ln.set_data(xdata, ndata)
            t_ln.set_data(xdata, tdata)
            for bar, mp_idx in zip(bars, mp_idxs): 
                bar.set_data(t[i], [0, 1e6])
            time_text.set_text(time_template % (t[i]))
            if not isinstance(lines[0], int): 
                for line, mp_idx in zip(lines, mp_idxs): line.set_data(t[i], self.mapped_mps[i, mp_idx]) 
                return n_ln, t_ln, time_text, *bars, *lines
            else: 
                return n_ln, t_ln, time_text, *bars
        ani = FuncAnimation(fig, animate, len(t), interval=(t[1] - t[0])*5000, repeat_delay=1e3, blit=True)
        # writer_video = FFMpegWriter(fps=dt*20000)
        # ani.save('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/full_pulse_train_vid_fast.mp4', dpi=300, writer=writer_video)
        
        plt.show()
        pass 

    def map_mps_to_prof(self, time_stamp: float, window_size: float = 0.002) -> Union[np.array, List]: 
        param_array = np.empty(len(self.control_param_labels))
        param_array[:]  = np.NaN
        for n, key in enumerate(self.control_param_labels): 
            offset = 0.001
            idx = self.mps_raw_labels.index(key)
            mp_val, mp_time = self.mps_raw[idx].data, self.mps_raw[idx].times
            relevant_window = np.logical_and(mp_time >= time_stamp - window_size, mp_time <= time_stamp + window_size)
            window = mp_val[relevant_window]
            while len(window) == 0: 
                relevant_window = np.logical_and(mp_time >= time_stamp - (window_size + offset), mp_time <= time_stamp + (window_size + offset))
                window = mp_val[relevant_window]
                offset += 0.001
            mean_of_param =  np.mean(window)
            param_array[n] = mean_of_param
        return param_array
        

    def setup_ml_entries(self, control_params: List[str], **kwargs): 


        """ 
        Need to match profiles with the corrresponding machine parameters 
        """
        self.ml_entries = []
        all_time_stamps = np.array([prof.time_stamp for prof in self.profiles_raw])
        self.control_param_labels = control_params
        # Check for some threshold to get corresponding windows  
        if kwargs.get('above_threshold'): 
            (param_threshold, val_threshold),  = kwargs['above_threshold'].items()
            mp_idx = self.mps_raw_labels.index(param_threshold)
            rel_param_data, rel_param_time = self.mps_raw[mp_idx].data, self.mps_raw[mp_idx].times
            t1, t2 = min(rel_param_time[rel_param_data >= val_threshold]), max(rel_param_time[rel_param_data >= val_threshold])
            time_window = np.logical_and(all_time_stamps >= t1, all_time_stamps <= t2)
        else: 
            time_window = [True]*len(all_time_stamps)
        
        profiles_in_window = np.asarray(self.profiles_raw, dtype=object)[time_window]
        time_stamps_in_window = all_time_stamps[time_window]

        mps_for_entries = np.empty((len(time_stamps_in_window), len(control_params)))
        # Map machine parameters to the time stamps in profiles raw 
        for k, t in enumerate(time_stamps_in_window): 
            mps_for_entries[k] = self.map_mps_to_prof(t)
            self.ml_entries.append(ML_ENTRY(time_stamp=t, profile=profiles_in_window[k], mps=mps_for_entries[k], mps_labels=control_params))

    def get_ML_ready_array(self): 
        profs, mps = np.empty((len(self.ml_entries), 2, 75)), np.empty((len(self.ml_entries), len(self.control_param_labels)))
        for p, entry in enumerate(self.ml_entries): 
            # _profs, _mps = entry.get_ml_stuff()
            # profs[p], mps[p] = _profs, _mps
            profs[p], mps[p] = entry.get_ml_stuff()
            
        return profs, mps

    def __repr__(self) -> str:
        pass


def load_raw_data() -> Union[dict, pd.DataFrame]:
    cols = ['shot', 'time', 'Teped(eV)', 'neped(10^19m^-3)', 'peped(Pa)', 'tewidth(psin)', 'newidth(psin)', 'pewidth(psin)', 'tepos(psin)', 'nepos(psin)', 'pepos(psin)', 'beta_n', 'Ip(MA)', 'Bt(T)', 'Rgeo(m)', 'aminor(m)', 'elongation', 'delta_upp', 'delta_low', 'P_NBI(MW)', 'P_ICRF(MW)', 'P_ECRH(MW)', 'P_TOT(MW)', 'P_RAD(MW)', 'Volume(m^3)', 'q95', 'Rmag(m)', 'Zmag(m)', 'Something(-)']

    with open(AUG_RAW_FILE_LOC, 'rb') as file: 
        AUG_DATA = pickle.load(file)
    AUG_PDB = pd.read_csv(AUG_PDB_FILE_LOC, header=3, usecols=cols)
    with open(JET_RAW_FILE_LOC, 'rb') as file: 
        JET_DATA = pickle.load(file)
    JET_PDB = pd.read_csv(JET_PDB_FILE_LOC) 

    return AUG_DATA, AUG_PDB, JET_DATA, JET_PDB

def load_classes_from_pickle() -> List[PULSE]: 
    with open(PICKLED_AUG_PULSES, 'rb') as file: 
        AUG_PULSES = pickle.load(file)
    return AUG_PULSES


def create_pulse_classes_from_raw(): 
    AUG_DATA, AUG_PDB, JET_DATA, JET_PDB = load_raw_data()
    set_pulses = set()
    all_pulses = []
    for index, row in tqdm(AUG_PDB.iterrows(), total=AUG_PDB.shape[0]):
        pulse_num, t1 = int(row['shot']), row['time']
        if (not AUG_DATA.get(pulse_num)) or pulse_num in set_pulses: 
            continue 

        profile_data, mp_data = AUG_DATA[pulse_num]['ida_profiles'], AUG_DATA[pulse_num]['machine_params']
        if profile_data is None or mp_data is None: 
            continue
        set_pulses.add(pulse_num)
        # gather the relevant profile and machine parameter into lists. 
        list_profs = [PROFILE(time_stamp=t, ne=profile_data['ne'].T[idx], te=profile_data['Te'].T[idx], radius=profile_data['radius'].T[idx], dne=profile_data['ne_unc'].T[idx], dte=profile_data['Te_unc'].T[idx]) for idx, t in enumerate(profile_data['time'])]       
        list_mps = [MACHINEPARAMETER(data=mp_data[key]['data'], name=key, times=mp_data[key]['time']) for key in REL_AUG_MP_COLS]
        pulse = PULSE(device='AUG', pulse_id=pulse_num, profiles_raw=list_profs, mps_raw=list_mps, mps_raw_labels=REL_AUG_MP_COLS)
        all_pulses.append(pulse)
    print(len(all_pulses))
    save_proccessed_classes(all_pulses, file_name='AUG_PDB_PYTHON_PULSES.pickle')

def create_ml_entries_from_pulse_classes(): 
    AUG_PULSES = load_classes_from_pickle()
    relevant_control_cols = ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT',  'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
    threshold = {'IpiFP': 0.5e6}
    for pulse in tqdm(AUG_PULSES):    
        pulse.setup_ml_entries(control_params = relevant_control_cols, above_threshold = threshold)
        # profiles, mps = pulse.get_ML_ready_array()
        # print(profiles.shape, mps.shape)
    save_proccessed_classes(AUG_PULSES, file_name='AUG_PDB_PYTHON_PULSES.pickle')

def main(): 
    # create_pulse_classes_from_raw() 
    create_ml_entries_from_pulse_classes()
    AUG_PULSES = load_classes_from_pickle()
    
    for pulse in AUG_PULSES: 
        # pulse.plot_raw_time_evolution(machine_params=['IpiFP', 'PNBI_TOT', 'D_tot', 'N_tot'])
        profiles, mps = pulse.get_ML_ready_array()
        print(profiles.shape, mps.shape)



def save_proccessed_classes(list_of_classes, file_name:str): 
    with open(PERSONAL_DATA_DIR_PROC + file_name, 'wb') as file: 
        pickle.dump(list_of_classes, file)

if __name__ == '__main__': 
    main()