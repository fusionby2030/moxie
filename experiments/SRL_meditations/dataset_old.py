import pickle
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd 

PERSONAL_DATA_DIR = '/home/kitadam/ENR_Sven/moxie/data/raw/'
PERSONAL_DATA_DIR_PROC = '/home/kitadam/ENR_Sven/moxie/data/processed/'


EXP_DIR = '/home/kitadam/ENR_Sven/test_moxie/experiments/AUG_JET_REMIX/'
# PICKLED_CLASSES_FILELOC = EXP_DIR + 'profile_data_classes.pickle'
PICKLED_CLASSES_FILELOC = PERSONAL_DATA_DIR_PROC + 'AUG_ALL_PROFILES.pickle'
PICKLED_PULSES_FILELOC = PERSONAL_DATA_DIR_PROC + 'AUG_ALL_PULSES.pickle'

def main(): 
    # create_dataclasses_from_raw()
    with open(PICKLED_PULSES_FILELOC, 'rb') as file: 
        all_pulses = pickle.load(file)
    for pulse in all_pulses: 
        pulse.control_labels = ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT',  'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
        pulse.map_mps_to_profiles()       
        pulse.plot_time_evolution(machine_params=['IpiFP', 'PNBI_TOT', 'D_tot', 'N_tot'])
    """
    datacls = ProfileSetModule()
    datacls.setup()
    train_iter = datacls.train_dataloader()
    for n, batch in enumerate(train_iter): 
        
        t_0_batch, t_1_batch = batch[:, 0, :, :], batch[:, 1, :, :]
        plt.plot(t_0_batch[0, 0, :], color='blue', lw=5)
        plt.plot(t_1_batch[0, 0, :], color='green', lw=5)

        plt.plot(t_0_batch[1, 0, :], color='dodgerblue', ls='--', lw=5)
        plt.plot(t_1_batch[1, 0, :], color='forestgreen', ls='--', lw=5)
        plt.show()
        break
    
    
    # create_dataclasses_from_raw()
    """
    



PERSONAL_DATA_DIR = '/home/kitadam/ENR_Sven/moxie/data/raw/'
PERSONAL_DATA_DIR_PROC = '/home/kitadam/ENR_Sven/moxie/data/processed/'


EXP_DIR = '/home/kitadam/ENR_Sven/test_moxie/experiments/AUG_JET_REMIX/'
# PICKLED_CLASSES_FILELOC = EXP_DIR + 'profile_data_classes.pickle'
PICKLED_CLASSES_FILELOC = PERSONAL_DATA_DIR_PROC + 'AUG_ALL_PROFILES.pickle'
PICKLED_PULSES_FILELOC = PERSONAL_DATA_DIR_PROC + 'AUG_ALL_PULSES.pickle'

AUG_RAW_FILE_LOC = PERSONAL_DATA_DIR + 'AUG_pdb_pulses_raw_data.pickle'
AUG_PDB_FILE_LOC = PERSONAL_DATA_DIR + 'AUG_PDB_CSV.csv'

JET_RAW_FILE_LOC = PERSONAL_DATA_DIR + 'JET_pdb_pulses_raw_data.pickle' # 'JET_pdb_pulses_raw_with_bo.pickle' #'JET_pdb_pulses_raw_data.pickle'
JET_PDB_FILE_LOC = PERSONAL_DATA_DIR + 'jet-pedestal-database.csv'

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
        profile_data, mp_data = AUG_DATA[pulse_num]['ida_profiles'], AUG_DATA[pulse_num]['machine_params']
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
        rel_mp_cols = ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'P_OH', 'PICR_TOT', 'PECR_TOT', 'P_TOT', 'SHINE_TH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
        list_mps = [MACHINEPARAMETER(param=mp_data[key]['data'], name=key, time=mp_data[key]['time']) for key in rel_mp_cols]
        machine_parameters = MACHINEPARAMETERS(params=list_mps, names=rel_mp_cols)
        cut_rad, cut_time, cut_ne, cut_te, cut_dne, cut_dte = sample_rad.T[sample_window], sample_time[sample_window], sample_ne.T[sample_window], sample_te.T[sample_window], sample_dne.T[sample_window], sample_dte.T[sample_window]
        pbar.set_postfix({'#': pulse_num, 't1': t1, '#Slices': len(cut_time)})
        total_slices += len(cut_time)
        sample_pulse_profs = [PROFILE_ENTRY(device='AUG', pulse_id=pulse_num, time_stamp=cut_time[idx], ne=cut_ne[idx], te=cut_te[idx], radius=cut_rad[idx], dne=cut_dne[idx], dte=cut_dte[idx]) for idx, _ in enumerate(cut_time)]
        sample_pulse = PULSE(device='AUG', pulse_id=pulse_num, profiles=sample_pulse_profs, mps=machine_parameters)
        sample_pulse.control_labels = ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT',  'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
        sample_pulse.map_mps_to_profiles()
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

# TODO: do this for the pulses and not all the profiles instead. 
def setup_data(all_pulses = None, train_test_split=True) -> List: 
    if all_pulses is None: 
        with open(PICKLED_PULSES_FILELOC, 'rb') as file: 
            all_pulses = pickle.load(file)
    AUG_PULSE_LIST = list(set([pulse.pulse_id for pulse in all_pulses if pulse.device == 'AUG']))
    
    if train_test_split: 
        train_val_ids = random.sample(AUG_PULSE_LIST, k=int(0.85*len(AUG_PULSE_LIST)))
        train_ids = random.sample(train_val_ids, k=int(0.75*len(train_val_ids)))
        val_ids = list(set(train_val_ids)  - set(train_ids))
        test_ids = list(set(AUG_PULSE_LIST) - set(train_val_ids))

        all_pulse_sets, train_pulse_sets, val_pulse_sets, test_pulse_sets = [], [], [], []
        
        for pulse in tqdm(all_pulses): 
            # pulse = PULSE(device='AUG', pulse_id=pulse_id, profiles=[prof for prof in all_profs if prof.pulse_id == pulse_id])
            if pulse.mapped_mps is None: 
                pulse.control_labels = ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT',  'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
                pulse.map_mps_to_profiles()
            pulse_id = pulse.pulse_id
            sets = pulse.get_time_obervables()
            all_pulse_sets.extend(sets)
            if pulse_id in train_ids: 
                train_pulse_sets.extend(sets)
            elif pulse_id in val_ids: 
                val_pulse_sets.extend(sets)
            else: 
                test_pulse_sets.extend(sets)
        return all_pulse_sets, (train_pulse_sets, val_pulse_sets, test_pulse_sets)
    else: 
        raise NotImplementedError('Shit')

@dataclass 
class MACHINEPARAMETER: 
    name: str
    param: Union[np.array, List]
    time: Union[np.array, List]

@dataclass 
class PROFILE_ENTRY: 
    device: str 
    pulse_id: int 
    time_stamp: float 
    ne: Union[List[float], np.array]
    te: Union[List[float], np.array]
    dne: Union[List[float], np.array]
    dte: Union[List[float], np.array]
    radius: Union[List[float], np.array]
    elm_frac: float = None
    mps: List[Union[np.array, MACHINEPARAMETER]] = None
    def get_ML_ready_array(self, from_SOL: bool = True) -> np.array:
        if from_SOL: 
            end_idx = np.searchsorted(self.radius, 1.15, side='left')
            beg_idx = end_idx - 50
        else: 
            beg_idx = 0
            end_idx = -1
        return np.vstack((1e-19*self.ne[beg_idx:end_idx], self.te[beg_idx:end_idx]))
    def get_SOL_radius(self): 
        end_idx = np.searchsorted(self.radius, 1.15, side='left')
        beg_idx = end_idx - 50
        return self.radius[beg_idx:end_idx]

@dataclass 
class MACHINEPARAMETERS:
    # ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'P_OH', 'PICR_TOT', 'PECR_TOT', 'P_TOT', 'SHINE_TH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
    params: Union[List[np.array], List[MACHINEPARAMETER]]
    names: List[str]
    def get_param(self, key): 
        idx = self.names.index(key)
        mp_val, mp_time = self.params[idx].param, self.params[idx].time
        return np.array(mp_val), np.array(mp_time) 

    def find_surrounding_points(self, key, t, t_window_size): 
        val, time = self.get_param(key)
        relevant_window = np.logical_and(time >= t - t_window_size, time <= t + t_window_size)
        return val[relevant_window]

    def mean_surrounding_points(self, key, t, t_window_size=0.002): 
        vals = self.find_surrounding_points(key, t, t_window_size)
        return np.mean(vals)
@dataclass 
class PULSE: 
    device: str 
    pulse_id: int 
    profiles: List[PROFILE_ENTRY]
    mps: MACHINEPARAMETERS
    mapped_mps: Union[np.array, list, MACHINEPARAMETERS] = None 
    control_labels: List[str] = None # 
    def plot_time_evolution(self, **kwargs): 
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        profs = np.array([prof.get_ML_ready_array() for prof in self.profiles])
        radii = np.array([prof.get_SOL_radius() for prof in self.profiles])
        t = np.array([prof.time_stamp for prof in self.profiles])
        fig, n_ax = plt.subplots()
        t_ax = n_ax.twinx()
        if kwargs.get('machine_params'): 
            axs = []
            bars = []
            lines = []
            mp_idxs = []
            offset = 0.0
            for axs_idx, param_name  in enumerate(kwargs['machine_params']): 
                vals, times = self.mps.get_param(param_name)
                axs.append(n_ax.inset_axes([0.75, 0.8 - offset, 0.15, 0.15]))
                offset += 0.2
                axs[axs_idx].plot(times, vals)
                bars.append(axs[axs_idx].axvline(t[0], color='gold'))
                mp_idx = self.control_labels.index(param_name)
                mp_idxs.append(mp_idx)
                ln, = axs[axs_idx].plot(t[0], self.mapped_mps[0, mp_idx], 'or')
                lines.append(ln)
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
            for bar, line, mp_idx in zip(bars, lines, mp_idxs): 
                bar.set_data(t[i], [0, 1e6])
                line.set_data(t[i], self.mapped_mps[i, mp_idx])
            time_text.set_text(time_template % (t[i]))
            return n_ln, t_ln, time_text, *bars, *lines
        ani = FuncAnimation(fig, animate, len(t), interval=(t[1] - t[0])*5000, repeat_delay=1e3, blit=True)
        # writer_video = FFMpegWriter(fps=dt*20000)
        # ani.save('/home/kitadam/ENR_Sven/test_moxie/experiments/SRL_meditations/full_pulse_train_vid_fast.mp4', dpi=300, writer=writer_video)
        
        plt.show()
        
    def get_time_obervables(self, window_size=2):
        # https://github.com/tcapelle/torchdata/blob/main/02_Custom_timeseries_datapipe.ipynb
        # return a list of profile pairs, one after the other 
        current_set = []
        full_set = []
        it = iter(self.profiles)
        it_mp = iter(self.mapped_mps)
        while True: 
            try: 
                while len(current_set) < window_size: 
                    current_set.append(next(it))
                full_set.append(current_set)
                current_set = [current_set[1]]
            except StopIteration: 
                return full_set    

    def map_mps_to_profiles(self): 
        self.profiles.sort(key=lambda x: x.time_stamp)
        hrts_time_steps = [prof.time_stamp for prof in self.profiles]
        self.mapped_mps = None 
        # Go through all the profile time steps 
        # Get the machine parameters for each time step 
        # Store in a numpy array, tabulated?  
        self.mapped_mps = np.array([np.array([self.mps.mean_surrounding_points(key, t) for key in self.control_labels]) for t in hrts_time_steps])
        assert len(self.mapped_mps) == len(hrts_time_steps)        


@dataclass 
class DATASET(Dataset):
    sets: List[List[PROFILE_ENTRY]]
    norms: Tuple = None 
    def denormalize_profiles(self, profiles): 
        def de_standardize(x, mu, var): 
            if isinstance(x, torch.Tensor): 
                mu, var = torch.from_numpy(mu), torch.from_numpy(var)
            return x*var + mu
        N_mean, N_var, T_mean, T_var = self.norms
        # N_mean, N_var, T_mean, T_var = N_mean[-60:-10], N_var[-60:-10], T_mean[-60:-10], T_var[-60:-10]
        profiles[:, 0, :] = de_standardize(profiles[:, 0, :], N_mean, N_var)
        profiles[:, 1, :] = de_standardize(profiles[:, 1, :], T_mean, T_var)
        return profiles
    def normalize_profiles(self, profiles): 
        def standardize(x, mu, var):
            if mu is None and var is None:
                mu = x.mean(0, keepdim=True)[0]
                var = x.std(0, keepdim=True)[0]
            x_normed = (x - mu ) / var
            return x_normed, mu, var
        N_mean, N_var, T_mean, T_var = self.norms
        # N_mean, N_var, T_mean, T_var = N_mean[-60:-10], N_var[-60:-10], T_mean[-60:-10], T_var[-60:-10]
        profiles[0, :], _, _ = standardize(profiles[0, :], N_mean, N_var)
        profiles[1, :], _, _ = standardize(profiles[1, :], T_mean, T_var)
        return profiles#,  N_mean, N_var, T_mean, T_var, 
    def get_normalizing(self, from_SOL=True): 
        # Should really not do the from_SOL here...
        train_set = np.array(self.sets)
        
        train_set_profs = np.array([profile.get_ML_ready_array() for profile in train_set[:, 0]])
        train_set_densities = 1e-19*train_set_profs[:, 0, :] # np.array([1e-19*profile.ne for profile in train_set[:, 0]])
        train_set_temperatures = train_set_profs[:, 1, :]# np.array([profile.te for profile in train_set[:, 0]])
        ne_mu, ne_var = np.mean(train_set_densities, 0), np.std(train_set_densities, 0)
        te_mu, te_var = np.mean(train_set_temperatures, 0), np.std(train_set_temperatures, 0)
        self.norms = (ne_mu, ne_var, te_mu, te_var)
        return ne_mu, ne_var, te_mu, te_var
    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx, normalize=True):
        item = self.sets[idx]
        t_0, t_1  = item 
        profs_t1 = t_0.get_ML_ready_array()
        profs_t2 = t_1.get_ML_ready_array()
        if normalize == True: 
            profs_t1 = self.normalize_profiles(profs_t1)
            profs_t2 = self.normalize_profiles(profs_t2)
        return np.array([profs_t1, profs_t2])


class ProfileSetModule(pl.LightningDataModule): 
    def __init__(self, data_dir: str = PICKLED_CLASSES_FILELOC, batch_size=512): 
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        with open(PICKLED_PULSES_FILELOC, 'rb') as file: 
            self.all_pulses = pickle.load(file)

    
    def setup(self, stage: Optional[str] = None): 
        self.all_pulse_sets, (train_set, val_set, test_set) = setup_data(self.all_pulses)
        
        self.train_pulse_sets = train_set # This is a List[List[PROFILE_ENTRY]]
        self.train_set = DATASET(train_set)
        self.ne_mu, self.ne_var, self.te_mu, self.te_var = self.train_set.get_normalizing()
        self.val_set = DATASET(val_set, norms=(self.ne_mu, self.ne_var, self.te_mu, self.te_var))
        self.test_set = DATASET(test_set, norms=(self.ne_mu, self.ne_var, self.te_mu, self.te_var))
        
    def get_normalizers(self): 
        return self.ne_mu, self.ne_var, self.te_mu, self.te_var 
    def train_dataloader(self): 
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): 
        return DataLoader(self.val_set, batch_size=self.batch_size)
    def test_dataloader(self): 
        return DataLoader(self.test_set, batch_size=self.batch_size)

def load_processed_data(): 
    with open(PICKLED_CLASSES_FILELOC, 'rb') as file: 
        all_profs = pickle.load(file)
    return all_profs

def load_raw_data() -> Union[dict, pd.DataFrame]:
    cols = ['shot', 'time', 'Teped(eV)', 'neped(10^19m^-3)', 'peped(Pa)', 'tewidth(psin)', 'newidth(psin)', 'pewidth(psin)', 'tepos(psin)', 'nepos(psin)', 'pepos(psin)', 'beta_n', 'Ip(MA)', 'Bt(T)', 'Rgeo(m)', 'aminor(m)', 'elongation', 'delta_upp', 'delta_low', 'P_NBI(MW)', 'P_ICRF(MW)', 'P_ECRH(MW)', 'P_TOT(MW)', 'P_RAD(MW)', 'Volume(m^3)', 'q95', 'Rmag(m)', 'Zmag(m)', 'Something(-)']

    with open(AUG_RAW_FILE_LOC, 'rb') as file: 
        AUG_DATA = pickle.load(file)
    AUG_PDB = pd.read_csv(AUG_PDB_FILE_LOC, header=3, usecols=cols)
    with open(JET_RAW_FILE_LOC, 'rb') as file: 
        JET_DATA = pickle.load(file)
    JET_PDB = pd.read_csv(JET_PDB_FILE_LOC) 

    return AUG_DATA, AUG_PDB, JET_DATA, JET_PDB


if __name__ == '__main__': 
    main()
