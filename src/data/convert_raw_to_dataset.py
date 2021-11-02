"""
Here we want to convert the mega pickle file into a dataset to use in ML

The output is a single time slice of neped profile.

So grab the time slices from neped profiles
y -> (num_pulses*num_time_steps, len_hrts)

SHAPE OF NEPED = (701, 63)

1. scalar averages of the input space for that time slice
    - X -> (num_pulses*num_time_steps, num_features)

2. the actual raw data from that time slice
    - X -> (num_pulses*num_times_steps, num_features, len_feature_series)
    - This will require either padding or interpolation!

"""

global num_pulses

input_cols_efit = ['Q95', 'RGEO', 'CRO', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'POHM']
input_cols_scal = ['BT']
input_cols_gash = ['ELER']
input_cols_nbi = ['P_NBI']
input_cols_icrh = ['P_ICRH']



import pickle
import numpy as np

def load_from_pickle(file_loc='/home/kitadam/ENR_Sven/moxie/data/raw/all_shots.pickle'):
    """ Just loading the data from the big pickle file """
    with open(file_loc, 'rb') as file:
        pulse_dicts = pickle.load(file, 'latin1')
    return pulse_dicts


def get_shots_with_hrts_data(pulses):
    """ Some of the pulses don't have neped data, so we filter filter these out  """
    for pulse in pulses.keys():
        if pulses[pulse]['outputs']['NE'] == []:
            pulses = pulses.pop(pulse)
    num_pulses = len(pulses) # set the global numbner of pulses as the len of pulses we are using
    return pulses

def window_col(ne_time, input_values, input_time, tau=0.07):
    windowed_co = np.zeros_like(ne_time)

    for t in range(len(ne_time)):

        pass

    pass


def window_all(pulse):
    """ here we want to determine which neped time slices we can take, as we need all inputs to exist for that given space"""
    ne_time = pulse['outputs']['time']


    pass

    """
    efit_digitized = np.digitize(pulse['inputs']['EFIT_T'], ne_time)
    bt_digitized = np.digitize(pulse['inputs']['BT']['time'], ne_time)
    eler_digitized = np.digitize(pulse['inputs']['ELER']['time'], ne_time)

    icrh_digitized = np.digitize(pulse['inputs']['P_ICRH']['time'], ne_time)

    nbi_digitized = np.digitize(pulse['inputs']['P_NBI']['time'], ne_time)

    """

def massage_pulse(pulse):
    if pulse['NBI']['values'] == []:
        pulse['NBI']['values'] = np.zeros(13104)
        pulse['NBI']['time'] = np.linspace(40, 75)

    neped_indexes_to_use = digitize_all(pulse)


def main():
    # Load Data
    file_loc = 'TBD'
    pulse_dicts = load_from_pickle(file_loc)
    # Filter out the shots without neped values (should be like 200)
    pulse_dicts = get_shots_with_hrts_data(pulse_dicts)
    # Now we have a bunch of pulses in dictionary form.
    # each pulse has an 'inputs' and 'outputs'
    # For each pulse, we need the neped time series
    # NB: IF a majority of the data from the time series
    # using the time series,

    pass


def time_slice_neped():
    n, x = len(neped_times)*num_pulses, len(input_space)
    X = np.zeros((n, x)) #






def get_scalar_averages_from_time_slices(neped_times, input_space):

    efit_times = input_space['T_efit'] # TODO: FIX NAME!
    new_inputs = {key: None for key in input_space.keys()}


    for col in input_space.keys():
        if isinstance(input_space[col], list):
            for j in range(0, len(neped_times) - 1):
                t1, t2 = neped_times[j], neped_times[j+1]
                average = average_a_feature(t1, t2, input_space[col])



def average_a_feature(t1, t2, feature_values, feature_times):
    # Get the indexes of feature_times from t1 -> t2
    # Average feature_values across those indexes
    # return average
    average = None

    return average




if __name__ == '__main__':
    main()
