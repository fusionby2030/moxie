"""
Here we want to convert the mega pickle file into a dataset to use in ML 

The output is a single time slice of neped profile. 

So grab the time slices from neped profiles
y -> (num_pulses*num_time_steps, len_hrts)

1. scalar averages of the input space for that time slice 
    - X -> (num_pulses*num_time_steps, num_features)

2. the actual raw data from that time slice
    - X -> (num_pulses*num_times_steps, num_features, len_feature_series)
    - This will require either padding or interpolation! 

"""

global num_pulses

import pickle 

def load_from_pickle(file_loc):
    with open(file_loc, 'rb') as file:
        pulse_dicts = pickle.load(file, 'latin1')
    return pulse_dicts


def get_shots_with_hrts_data(pulses):
    for pulse in pulses.keys():
        if pulses[pulse]['outputs']['NE'] == []:
            pulses = pulses.pop(pulse)

    num_pulses = len(pulses)
    return pulses


def average_a_feature(t1, t2, feature_values, feature_times):
    # Get the indexes of feature_times from t1 -> t2 
    # Average feature_values across those indexes 
    # return average
    average = None
    
    return average


def get_scalar_averages_from_time_slices(neped_times, input_space):

    efit_times = input_space['T_efit'] # TODO: FIX NAME! 
    new_inputs = {key: None for key in input_space.keys()}
    
    
    for col in input_space.keys():
        if isinstance(input_space[col], list):
            for j in range(0, len(neped_times) - 1):
                t1, t2 = neped_times[j], neped_times[j+1]
                average = average_a_feature(t1, t2, input_space[col])



def time_slice_neped(): 
    n, x = len(neped_times)*num_pulses, len(input_space)
    X = np.zeros((n, x)) #  


def main():
    # Load Data
    file_loc = 'TBD'
    pulse_dicts = load_from_pickle(file_loc)
    # Filter out the shots without neped values (should be like 200)
    pulse_dicts = get_shots_with_hrts_data(pulse_dicts)
    # Do some cutting of the data 
    pass 


if __name__ == '__main__': 
    main()
