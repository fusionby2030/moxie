import pickle

"""
So we double stored the time values for the following resources

But RCO (minor radius supposedly, is not existing!)

"""

if __name__ == '__main__':
    file1 = '/home/kitadam/ENR_Sven/moxie/data/raw/from_79000_79449.pickle' # This can be read into memory!
    # file2 = '/home/kitadam/ENR_Sven/moxie/data/raw/output_till_99552.gzip'
    with open(file1, 'rb') as file:
        pulse_dict_1 = pickle.load(file, encoding='latin1')

    repeated_cols = ['Q95', 'RGEO', 'RCO', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'BTAX', 'POHM']

    for pulse, values in pulse_dict_1.items():
        print(pulse)
        inputs = values['inputs']
        time_BT = inputs['BTAX']['time']
        inputs['time'] = time_BT
        for col in repeated_cols:
            if inputs[col]['time'] == time_BT:
                print(col)
                inputs[col] = inputs[col]['values']
            else:
                print('{} has different time values than BT'.format(col))

                pass


        # time_q95 = inputs['Q95']['time']
        # break

    # all_pulses = {**pulse_dict_1, **pulse_dict_2}

    # final_storage = '/home/kitadam/ENR_Sven/moxie/data/raw/all_pulses_v1.pickle'
    # with open(final_storage, 'wb') as file:
    #     pickle.dump(all_pulses, file)
