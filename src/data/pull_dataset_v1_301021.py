# import numpy as np

""" 
This is a file to pull timeseries pulse data from the Heimdal cluster. 

You need to do this on JET cluster. 
"""


from ppf import *
import json
import pickle
import gzip
"""
Target dictionary:

{'pulse_1': {'inputs': {'BT': {'values': np.array, 'time': np.array},
                        'IP': {'values': np.array, 'time': np.array}, ...}
            'outputs': {'time': {'R': np.array, 'neped'}}
            },
'pulse_2': {'inputs': {'BT': {'values': np.array, 'time': np.array},
                        'IP': {'values': np.array, 'time': np.array}, ...}
            'outputs': {'time': {'R': np.array, 'neped'}}
            },
            }

"""

def get_hrts_flag_from_shot(shot, dda='HRTS', dtyp='NE', uid='jetppf'):
    flag,ier=ppfgsf(shot,0,dda,dtyp,mxstat=200)
    if flag[0] != 0:
        return True
    return False

def get_output(shot, dtyp):
    data, x, t, nd, nx, nt, dunits, xunits, tunits, desc, comm, seq, ier= ppfdata(shot, dda='HRTS', dtyp=dtyp, reshape=3)
    return data

def gather_all_outputs(shot):
    output_list = ['NE', 'DNE']
    output_dict = {key: None for key in output_list}
    for col in output_list:
        output_dict[col] = get_output(shot, col)
    return output_dict


def get_input(shot, dda='EFIT', dtype='ELON'):
    data,x,t,nd,nx,nt,dunits,xunits,tunits,desc,comm,seq,ier = ppfdata(shot,dda,dtype,seq=0,uid="jetppf",device="JET", fix0=0,reshape=0,no_x=0,no_t=0,no_data=0)
    return data, t

def gather_all_inputs(shot):
    # Input cols is the
    input_cols_efit = ['Q95', 'RGEO', 'RCO', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'BTAX', 'POHM']
    input_cols_gash = ['ELER']
    input_cols_nbi = ['P_NBI']
    input_cols_icrh = ['P_ICRH']

    all_inputs = input_cols_efit + input_cols_gash + input_cols_nbi + input_cols_icrh
    input_dicts = {key: None for key in all_inputs}

    for col in input_cols_efit:
        data, t = get_input(shot, dtype=col)
        input_dicts[col] = {'values': data.tolist(), 'time': t.tolist()}

    data, t = get_input(shot, dda='GASH', dtype='ELER')
    input_dicts['ELER'] = {'values': data.tolist(), 'time': t.tolist()}

    data, t = get_input(shot, dda='NBI', dtype='PTOT')
    input_dicts['P_NBI'] = {'values': data.tolist(), 'time': t.tolist()}

    data, t = get_input(shot, dda='ICRH', dtype='PTOT')
    input_dicts['P_ICRH'] = {'values': data.tolist(), 'time': t.tolist()}

    return input_dicts


def main():
    validated_shots = []
    shot_start = 79000
    valid_count = 0
    pulse_dict = {}
    # ACtual end is 99552
    for pulse_num in range(shot_start, 99552):
        if get_hrts_flag_from_shot(pulse_num):
            valid_count += 1
            print('\n-----SHOT {} is validated-------{} Total'.format(pulse_num, valid_count))
            validated_shots.append(pulse_num)
            print('Gathering Outputs')
            outputs_dict = gather_all_outputs(pulse_num)
            print('Gathering inputs')
            inputs_dict = gather_all_inputs(pulse_num)

            pulse_dict[str(pulse_num)] = {'inputs': inputs_dict, 'outputs': outputs_dict}
        else:
            continue
    print('\n {} Shots collected'.format(valid_count))
    output_loc = './output.gzip'
    with gzip.open(output_loc, 'wb') as file:
        pickle.dump(pulse_dict, file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

