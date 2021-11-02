# import numpy as np
from ppf import *
import json
import pickle
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
    return data, t

def gather_all_outputs(shot):
    output_list = ['NE', 'DNE']
    output_dict = {key: None for key in output_list}
    for col in output_list:
        data, t = get_output(shot, col)
        output_dict[col] = data
    output_dict['time'] = t
    return output_dict


def get_input(shot, dda='EFIT', dtype='ELON'):
    data,x,t,nd,nx,nt,dunits,xunits,tunits,desc,comm,seq,ier = ppfdata(shot,dda,dtype,seq=0,uid="jetppf",device="JET", fix0=0,reshape=0,no_x=0,no_t=0,no_data=0)
    return data, t

def gather_all_inputs(shot):
    # Input cols is the
    input_cols_efit = ['Q95', 'RGEO', 'CRO', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'POHM']
    input_cols_scal = ['BT']
    input_cols_gash = ['ELER']
    input_cols_nbi = ['P_NBI']
    input_cols_icrh = ['P_ICRH']

    all_inputs = input_cols_efit + input_cols_gash + input_cols_nbi + input_cols_icrh + input_cols_scal
    input_dicts = {key: None for key in all_inputs}

    for col in input_cols_efit:
        data, t = get_input(shot, dtype=col)
        # input_dicts[col] = {'values': data.tolist(), 'time': t.tolist()}
        input_dicts[col] = data.tolist()
        if col == 'Q95':
            input_dicts['EFIT_T'] = t.tolist()

    data, t = get_input(shot, dda='SCAL', dtype='BT')
    input_dicts['BT'] = {'values': data.tolist(), 'time': t.tolist()}

    data, t = get_input(shot, dda='GASH', dtype='ELER')
    input_dicts['ELER'] = {'values': data.tolist(), 'time': t.tolist()}

    data, t = get_input(shot, dda='NBI', dtype='PTOT')
    input_dicts['P_NBI'] = {'values': data.tolist(), 'time': t.tolist()}

    data, t = get_input(shot, dda='ICRH', dtype='PTOT')
    input_dicts['P_ICRH'] = {'values': data.tolist(), 'time': t.tolist()}

    return input_dicts


def save_batch(cached_dict, start, stop):
    save_loc = './from_{}_{}.pickle'.format(start, stop)
    with open(save_loc, 'wb') as file:
        pickle.dump(cached_dict, file, pickle.HIGHEST_PROTOCOL)

    print('Saved batch of shots from {} --> {}'.format(start, stop))

def main():
    with open('./list_of_shots.pickle', 'rb') as file:
        validated_shots = pickle.load(file)
        # Should be 4942
        # validated_shots = []

    valid_count = 0
    pulse_dict = {}
    all_dict = {}
    cache_start = validated_shots[0]
    # ACtual end is 99552
    for pulse_num in validated_shots:
        if get_hrts_flag_from_shot(pulse_num):
            valid_count += 1
            print('\n-----SHOT {} is validated-------{} Total'.format(pulse_num, valid_count))
            # validated_shots.append(pulse_num)
            print('Gathering Outputs')
            outputs_dict = gather_all_outputs(pulse_num)
            print('Gathering inputs')
            inputs_dict = gather_all_inputs(pulse_num)
            # pulse_dict[str(pulse_num)] = {'inputs': inputs_dict, 'outputs': outputs_dict}
            all_dict[str(pulse_num)] = {'inputs': inputs_dict, 'outputs': outputs_dict}

        else:
            continue
        # if valid_count % 500 == 0 or pulse_num == 99552:
        #     save_batch(pulse_dict, cache_start, pulse_num)
        #     cache_start = pulse_num
        #     pulse_dict = {}

    with open('./all_shots.pickle', 'wb') as file:
        pickle.dump(all_dict, file)

    print('\n {} Shots collected'.format(valid_count))
    # with open('./list_of_shots.pickle', 'wb') as file:
    #     pickle.dump(validated_shots, file)
    # output_loc = './output_till_99552.gzip'
    # with gzip.open(output_loc, 'wb') as file:
    #     pickle.dump(pulse_dict, file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
