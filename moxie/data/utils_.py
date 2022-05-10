def standardize_simple(x, mu=None, var=None):
    if mu is not None and var is not None:
        x_normed = (x - mu ) / var
        return x_normed
    else:
        mu = x.mean(0, keepdim=True)[0]
        var = x.std(0, keepdim=True)[0]
        x_normed = (x - mu ) / var
        return x_normed, mu, var

def normalize_profiles(profiles, mu_T=None, var_T=None, mu_D=None, var_D=None): 
    if mu_D is not None and var_D is not None and mu_T is not None and var_T is not None: 
        profiles[:, 0] = standardize_simple(profiles[:, 0], mu_D, var_D)
        profiles[:, 1] = standardize_simple(profiles[:, 1], mu_T, var_T)
        return profiles 
    else:  
        profiles[:, 0], mu_D, var_D = standardize_simple(profiles[:, 0])
        profiles[:, 1], mu_T, var_T = standardize_simple(profiles[:, 1])
        return profiles, mu_D, var_D, mu_T, var_T

def de_standardize(x, mu, var): 
    return (x*var) + mu

def standardize(x, mu, var): 
    return (x - mu) / var


import pickle 
def load_data(dataset_choice='SANDBOX_NO_VARIATIONS', file_loc='/home/kitadam/ENR_Sven/moxie/data/processed/pedestal_profiles_ML_READY_ak_19042022.pickle', elm_timings=False):
    """
    
    Returns 
        (train_tuple), (val_tuple), (test_tuple)
        where each tuple contains (profiles, machine_parameters, masks, psi_coords, rmid_coords, pulse_ids)
    """
    with open(file_loc, 'rb') as file:
        # test_X, test_y, test_mask, test_radii, test_real_space_radii, test_ids
        massive_dict = pickle.load(file)
        full_dict = massive_dict[dataset_choice]
        massive_dict = {}
    if elm_timings: 
        train_X, train_y, train_mask, train_radii, train_real_space_radii, train_ids, train_uncert, train_elm_timings = full_dict['train_dict']['padded']['profiles'],full_dict['train_dict']['padded']['controls'], full_dict['train_dict']['padded']['masks'], full_dict['train_dict']['padded']['radii'] ,full_dict['train_dict']['padded']['real_space_radii'] , full_dict['train_dict']['padded']['pulse_time_ids'], full_dict['train_dict']['padded']['uncerts'], full_dict['train_dict']['padded']['elm_timings_frass']    
        val_X, val_y, val_mask, val_radii, val_real_space_radii, val_ids, val_uncert, val_elm_timings = full_dict['val_dict']['padded']['profiles'],full_dict['val_dict']['padded']['controls'], full_dict['val_dict']['padded']['masks'], full_dict['val_dict']['padded']['radii'], full_dict['val_dict']['padded']['real_space_radii'], full_dict['val_dict']['padded']['pulse_time_ids'], full_dict['val_dict']['padded']['uncerts'], full_dict['val_dict']['padded']['elm_timings_frass']
        test_X, test_y, test_mask, test_radii, test_real_space_radii, test_ids, test_uncert, test_elm_timings = full_dict['test_dict']['padded']['profiles'],full_dict['test_dict']['padded']['controls'], full_dict['test_dict']['padded']['masks'], full_dict['test_dict']['padded']['radii'], full_dict['test_dict']['padded']['real_space_radii'], full_dict['test_dict']['padded']['pulse_time_ids'], full_dict['test_dict']['padded']['uncerts'], full_dict['test_dict']['padded']['elm_timings_frass']
        return (train_X, train_y, train_mask, train_radii, train_real_space_radii, train_ids, train_uncert, train_elm_timings), (val_X, val_y, val_mask, val_radii, val_real_space_radii, val_ids, val_uncert, val_elm_timings), (test_X, test_y, test_mask, test_radii, test_real_space_radii, test_ids, test_uncert, test_elm_timings)
    else: 
        train_X, train_y, train_mask, train_radii, train_real_space_radii, train_ids, train_uncert = full_dict['train_dict']['padded']['profiles'],full_dict['train_dict']['padded']['controls'], full_dict['train_dict']['padded']['masks'], full_dict['train_dict']['padded']['radii'] ,full_dict['train_dict']['padded']['real_space_radii'] , full_dict['train_dict']['padded']['pulse_time_ids'], full_dict['train_dict']['padded']['uncerts']
        val_X, val_y, val_mask, val_radii, val_real_space_radii, val_ids, val_uncert = full_dict['val_dict']['padded']['profiles'],full_dict['val_dict']['padded']['controls'], full_dict['val_dict']['padded']['masks'], full_dict['val_dict']['padded']['radii'], full_dict['val_dict']['padded']['real_space_radii'], full_dict['val_dict']['padded']['pulse_time_ids'], full_dict['val_dict']['padded']['uncerts']
        test_X, test_y, test_mask, test_radii, test_real_space_radii, test_ids, test_uncert = full_dict['test_dict']['padded']['profiles'],full_dict['test_dict']['padded']['controls'], full_dict['test_dict']['padded']['masks'], full_dict['test_dict']['padded']['radii'], full_dict['test_dict']['padded']['real_space_radii'], full_dict['test_dict']['padded']['pulse_time_ids'], full_dict['test_dict']['padded']['uncerts']
        return (train_X, train_y, train_mask, train_radii, train_real_space_radii, train_ids, train_uncert), (val_X, val_y, val_mask, val_radii, val_real_space_radii, val_ids, val_uncert), (test_X, test_y, test_mask, test_radii, test_real_space_radii, test_ids, test_uncert)
        