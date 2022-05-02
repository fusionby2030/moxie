"""
Reused utility modules for all models. 
"""
import torch 
from typing import List, Callable, Union, Any, TypeVar, Tuple 

Tensor = TypeVar('torch.tensor')

boltzmann_constant = 1.380e-23
e_c = 1.602e-19
mu_0 = 1.256e-6 

def get_new_beta_mach_sup(cutoff_1, cutoff_2, beta_mach_sup, beta_mach_unsup, current_iteration): 
    m = (beta_mach_sup - beta_mach_unsup) / (cutoff_2 - cutoff_1)
    b = beta_mach_unsup - m*cutoff_1
    return m*current_iteration + b

def conditional_inference_simple_mean(PULSE, model=None): 
    X, PSI, MASK, ID, MPS, _ = PULSE
    X[:, 0] = standardize(X[:, 0], D_norm, D_var)
    X[:, 1] = standardize(X[:, 1], T_norm, T_var)
    MPS = standardize(MPS, MP_norm, MP_var)
    with torch.no_grad():         
        cond_mu, cond_var =  model.p_zmachx(MPS)
        z_mach = model.reparameterize(cond_mu, cond_var)
        mu_stoch, log_var_stoch, mu_mach, log_var_mach = model.q_zy(X)
        z_stoch = mu_stoch
        z_mach = cond_mu
        z_conditional = torch.cat((z_stoch, z_mach), 1)
        out_profs_cond = model.p_yhatz(z_conditional)
    out_profs_cond[:, 0] = de_standardize(out_profs_cond[:, 0], D_norm, D_var)
    out_profs_cond[:, 1] = de_standardize(out_profs_cond[:, 1], T_norm, T_var)
    MPS = de_standardize(MPS, MP_norm, MP_var)
    X[:, 0] = de_standardize(X[:, 0], D_norm, D_var)
    X[:, 1] = de_standardize(X[:, 1], T_norm, T_var)
    return out_profs_cond

def normalize_profiles(profiles, mu_T=None, var_T=None, mu_D=None, var_D=None, de_norm=False): 
    if de_norm: 
        profiles[:, 0] = de_standardize(profiles[:, 0], mu_D, var_D)
        profiles[:, 1] = de_standardize(profiles[:, 1], mu_T, var_T)
    else: 
        profiles[:, 0] = standardize(profiles[:, 0], mu_D, var_D)
        profiles[:, 1] = standardize(profiles[:, 1], mu_T, var_T)
    return profiles

# Helper Functions
def de_standardize(x, mu, var):
    return (x*var) + mu

def standardize(x, mu, var):
    return (x - mu) / var

def torch_shaping_approx(minor_radius, tri_u, tri_l, elongation):
    triangularity = (tri_u + tri_l) / 2.0
    b = elongation*minor_radius
    gamma_top = -(minor_radius + triangularity)
    gamma_bot = minor_radius - triangularity
    alpha_top = -gamma_top / (b*b)
    alpha_bot = -gamma_bot / (b*b)
    top_int = (torch.arcsinh(2*torch.abs(alpha_top)*b) + 2*torch.abs(alpha_top)*b*torch.sqrt(4*alpha_top*alpha_top*b*b+1)) / (2*torch.abs(alpha_top))
    bot_int = (torch.arcsinh(2*torch.abs(alpha_bot)*b) + 2*torch.abs(alpha_bot)*b*torch.sqrt(4*alpha_bot*alpha_bot*b*b+1)) / (2*torch.abs(alpha_bot))
    return bot_int + top_int 

def bpol_approx(mp_tensors): 
    shaping = torch_shaping_approx(mp_tensors[:, 2], mp_tensors[:, 4], mp_tensors[:, 5], mp_tensors[:, 6])
    return mu_0*mp_tensors[:, 8] / shaping

def beta_approximation(profiles_tensors, mp_tensors):
    """
    To approximate beta! 
    The factor of 2 at the front is to compensate the ions which are nowhere to be found in this analysis. 
    The additional factor of 100 is to get it in percent form. 
    """
    density, temperature = profiles_tensors[:, 0, :], profiles_tensors[:, 1, :]
    pressure_prof = density*temperature
    pressure_average = pressure_prof[:, 0]     
    # TODO: This beta average is not really realistic I find... but am interested to see how it impacts
    bpol = bpol_approx(mp_tensors)
    bt = mp_tensors[:, 9]
    return (100*2)*e_c*2*mu_0 * pressure_average / (bt*bt + bpol*bpol)

def static_pressure_stored_energy_approximation(profs, mask): 
    return boltzmann_constant*torch.prod(profs.masked_fill_(~mask, 0), 1).sum(1)

def get_conv_output_size(initial_input_size, number_blocks, max_pool=True):
    """ The conv blocks we use keep the same size but use max pooling, so the output of all convolution blocks will be of length input_size / 2"""
    if max_pool==False:
        return initial_input_size
    out_size = initial_input_size
    for i in range(number_blocks):
        out_size = int(out_size / 2)
    return out_size

def get_trans_output_size(input_size, stride, padding, kernel_size):
    """ A function to get the output length of a vector of length input_size after a tranposed convolution layer"""
    return (input_size -1)*stride - 2*padding + (kernel_size - 1) +1

def get_final_output(initial_input_size, number_blocks, number_trans_per_block, stride, padding, kernel_size):
    """A function to get the final output size after tranposed convolution blocks"""
    out_size = initial_input_size
    for i in range(number_blocks):
        for k in range(number_trans_per_block):
            out_size = get_trans_output_size(out_size, stride, padding, kernel_size)
    return out_size
