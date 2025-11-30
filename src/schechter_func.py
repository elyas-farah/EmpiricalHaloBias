import numpy as np
from mpmath import gammainc
from mpmath import erf



def gammainc_vec(a, x):
    result = []
    for i in range(x.size):
        result.append(float(gammainc(a[i], x[i])))
        
    return np.array(result)

def erf_vec(x):
    result = []
    for i in range(x.size):
        result.append(float(erf(x[i])))
        
    return np.array(result)


def schechter_function(M, log10_ns,  alpha, log10_M0, beta = 1):    
    M0 = 10**log10_M0
    ns = 10**log10_ns
    
    term  = ns * (M / M0)**alpha * np.exp(-(M/M0)**beta) 
    

    return term
    
    
def nh_lik(log10_Mmin, log10_Mmax, *params):
    '''A function that represents the predicted halo number counts in bins of delta_m, used in the likelihood function.'''
    log10_M0, log10_ns, alpha, beta = params
    
    
    
    
    
    
    log10_min = log10_Mmin - log10_M0
    log10_max = log10_Mmax - log10_M0
    
    Gamma_min = gammainc((alpha + 1)/beta, (10**log10_min)**beta)
    Gamma_max = gammainc((alpha + 1)/beta, (10**log10_max)**beta)
    term1 = (10**log10_ns)*(10**log10_M0)*(1/beta)* (Gamma_min - Gamma_max)
    
  
    return term1

def minus_Likelihood(delta_2d_arr, halo_cat, delta_m_min, delta_m_max, log10_Mmin, log10_Mmax, slab_width = 125, *params):
    '''Poissonian likelihood '''
    
    log10_ns, alpha, log10_M0,  beta = params
    

    
    # print("params loaded")
    slab_id = (halo_cat.z_halo // slab_width).astype(int)
    # print("slabs are splite")
    
    
    delta_halo = delta_2d_arr[slab_id, halo_cat.ix_halo, halo_cat.iy_halo]
    
    delta_halo_mask = np.where((delta_halo > delta_m_min) & (delta_halo < delta_m_max))
    delta_m_mask_size = np.where((delta_2d_arr.flatten() > delta_m_min) & (delta_2d_arr.flatten() < delta_m_max))[0].size
    
    
    halo_mass_masked = halo_cat.halo_mass[delta_halo_mask]
    
    
    term1 = np.log(schechter_function(halo_mass_masked, log10_ns, alpha, log10_M0, beta))
    
    term2 = nh_lik(log10_Mmin, log10_Mmax, log10_M0, log10_ns, alpha, beta)
    
    output = -1*(np.sum(term1) - (delta_m_mask_size*term2))
    return output


def nh(log10_one_delta_m, log10_Mmin, log10_Mmax, *funcs):
    log10_ns_func, alpha_func, log10_M0_func, beta_func = funcs
    
    log10_ns = log10_ns_func(log10_one_delta_m)
    alpha = alpha_func(log10_one_delta_m)
    log10_M0 = log10_M0_func(log10_one_delta_m)
    beta = beta_func(log10_one_delta_m)
    
    log10_min = log10_Mmin - log10_M0
    log10_max = log10_Mmax - log10_M0
    
    Gamma_min = gammainc_vec((alpha + 1)/beta, (10**log10_min)**beta)
    Gamma_max = gammainc_vec((alpha + 1)/beta, (10**log10_max)**beta)
    term1 = (10**log10_ns)*(10**log10_M0)*(1/beta)* (Gamma_min - Gamma_max)
    
    
    
   
    return term1