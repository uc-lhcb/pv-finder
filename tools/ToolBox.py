import sys
import math
import numba
import numpy as np

## ========================================================================================
@numba.vectorize(nopython=True)
def norm_cdf(mu, sigma, x):
    """
    Cumulative distribution function for the standard normal distribution.

    Much faster than scipy.stats.norm.cdf even without jit (if added). Use
    np.erf for non-vectorized version. (adds ~1 second)
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

## ========================================================================================

## ========================================================================================
#@numba.vectorize(nopython=True)
def get_res(nTrks, config):
    ## Parameters' values for the resolution as a function of the number 
    ## of tracks originating from the PV.
    A_res = config['A_res']
    B_res = config['B_res']
    C_res = config['C_res']
    
    ## Resolution function (in [mm]? --> to be checked!!)
    return (A_res * np.power(nTrks, -1 * B_res) + C_res) * 0.001
## ========================================================================================

#def sort_ak_arrays(sorting_ak, l_arrays):
#    