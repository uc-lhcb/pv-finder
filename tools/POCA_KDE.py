'''
Methods to compute udpated POCA KDEs.
We use the original approach from approach in:
https://gitlab.cern.ch/mstahl/Rec/-/blob/d3cc26d2b979c77d2fa83d5d23213a61918259d4/Tr/PatPV/src/PVFinder.cpp
to compute the POCA KDEs but with an update binning scheme.
'''

import awkward as ak
import numpy as np
import numpy.ma as ma
from math import *
from tools.named_tuples_def import InputData_tuple
from utils.utilities import Timer
import numba as nb

## Constant values defining the ranges, 
## number of bins in the (x,y) grid searches
## the bins widths...
_zmin     = -100
_zmax     =  300
_nbins_xy =   20
_ninterxy =    3
_xymin    =   -0.4
_xymax    =    0.4
_interxy  =    0.01


## ============================================================================================================    
@nb.jit(fastmath=True,nopython=True)
def get_bin_center(nbins, _min, _max, ibin):
    ## Method to return the center of a considered voxel in a given axis
    return (ibin + 0.5) / nbins * (_max - _min) + _min

## ============================================================================================================    
_px_coarse_grid = np.empty(_nbins_xy)
_py_coarse_grid = np.empty(_nbins_xy)
for i in range(_nbins_xy):
    _px_coarse_grid[i] = get_bin_center(_nbins_xy, _xymin, _xymax, i)
    _py_coarse_grid[i] = get_bin_center(_nbins_xy, _xymin, _xymax, i)

    
## ============================================================================================================    
#@nb.jit(fastmath=True,nopython=True)
def get_tracks_in_range(array,filt):
    
    ## Use the filter to return an ak array with only values
    ## usefull for the computation to be done after
    tracks_in_range = array[filt]
    
    return tracks_in_range
    
## ============================================================================================================    
#@nb.jit(fastmath=True,nopython=True)
def get_filter_tracks_in_range(z, poca_z, major_axis_z):
    
    ## Get the arrays of minimum and maximum z values to be considered
    zmin = (poca_z-3*major_axis_z)
    zmax = (poca_z+3*major_axis_z)
    
    ## To use the mask filtering we need an array of shape N_tracks with 'True' 
    ## where the tracks ARE in the range considered and 'False' otherwise.
    ## Indeed, the get_tracks_in_range will return values that are True , i.e. not filtered out
    filt = np.logical_and( zmin < z, z <= zmax)

    return filt

## ============================================================================================================    
def get_determinant(x, y, Mdz, m1dx, tx, ty):

    print("In get_determinant")

    # To avoid having issues with dividing by zero...
    epsilon = 0.0000001
    txSq_tySq    = tx*tx + ty*ty
    txSq_tySq_Sq = txSq_tySq * txSq_tySq
    m1dx_Sq      = m1dx*m1dx
    xSq          = x*x + epsilon
    ySq          = y*y
    xtx_yty_Sq   = (x*tx + y*ty) * (x*tx + y*ty) + epsilon
    
    det = (Mdz * m1dx_Sq) * (np.sqrt(txSq_tySq + 1) / xSq) * (xSq + ySq + xtx_yty_Sq/txSq_tySq_Sq)
    
    return det

## ============================================================================================================    
@nb.jit(fastmath=True,nopython=True)
def get_PDFs(point_x,       point_y,       point_z,
             poca_x,        poca_y,        poca_z,
             major_axis_x,  major_axis_y,  major_axis_z,
             minor_axis1_x, minor_axis1_y, minor_axis1_z,
             minor_axis2_x, minor_axis2_y, minor_axis2_z,
             det):

    '''
    
      auto const xvec = _center-scan_point;
      auto const chisq = std::pow(xvec.Dot(_minor_axis1.Unit()),2)/_minor_axis1.Mag2() +
                         std::pow(xvec.Dot(_minor_axis2.Unit()),2)/_minor_axis2.Mag2() +
                         std::pow(xvec.Dot(_major_axis.Unit()),2)/_major_axis.Mag2();


    '''    
    xvec_x = poca_x - point_x
    xvec_y = poca_y - point_y
    xvec_z = poca_z - point_z
    
    major_axis_magSq  = (major_axis_x  * major_axis_x)  + (major_axis_y  * major_axis_y)  + (major_axis_z  * major_axis_z)
    minor_axis1_magSq = (minor_axis1_x * minor_axis1_x) + (minor_axis1_y * minor_axis1_y) + (minor_axis1_z * minor_axis1_z)
    minor_axis2_magSq = (minor_axis2_x * minor_axis2_x) + (minor_axis2_y * minor_axis2_y) + (minor_axis2_z * minor_axis2_z)
    
    xvec_dot_major_axis  = (xvec_x * major_axis_x)  + (xvec_y * major_axis_y)  + (xvec_z * major_axis_z) 
    xvec_dot_minor_axis1 = (xvec_x * minor_axis1_x) + (xvec_y * minor_axis1_y) + (xvec_z * minor_axis1_z)
    xvec_dot_minor_axis2 = (xvec_x * minor_axis2_x) + (xvec_y * minor_axis2_y) + (xvec_z * minor_axis2_z)
    
    xvec_dot_major_axis_norm_Sq  = (xvec_dot_major_axis  * xvec_dot_major_axis)  / (major_axis_magSq  * major_axis_magSq) 
    xvec_dot_minor_axis1_norm_Sq = (xvec_dot_minor_axis1 * xvec_dot_minor_axis1) / (minor_axis1_magSq * minor_axis1_magSq) 
    xvec_dot_minor_axis2_norm_Sq = (xvec_dot_minor_axis2 * xvec_dot_minor_axis2) / (minor_axis2_magSq * minor_axis2_magSq)
    
    chiSq = xvec_dot_major_axis_norm_Sq + xvec_dot_minor_axis1_norm_Sq + xvec_dot_minor_axis2_norm_Sq
            
    return np.exp(-0.5*chiSq)/np.sqrt(det)
    
## ============================================================================================================    
@nb.jit(fastmath=True,nopython=True)
def get_poca_PDFs(pz,
                  poca_x,        poca_y,        poca_z,
                  major_axis_x,  major_axis_y,  major_axis_z,
                  minor_axis1_x, minor_axis1_y, minor_axis1_z,
                  minor_axis2_x, minor_axis2_y, minor_axis2_z,
                  poca_det):

    poca_PDFs = np.empty((_nbins_xy,_nbins_xy,poca_x.shape[0]))

    for bx in range(_nbins_xy):
        for by in range(_nbins_xy):
            poca_PDFs[bx][by] = get_PDFs(_px_coarse_grid[bx], _py_coarse_grid[by], pz,
                                         poca_x, poca_y, poca_z,
                                         major_axis_x,  major_axis_y,  major_axis_z,
                                         minor_axis1_x, minor_axis1_y, minor_axis1_z,
                                         minor_axis2_x, minor_axis2_y, minor_axis2_z,
                                         poca_det
                                        )
            
    return poca_PDFs
    
## ============================================================================================================    
@nb.jit(fastmath=True,nopython=True)
def get_PDFs_XY_precalculated(point_z,
                              poca_z,
                              major_axis_z,
                              minor_axis1_z,
                              minor_axis2_z,
                              det,
                              vec_dot_major_axis_x,
                              vec_dot_minor_axis1_x,
                              vec_dot_minor_axis2_x,
                              vec_dot_major_axis_y,
                              vec_dot_minor_axis1_y,
                              vec_dot_minor_axis2_y,
                              major_axis_magSqSq,
                              minor_axis1_magSqSq,
                              minor_axis2_magSqSq,
                              ):

    '''
    
      auto const xvec = _center-scan_point;
      auto const chisq = std::pow(xvec.Dot(_minor_axis1.Unit()),2)/_minor_axis1.Mag2() +
                         std::pow(xvec.Dot(_minor_axis2.Unit()),2)/_minor_axis2.Mag2() +
                         std::pow(xvec.Dot(_major_axis.Unit()),2)/_major_axis.Mag2();


    '''    
    vec_z = poca_z - point_z
    
    vec_dot_major_axis  = vec_dot_major_axis_x  + vec_dot_major_axis_y  + (vec_z * major_axis_z) 
    vec_dot_minor_axis1 = vec_dot_minor_axis1_x + vec_dot_minor_axis1_y + (vec_z * minor_axis1_z)
    vec_dot_minor_axis2 = vec_dot_minor_axis2_x + vec_dot_minor_axis2_y + (vec_z * minor_axis2_z)
    
    vec_dot_major_axis_norm_Sq  = (vec_dot_major_axis  * vec_dot_major_axis)  / (major_axis_magSqSq) 
    vec_dot_minor_axis1_norm_Sq = (vec_dot_minor_axis1 * vec_dot_minor_axis1) / (minor_axis1_magSqSq) 
    vec_dot_minor_axis2_norm_Sq = (vec_dot_minor_axis2 * vec_dot_minor_axis2) / (minor_axis2_magSqSq)
    
    chiSq = vec_dot_major_axis_norm_Sq + vec_dot_minor_axis1_norm_Sq + vec_dot_minor_axis2_norm_Sq
            
    return np.exp(-0.5*chiSq)/np.sqrt(det)
    
## ============================================================================================================    
@nb.jit(fastmath=True,nopython=True)
def compute_poca_PDFs(n_bins_z,
                      poca_x,        poca_y,        poca_z,
                      major_axis_x,  major_axis_y,  major_axis_z,
                      minor_axis1_x, minor_axis1_y, minor_axis1_z,
                      minor_axis2_x, minor_axis2_y, minor_axis2_z,
                      poca_det):
    
    ## -------------------------------------------------------
    ## Pre-compute all necessary information from XY
    nTracks = poca_x.shape[0]
    vec_dot_major_axis_x  = np.empty((_nbins_xy,nTracks))
    vec_dot_minor_axis1_x = np.empty((_nbins_xy,nTracks))
    vec_dot_minor_axis2_x = np.empty((_nbins_xy,nTracks))
    vec_dot_major_axis_y  = np.empty((_nbins_xy,nTracks))
    vec_dot_minor_axis1_y = np.empty((_nbins_xy,nTracks))
    vec_dot_minor_axis2_y = np.empty((_nbins_xy,nTracks))

    for i in range(_nbins_xy):
        ## X information
        vec_x = (poca_x - _px_coarse_grid[i])
        vec_dot_major_axis_x[i]  = vec_x * major_axis_x
        vec_dot_minor_axis1_x[i] = vec_x * minor_axis1_x
        vec_dot_minor_axis2_x[i] = vec_x * minor_axis2_x
        
        ## Y information
        vec_y = (poca_y - _px_coarse_grid[i])
        vec_dot_major_axis_y[i]  = vec_y * major_axis_y
        vec_dot_minor_axis1_y[i] = vec_y * minor_axis1_y
        vec_dot_minor_axis2_y[i] = vec_y * minor_axis2_y
            
    ## -------------------------------------------------------
    ## Looping over z bins:
    for bz in range(n_bins_z):
        #print("At z bin:",bz)
        ## Get z center of the bin 'bz':
        pz = get_bin_center(n_bins_z, _zmin, _zmax, bz)

        ## Get filter to remove tracks not of interest for z bin 'bz':        
        filt = get_filter_tracks_in_range(pz, poca_z, major_axis_z)

        indexes_tracks_in_range = np.where(filt)
            
        best_x_bin = -99
        best_y_bin = -99
        sum_poca_PDFs = -1
            
        for iTrack in indexes_tracks_in_range:
            
            poca_z_filt = poca_z[iTrack]
            
            major_axis_z_filt  = major_axis_z[iTrack]
            minor_axis1_z_filt = minor_axis1_z[iTrack]
            minor_axis2_z_filt = minor_axis2_z[iTrack]
            poca_det_filt      = poca_det[iTrack]
            
            major_axis_magSq_filt  = (major_axis_x[iTrack] * major_axis_x[iTrack]) + (major_axis_y[iTrack] * major_axis_y[iTrack]) + (major_axis_z[iTrack] * major_axis_z[iTrack])
            minor_axis1_magSq_filt = (minor_axis1_x[iTrack] * minor_axis1_x[iTrack]) + (minor_axis1_y[iTrack] * minor_axis1_y[iTrack]) + (minor_axis1_z[iTrack] * minor_axis1_z[iTrack])
            minor_axis2_magSq_filt = (minor_axis2_x[iTrack] * minor_axis2_x[iTrack]) + (minor_axis2_y[iTrack] * minor_axis2_y[iTrack]) + (minor_axis2_z[iTrack] * minor_axis2_z[iTrack])
        
            major_axis_magSqSq_filt  = major_axis_magSq_filt *major_axis_magSq_filt
            minor_axis1_magSqSq_filt = minor_axis1_magSq_filt*minor_axis1_magSq_filt
            minor_axis2_magSqSq_filt = minor_axis2_magSq_filt*minor_axis2_magSq_filt
            
            for bx in range(_nbins_xy):
                vec_dot_major_axis_x_filt  = vec_dot_major_axis_x[bx][iTrack]
                vec_dot_minor_axis1_x_filt = vec_dot_minor_axis1_x[bx][iTrack]
                vec_dot_minor_axis2_x_filt = vec_dot_minor_axis2_x[bx][iTrack]
                for by in range(_nbins_xy):                
                    vec_dot_major_axis_y_filt  = vec_dot_major_axis_y[bx][iTrack]
                    vec_dot_minor_axis1_y_filt = vec_dot_minor_axis1_y[bx][iTrack]
                    vec_dot_minor_axis2_y_filt = vec_dot_minor_axis2_y[bx][iTrack]
                    poca_PDF = get_PDFs_XY_precalculated(pz,
                                                         poca_z_filt,
                                                         major_axis_z_filt,
                                                         minor_axis1_z_filt,
                                                         minor_axis2_z_filt,
                                                         poca_det_filt,
                                                         vec_dot_major_axis_x_filt,
                                                         vec_dot_minor_axis1_x_filt,
                                                         vec_dot_minor_axis2_x_filt,
                                                         vec_dot_major_axis_y_filt,
                                                         vec_dot_minor_axis1_y_filt,
                                                         vec_dot_minor_axis2_y_filt,
                                                         major_axis_magSqSq_filt,
                                                         minor_axis1_magSqSq_filt,
                                                         minor_axis2_magSqSq_filt
                                                        )
                    #sum_poca_PDFs += poca_PDF
        '''
        ## Filter all tracks ak arrays:        
        poca_x_filt        = get_tracks_in_range(poca_x,filt)
        poca_y_filt        = get_tracks_in_range(poca_y,filt)
        poca_z_filt        = get_tracks_in_range(poca_z,filt)
        major_axis_x_filt  = get_tracks_in_range(major_axis_x,filt)
        major_axis_y_filt  = get_tracks_in_range(major_axis_y,filt)
        major_axis_z_filt  = get_tracks_in_range(major_axis_z,filt)
        minor_axis1_x_filt = get_tracks_in_range(minor_axis1_x,filt)
        minor_axis1_y_filt = get_tracks_in_range(minor_axis1_y,filt)
        minor_axis1_z_filt = get_tracks_in_range(minor_axis1_z,filt)
        minor_axis2_x_filt = get_tracks_in_range(minor_axis2_x,filt)
        minor_axis2_y_filt = get_tracks_in_range(minor_axis2_y,filt)
        minor_axis2_z_filt = get_tracks_in_range(minor_axis2_z,filt)
        poca_det_filt      = get_tracks_in_range(poca_det,filt)
        
        
        major_axis_magSq_filt  = (major_axis_x_filt * major_axis_x_filt) + (major_axis_y_filt * major_axis_y_filt) + (major_axis_z_filt * major_axis_z_filt)
        minor_axis1_magSq_filt = (minor_axis1_x_filt * minor_axis1_x_filt) + (minor_axis1_y_filt * minor_axis1_y_filt) + (minor_axis1_z_filt * minor_axis1_z_filt)
        minor_axis2_magSq_filt = (minor_axis2_x_filt * minor_axis2_x_filt) + (minor_axis2_y_filt * minor_axis2_y_filt) + (minor_axis2_z_filt * minor_axis2_z_filt)
        
        major_axis_magSqSq_filt  = major_axis_magSq_filt *major_axis_magSq_filt
        minor_axis1_magSqSq_filt = minor_axis1_magSq_filt*minor_axis1_magSq_filt
        minor_axis2_magSqSq_filt = minor_axis2_magSq_filt*minor_axis2_magSq_filt
        
        for bx in range(_nbins_xy):
            vec_dot_major_axis_x_filt  = get_tracks_in_range(vec_dot_major_axis_x[bx],filt)
            vec_dot_minor_axis1_x_filt = get_tracks_in_range(vec_dot_minor_axis1_x[bx],filt)
            vec_dot_minor_axis2_x_filt = get_tracks_in_range(vec_dot_minor_axis2_x[bx],filt)
            for by in range(_nbins_xy):                
                vec_dot_major_axis_y_filt = get_tracks_in_range(vec_dot_major_axis_y[by],filt)
                vec_dot_minor_axis1_y_filt = get_tracks_in_range(vec_dot_minor_axis1_y[bx],filt)
                vec_dot_minor_axis2_y_filt = get_tracks_in_range(vec_dot_minor_axis2_y[bx],filt)
                poca_PDFs = get_PDFs_XY_precalculated(pz,
                                                      poca_z_filt,
                                                      major_axis_z_filt,
                                                      minor_axis1_z_filt,
                                                      minor_axis2_z_filt,
                                                      poca_det_filt,
                                                      vec_dot_major_axis_x_filt,
                                                      vec_dot_minor_axis1_x_filt,
                                                      vec_dot_minor_axis2_x_filt,
                                                      vec_dot_major_axis_y_filt,
                                                      vec_dot_minor_axis1_y_filt,
                                                      vec_dot_minor_axis2_y_filt,
                                                      major_axis_magSqSq_filt,
                                                      minor_axis1_magSqSq_filt,
                                                      minor_axis2_magSqSq_filt
                                                     )
                KDE_XY = np.sum(poca_PDFs)
                if KDE_XY > sum_poca_PDFs:
                    sum_poca_PDFs = KDE_XY
                    best_x_bin = bx
                    best_y_bin = by
                poca_PDFs = get_PDFs(px, py, pz,
                                     poca_x_filt, poca_y_filt, poca_z_filt,
                                     major_axis_x_filt,  major_axis_y_filt,  major_axis_z_filt,
                                     minor_axis1_x_filt, minor_axis1_y_filt, minor_axis1_z_filt,
                                     minor_axis2_x_filt, minor_axis2_y_filt, minor_axis2_z_filt,
                                     poca_det_filt                                          
                                    )
                KDE_XY = np.sum(poca_PDFs)
                if KDE_XY > sum_poca_PDFs:
                    sum_poca_PDFs = KDE_XY
                    best_x_bin = bx
                    best_y_bin = by
        '''
'''               
## ============================================================================================================    
def Compute_updated_poca_KDE(input_tuple, n_bins_poca_kde):
    
    ## ************************************************************    
    ## Get the tracks information from the input tuple
    poca_x        = getattr(input_tuple, "poca_x")
    poca_y        = getattr(input_tuple, "poca_y")
    poca_z        = getattr(input_tuple, "poca_z")
    major_axis_x  = getattr(input_tuple, "major_axis_x")
    major_axis_y  = getattr(input_tuple, "major_axis_y")
    major_axis_z  = getattr(input_tuple, "major_axis_z")
    minor_axis1_x = getattr(input_tuple, "minor_axis1_x")
    minor_axis1_y = getattr(input_tuple, "minor_axis1_y")
    minor_axis1_z = getattr(input_tuple, "minor_axis1_z")
    minor_axis2_x = getattr(input_tuple, "minor_axis2_x")
    minor_axis2_y = getattr(input_tuple, "minor_axis2_y")
    minor_axis2_z = getattr(input_tuple, "minor_axis2_z")
    tx            = getattr(input_tuple, "recon_tx")
    ty            = getattr(input_tuple, "recon_ty")    
    ## Now calculate the determinant that is needed to compute the KDEs        
    poca_det      = get_determinant(poca_x, poca_y, 
                                    major_axis_z, minor_axis1_x,
                                    tx, ty)

    ## -------------------------------------------------------
    ## Get the number of events to loop over 
    nEvts = len(poca_x)
    print("Looping over %s events..."%nEvts)
    
    ## -------------------------------------------------------
    ## Defining a dict to store all the poca KDE values 
    d_poca_KDEs = {"KDE_A":     np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_A_xmax":np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_A_ymax":np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_B":     np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_B_xmax":np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_B_ymax":np.empty((nEvts,n_bins_poca_kde)),
                  }  
    
    
    for iEvt in range(nEvts): 
        print("At event",iEvt)
        
        poca_x_i        = ak.to_numpy(poca_x[iEvt])
        poca_y_i        = ak.to_numpy(poca_y[iEvt])
        poca_z_i        = ak.to_numpy(poca_z[iEvt])
        major_axis_x_i  = ak.to_numpy(major_axis_x[iEvt])
        major_axis_y_i  = ak.to_numpy(major_axis_y[iEvt])
        major_axis_z_i  = ak.to_numpy(major_axis_z[iEvt])
        minor_axis1_x_i = ak.to_numpy(minor_axis1_x[iEvt])
        minor_axis1_y_i = ak.to_numpy(minor_axis1_y[iEvt])
        minor_axis1_z_i = ak.to_numpy(minor_axis1_z[iEvt])
        minor_axis2_x_i = ak.to_numpy(minor_axis2_x[iEvt])
        minor_axis2_y_i = ak.to_numpy(minor_axis2_y[iEvt])
        minor_axis2_z_i = ak.to_numpy(minor_axis2_z[iEvt])
        poca_det_i      = ak.to_numpy(poca_det[iEvt])
        
        compute_poca_PDFs(n_bins_poca_kde,
                          poca_x_i, poca_y_i, poca_z_i,
                          major_axis_x_i,  major_axis_y_i,  major_axis_z_i,
                          minor_axis1_x_i, minor_axis1_y_i, minor_axis1_z_i,
                          minor_axis2_x_i, minor_axis2_y_i, minor_axis2_z_i,
                          poca_det_i
                         )

        
    return InputData_tuple(
        
        getattr(input_tuple, "pv_x"),
        getattr(input_tuple, "pv_y"),
        getattr(input_tuple, "pv_z"),
        getattr(input_tuple, "pv_ntracks"),
        getattr(input_tuple, "pv_cat"),
        getattr(input_tuple, "pv_key"), 

        getattr(input_tuple, "sv_x"),
        getattr(input_tuple, "sv_y"),
        getattr(input_tuple, "sv_z"),
        getattr(input_tuple, "sv_ntracks"),
        getattr(input_tuple, "sv_cat"),
        getattr(input_tuple, "svr_pv_key"), 

        getattr(input_tuple, "poca_x"),               
        getattr(input_tuple, "poca_y"),              
        getattr(input_tuple, "poca_z"),              
        getattr(input_tuple, "major_axis_x"),        
        getattr(input_tuple, "major_axis_y"),        
        getattr(input_tuple, "major_axis_z"),        
        getattr(input_tuple, "minor_axis1_x"),       
        getattr(input_tuple, "minor_axis1_y"),       
        getattr(input_tuple, "minor_axis1_z"),       
        getattr(input_tuple, "minor_axis2_x"),       
        getattr(input_tuple, "minor_axis2_y"),       
        getattr(input_tuple, "minor_axis2_z"),       

        getattr(input_tuple, "recon_pv_key"), 

        getattr(input_tuple, "recon_tx"), 
        getattr(input_tuple, "recon_ty"), 

        getattr(input_tuple, "IP_KDE"),
        getattr(input_tuple, "IP_KDE_xMax"),
        getattr(input_tuple, "IP_KDE_yMax"),
        
        getattr(input_tuple, "poca_KDE_A"),
        getattr(input_tuple, "poca_KDE_A_xMax"),
        getattr(input_tuple, "poca_KDE_A_yMax"),
        getattr(input_tuple, "poca_KDE_B"),
        getattr(input_tuple, "poca_KDE_B_xMax"),
        getattr(input_tuple, "poca_KDE_B_yMax"),
    ) 
        
'''        
def Compute_updated_poca_KDE(input_tuple, n_bins_poca_kde):
    
    ## ************************************************************    
    ## Get the tracks information from the input tuple
    poca_x        = getattr(input_tuple, "poca_x")
    poca_y        = getattr(input_tuple, "poca_y")
    poca_z        = getattr(input_tuple, "poca_z")
    major_axis_x  = getattr(input_tuple, "major_axis_x")
    major_axis_y  = getattr(input_tuple, "major_axis_y")
    major_axis_z  = getattr(input_tuple, "major_axis_z")
    minor_axis1_x = getattr(input_tuple, "minor_axis1_x")
    minor_axis1_y = getattr(input_tuple, "minor_axis1_y")
    minor_axis1_z = getattr(input_tuple, "minor_axis1_z")
    minor_axis2_x = getattr(input_tuple, "minor_axis2_x")
    minor_axis2_y = getattr(input_tuple, "minor_axis2_y")
    minor_axis2_z = getattr(input_tuple, "minor_axis2_z")
    tx            = getattr(input_tuple, "recon_tx")
    ty            = getattr(input_tuple, "recon_ty")
    
    poca_det      = get_determinant(poca_x, poca_y, 
                                    major_axis_z, minor_axis1_x,
                                    tx, ty)
    
    nEvts = len(poca_x)
    
    ## -------------------------------------------------------
    ## Defining a dict to store all the poca KDE values 
    d_poca_KDEs = {"KDE_A":     np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_A_xmax":np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_A_ymax":np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_B":     np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_B_xmax":np.empty((nEvts,n_bins_poca_kde)),
                   "KDE_B_ymax":np.empty((nEvts,n_bins_poca_kde)),
                  }  
    
    d_poca_x_filt        = {}
    d_poca_y_filt        = {}
    d_poca_z_filt        = {}
    d_major_axis_x_filt  = {}
    d_major_axis_y_filt  = {}
    d_major_axis_z_filt  = {}
    d_minor_axis1_x_filt = {}
    d_minor_axis1_y_filt = {}
    d_minor_axis1_z_filt = {}
    d_minor_axis2_x_filt = {}
    d_minor_axis2_y_filt = {}
    d_minor_axis2_z_filt = {}
    d_poca_det_filt      = {}
    
    ## -------------------------------------------------------
    ## Let start by filtering tracks of interest in each z bin
    ## Looping over z bins:
    for bz in range(n_bins_poca_kde):
        if bz%50==0:
            print("At z bin:",bz)
        ## Get z center of the bin 'bz':
        z = get_bin_center(n_bins_poca_kde, _zmin, _zmax, bz)

        ## Get filter to remove tracks not of interest for z bin 'bz':        
        filt = get_filter_tracks_in_range(z, poca_z, major_axis_z)

        d_poca_x_filt[bz]        = get_tracks_in_range(poca_x,filt)
        d_poca_y_filt[bz]        = get_tracks_in_range(poca_y,filt)
        d_poca_z_filt[bz]        = get_tracks_in_range(poca_z,filt)
        d_major_axis_x_filt[bz]  = get_tracks_in_range(major_axis_x,filt)
        d_major_axis_y_filt[bz]  = get_tracks_in_range(major_axis_y,filt)
        d_major_axis_z_filt[bz]  = get_tracks_in_range(major_axis_z,filt)
        d_minor_axis1_x_filt[bz] = get_tracks_in_range(minor_axis1_x,filt)
        d_minor_axis1_y_filt[bz] = get_tracks_in_range(minor_axis1_y,filt)
        d_minor_axis1_z_filt[bz] = get_tracks_in_range(minor_axis1_z,filt)
        d_minor_axis2_x_filt[bz] = get_tracks_in_range(minor_axis2_x,filt)
        d_minor_axis2_y_filt[bz] = get_tracks_in_range(minor_axis2_y,filt)
        d_minor_axis2_z_filt[bz] = get_tracks_in_range(minor_axis2_z,filt)
        ## Now calculate the determinant that is needed to compute the KDEs        
        d_poca_det_filt[bz]      = get_tracks_in_range(poca_det,filt)
    
    
    
    '''
    ## -------------------------------------------------------
    ## Looping over z bins:
    for bz in range(n_bins_poca_kde):
        print("At z bin:",bz)
        ## Get z center of the bin 'bz':
        z = get_bin_center(n_bins_poca_kde, _zmin, _zmax, bz)

        ## Get filter to remove tracks not of interest for z bin 'bz':        
        filt = get_filter_tracks_in_range(z, poca_z, major_axis_z)

      
        ## Filter all tracks ak arrays:     
        poca_x_filt        = get_tracks_in_range(poca_x,filt)
        poca_y_filt        = get_tracks_in_range(poca_y,filt)
        poca_z_filt        = get_tracks_in_range(poca_z,filt)
        major_axis_x_filt  = get_tracks_in_range(major_axis_x,filt)
        major_axis_y_filt  = get_tracks_in_range(major_axis_y,filt)
        major_axis_z_filt  = get_tracks_in_range(major_axis_z,filt)
        minor_axis1_x_filt = get_tracks_in_range(minor_axis1_x,filt)
        minor_axis1_y_filt = get_tracks_in_range(minor_axis1_y,filt)
        minor_axis1_z_filt = get_tracks_in_range(minor_axis1_z,filt)
        minor_axis2_x_filt = get_tracks_in_range(minor_axis2_x,filt)
        minor_axis2_y_filt = get_tracks_in_range(minor_axis2_y,filt)
        minor_axis2_z_filt = get_tracks_in_range(minor_axis2_z,filt)
        ## Now calculate the determinant that is needed to compute the KDEs        
        poca_det_filt      = get_tracks_in_range(poca_det,filt)
        
        ## We want to transform the ak arrays to np arrays to use numba and increase speed!
        ## Before we need to flatten the ak arrays to be able to convert them into np arrays
        ## Start by keeping tracks of the shape of the initial ak arrays to transform back the np output
        shape = ak.num(get_tracks_in_range(filt,filt))
        
        poca_x_filt_flat_np        = ak.to_numpy(ak.flatten(poca_x_filt))
        poca_y_filt_flat_np        = ak.to_numpy(ak.flatten(poca_y_filt))
        poca_z_filt_flat_np        = ak.to_numpy(ak.flatten(poca_z_filt))
        major_axis_x_filt_flat_np  = ak.to_numpy(ak.flatten(major_axis_x_filt))
        major_axis_y_filt_flat_np  = ak.to_numpy(ak.flatten(major_axis_y_filt))
        major_axis_z_filt_flat_np  = ak.to_numpy(ak.flatten(major_axis_z_filt))
        minor_axis1_x_filt_flat_np = ak.to_numpy(ak.flatten(minor_axis1_x_filt))
        minor_axis1_y_filt_flat_np = ak.to_numpy(ak.flatten(minor_axis1_y_filt))
        minor_axis1_z_filt_flat_np = ak.to_numpy(ak.flatten(minor_axis1_z_filt))
        minor_axis2_x_filt_flat_np = ak.to_numpy(ak.flatten(minor_axis2_x_filt))
        minor_axis2_y_filt_flat_np = ak.to_numpy(ak.flatten(minor_axis2_y_filt))
        minor_axis2_z_filt_flat_np = ak.to_numpy(ak.flatten(minor_axis2_z_filt))
        poca_det_filt_flat_np      = ak.to_numpy(ak.flatten(poca_det_filt))

        poca_PDFs_flat_np = get_poca_PDFs(z,
                                          poca_x_filt_flat_np, poca_y_filt_flat_np, poca_z_filt_flat_np,
                                          major_axis_x_filt_flat_np,  major_axis_y_filt_flat_np,  major_axis_z_filt_flat_np,
                                          minor_axis1_x_filt_flat_np, minor_axis1_y_filt_flat_np, minor_axis1_z_filt_flat_np,
                                          minor_axis2_x_filt_flat_np, minor_axis2_y_filt_flat_np, minor_axis2_z_filt_flat_np,
                                          poca_det_filt_flat_np                                          
                                         )
    '''
        