'''
Methods to compute the target histograms when using POCA KDE.

These gaussian like distributions with widths based on a formula relating the PV resolution with the number of tracks originating from the PVs such that:

pv_res = A_res*np.power(nTrks, -1*B_res) + C_res

where A_res, B_res and C_res are constants determined from simulation for the Upgrade VELO TDR LHCb-PUB-2017-005.

The true number of tracks, nTrks, associated to each true PV is used here.

'''

import awkward as ak
import numpy as np
from utils.utilities import Timer
from tools.ToolBox import norm_cdf, get_res
from tools.named_tuples_def import InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple

def Compute_target_hists(input_tuple, n_bins_target_hist, n_tracks_valid_PVs, config):

    '''
    IMPORTANT !!! 
    
    Note that for the moment only target histogram including valid PVs are considered 
    when constructing the target histograms. This can easily be modified in the future!
    '''
    print("")
    print("-"*100)
    print("    ----------------------------------- ")
    print("    >>>>>>>>>>>>  WARNING  <<<<<<<<<<<< ")
    print("    ----------------------------------- ")
    print("")
    print(" Only valid primary vertexes are considered when constructing the target histograms")
    print(" i.e. for PV(nTracks>%s)"%n_tracks_valid_PVs)
    print("")
    print("-"*100)
    print("")

    ## Let start by retrieving the vertexes true number of tracks and category
    pv_z     = getattr(input_tuple, "pv_z")
    pv_ntrks = getattr(input_tuple, "pv_ntracks")
    pv_cat   = getattr(input_tuple, "pv_cat") 
    ## Currently only considering PVs (uncomment the correponding lines to include other vertexes)
    '''
    sv_z     = getattr(input_tuple, "sv_z")
    sv_ntrks = getattr(input_tuple, "sv_ntracks")
    sv_cat   = getattr(input_tuple, "sv_cat") 
    '''
    
    ## Get the resolutions for all vertexes
    pv_res   = get_res(pv_ntrks, config)
    ## Currently only considering PVs (uncomment the correponding lines to include other vertexes)
    '''
    sv_res   = get_res(sv_ntrks, config)
    '''
    ## Define here a default minimal resolution (in [mm])
    res_min = config['res_min_target']
    ## The vertexes true resolution as a function of nTracks will be increased 
    ## if res < 150 microns such that the target peaks will get more narrow and 
    ## the areas larger as the number of tracks increases.
    pv_res_scale = np.where((res_min / pv_res) > 1, (res_min / pv_res), 1)
    '''
    sv_res_scale = np.where((0.15 / sv_res) > 1, (0.15 / sv_res), 1)
    '''

    ## Keep record of the number of events
    n_Evts   = len(pv_ntrks)
    
    ## Here comes the definitions for constructing the target hist structure
    z_min = config['z_min'] 
    z_max = config['z_max'] 
    z_range = config['z_max'] - config['z_min']
    
    ## Define the z_range over n_bins_target_hist
    zRange_over_nBins_ratio = n_bins_target_hist / z_range
    ## Bins width definition
    bin_width = z_range/n_bins_target_hist
    half_bin_width = bin_width/2.
    
    ## Edges of the bins definition
    edges = np.array([-half_bin_width, half_bin_width])
    ## Range of z values of the target hists
    zvals_range = (z_min + half_bin_width, z_max - half_bin_width)

    ## Get the bin number corresponding to the PV position in Z: 
    pv_bin = ak.values_astype(np.floor((pv_z - zvals_range[0]) * zRange_over_nBins_ratio), "int16")
    '''
    sv_bin = ak.values_astype(np.floor((sv_z - zvals_range[0]) * zRange_over_nBins_ratio), "int16")
    '''
    ## This is a placeholder to hold the values of the vertexes peak structure
    ## +/- 5 bins around peak maximum seems a good compromise, as for more bins the 
    ## hist values become quickly tiny. 
    nBins_around_vtx_peak = 5    
    vtx_bins = np.arange(-nBins_around_vtx_peak, nBins_around_vtx_peak+1)
    
    ## Define the bins basis for which the normal cumulative distribution will be evaluated
    ## For each vertex, the vxt position (in z basis) will be added
    ## The output is a set of two list values corresponding to the equivalent of bins edges shifted by one bin
    vtx_z_bins_ranges = bin_width * vtx_bins[np.newaxis, :] + edges[:, np.newaxis] + zvals_range[0]
    
    ## To keep the same logic as originaly implemented in PV-finder, we keep a [4 * n_Evts * nBins] shape
    ## for the target hist, with four targetHists, one for each of the following cases:
    ##      0==valid PVs, 
    ##      1==unvalid PVs,
    ##      2==valid SVs, 
    ##      3==unvalid SVs
    targetHists_type = np.float16 # This param needs to go in a general configuration file 
    targetHists = np.zeros([4, n_Evts, n_bins_target_hist], dtype=targetHists_type)
            
    ## Here starts the loop on all events (this is certainly not ideal and could be improved)
    ## However, this method is supposed to be called once, so not a huge issue is this is slow
    for iEvt in range(n_Evts):

        ## Get a structured tuple of shape [n_vtxCats * (<vtx_z>, <vtx_bin>, <vtx_res>, <vtx_res_scale>)]
        ## Currently only for valid and unvalid PVs. 
        ## To include other vertexes, get the correponding lines from the commented out example below...
        all_vtx_info = (
            ## <vtx_z>, <vtx_bin>, <vtx_nTracks> for valid PVs
            (pv_z[iEvt][pv_cat[iEvt]   == 1], 
             pv_bin[iEvt][pv_cat[iEvt] == 1], 
             pv_res[iEvt][pv_cat[iEvt] == 1], 
             pv_res_scale[iEvt][pv_cat[iEvt] == 1]),            
            ## <vtx_z>, <vtx_nTracks> for unvalid PVs
            (pv_z[iEvt][pv_cat[iEvt]   != 1], 
             pv_bin[iEvt][pv_cat[iEvt] != 1], 
             pv_res[iEvt][pv_cat[iEvt] != 1],
             pv_res_scale[iEvt][pv_cat[iEvt] != 1]),
        )
        '''
        all_vtx_info = (
            ## <vtx_z>, <vtx_bin>, <vtx_nTracks> for valid PVs
            (pv_z[iEvt][pv_cat[iEvt]   == 1], 
             pv_bin[iEvt][pv_cat[iEvt] == 1], 
             pv_res[iEvt][pv_cat[iEvt] == 1], 
             pv_res_scale[iEvt][pv_cat[iEvt] == 1]),            
            ## <vtx_z>, <vtx_nTracks> for unvalid PVs
            (pv_z[iEvt][pv_cat[iEvt]   != 1], 
             pv_bin[iEvt][pv_cat[iEvt] != 1], 
             pv_res[iEvt][pv_cat[iEvt] != 1],
             pv_res_scale[iEvt][pv_cat[iEvt] != 1]),
            ## <vtx_z>, <vtx_nTracks> for valid SVs, mostly nTracks==2
            (sv_z[iEvt][sv_cat[iEvt]   == 1], 
             sv_bin[iEvt][sv_cat[iEvt] == 1], 
             sv_res[iEvt][sv_cat[iEvt] == 1], 
             sv_res_scale[iEvt][sv_cat[iEvt] == 1]),
            ## <vtx_z>, <vtx_nTracks> for unvalid SVs, with nTracks==1
            (sv_z[iEvt][sv_cat[iEvt]   != 1], 
             sv_bin[iEvt][sv_cat[iEvt] != 1], 
             sv_res[iEvt][sv_cat[iEvt] != 1], 
             sv_res_scale[iEvt][sv_cat[iEvt] != 1]),
        )
        '''
                
        ## Looping over the vertexes categories and getting the following info:
        ##   vtx_info[0] == <vtx_z>
        ##   vtx_info[1] == <vtx_bin>
        ##   vtx_info[2] == <vtx_res>
        ##   vtx_info[3] == <vtx_res_scale>
                
        for vtxCat, vtx_info in enumerate(all_vtx_info):
            
            ## Get the number of vertexes
            nVtx = len(vtx_info[0])

            ## Looping over the vertexes
            for iVtx in range(nVtx):

                vtx_z         = vtx_info[0][iVtx]                
                ## Restrict to the valid range in z
                if (vtx_z<z_min or vtx_z>z_max): continue
                
                vtx_bin       = vtx_info[1][iVtx]
                vtx_res       = vtx_info[2][iVtx]
                vtx_res_scale = vtx_info[3][iVtx]
                                
                ## Get the bins like values of z to evaluate the normal cumulative distribution
                vtx_z_bins    = (vtx_bin / zRange_over_nBins_ratio + vtx_z_bins_ranges)
                
                ## Get the values of the normal cumulative ditribution for the two set of vtx_z_bins
                norm_cdf_vals = norm_cdf(vtx_z, vtx_res, vtx_z_bins)
                ## Take the difference of norm_cdf_Vals to get the normal PDF values as vtx_peak_Vals
                ## The vtx_peak_vals are further scaled based on the vtx_res_scale values defined above
                
                vtx_peak_vals = (norm_cdf_vals[1] - norm_cdf_vals[0])*vtx_res_scale

                try:
                    targetHists[vtxCat, iEvt, vtx_bins + vtx_bin] += vtx_peak_vals
                    iEvt_target_hist = targetHists[vtxCat][iEvt]
                except IndexError:
                    msgs.append(
                        f"Ignored vertex at bin {vtx_bin} at {vtx_z:.4g} in event {iEvt}, vertex category {vtxCat}"
                    )
                    
        ## Update the valid PV target hist such that if the unvalid PV target hist 
        ## has any significant contribution, it is set to nan.
        ## This will allow to ignore the corresponding bins when evaluating the
        ## performances from the predicted hist.
        targetHists[0][iEvt][np.where(targetHists[1][iEvt]>0.001)] = np.nan
                
    return InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple(
    
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

        getattr(input_tuple, "poca_x"),               
        getattr(input_tuple, "poca_y"),              
        getattr(input_tuple, "poca_z"),              

        getattr(input_tuple, "poca_A"),               
        getattr(input_tuple, "poca_B"),              
        getattr(input_tuple, "poca_C"),              
        getattr(input_tuple, "poca_D"),               
        getattr(input_tuple, "poca_E"),              
        getattr(input_tuple, "poca_F"),              

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
        getattr(input_tuple, "recon_cat"), 
        getattr(input_tuple, "recon_ov_x"), 
        getattr(input_tuple, "recon_ov_y"), 
        getattr(input_tuple, "recon_ov_z"), 

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
        
        targetHists,
    ) 
    