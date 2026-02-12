'''
Methods to sort and manipulate data for PVs and tracks information.
'''

import awkward as ak
import numpy as np
from tools.named_tuples_def import InputData_tuple
from utils.utilities import Timer

def Sort_PVs_by_Z_and_update_PV_key(input_tuple, nTracks_ValidPVs = 0):

    ## ************************************************************    
    ## Get the PV information from the input tuple
    pv_x = getattr(input_tuple, "pv_x")
    pv_y = getattr(input_tuple, "pv_y")
    pv_z = getattr(input_tuple, "pv_z")
    pv_n = getattr(input_tuple, "pv_ntracks")
    pv_c = getattr(input_tuple, "pv_cat")
    pv_k = getattr(input_tuple, "pv_key")
        
    ## Get the indexes for the sorting by Z as an ak array 
    pv_index_by_z   = ak.argsort(pv_z)

    ## Sort all input PV information by Z 
    pv_x_sorted = pv_x[pv_index_by_z]
    pv_y_sorted = pv_y[pv_index_by_z]
    pv_z_sorted = pv_z[pv_index_by_z]
    pv_n_sorted = pv_n[pv_index_by_z]
    pv_c_sorted = pv_c[pv_index_by_z]
    pv_k_sorted = pv_k[pv_index_by_z]
        
    ## Filter PV information with a condition on 'pv_ntracks'
    pv_x_sorted_filt = pv_x_sorted[pv_n_sorted>nTracks_ValidPVs]
    pv_y_sorted_filt = pv_y_sorted[pv_n_sorted>nTracks_ValidPVs]
    pv_z_sorted_filt = pv_z_sorted[pv_n_sorted>nTracks_ValidPVs]
    pv_n_sorted_filt = pv_n_sorted[pv_n_sorted>nTracks_ValidPVs]
    pv_c_sorted_filt = pv_c_sorted[pv_n_sorted>nTracks_ValidPVs]
    pv_k_sorted_filt = pv_k_sorted[pv_n_sorted>nTracks_ValidPVs]
                
    ## Since it seems more 'natural' to use keys ranging from 0 to N 
    ## once the PVs are sorted by Z, simply update the pv keys with the 
    ## indexes of the newly sorted and filtered pv_z ak array.
    pv_k_updated = ak.argsort(pv_z_sorted_filt)
            
    ## Since only PV information changed, create a new namedtuple to be returned
    ## with all other information from the input_tuple
    output_tuple = InputData_tuple(
        
        pv_x_sorted_filt,
        pv_y_sorted_filt,
        pv_z_sorted_filt,
        pv_n_sorted_filt,
        pv_c_sorted_filt,
        pv_k_updated, 

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
    
    ## ************************************************************    
    ## In order to maintain track to PV key information, we simply
    ## create a dictionnary with the correspondance between old PV keys 
    ## and updated PV keys. 
    ## This dictionnary is an output of the method and can be used later 
    ## when sorting tracks for example.
    ##
    ## Dictionnary structure is: 
    ##
    ## {'evt_0': 
    ##         {-999: -1, 
    ##          1st_old_key_by_z: 0, 
    ##          2nd_old_key_by_z: 1,
    ##          3rd_old_key_by_z: 2,
    ##          .
    ##          .
    ##          .
    ##          Nth_old_key_by_z: N-1    
    ##         }
    ## }
    ##
    ## Note that the pair {key = -999: value = -1} is added because some tracks are 
    ## not actually matched to a MC track, meaning that they do not match any PV.
    d_pv_k = {}
    for evt, pv_k_sorted_filt_keys in enumerate(pv_k_sorted_filt):        
        d_pv_k[evt] = {}
        d_pv_k[evt][-999] = -1
        for i, key in enumerate(pv_k_sorted_filt_keys):
            d_pv_k[evt][key] = i
        '''
        if evt<10:
            print('d_pv_k_sorted_new[%s]'%evt,d_pv_k[evt])
        elif evt==175:
            print('d_pv_k_sorted_new[%s]'%evt,d_pv_k[evt])
        '''
    ## ************************************************************    
        
        
    return output_tuple, d_pv_k


def Sort_tracks_by_Z_and_update_T2PV_key(input_tuple, d_pv_key):

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
    recon_tx      = getattr(input_tuple, "recon_tx")
    recon_ty      = getattr(input_tuple, "recon_ty")
    t2pv_key      = getattr(input_tuple, "recon_pv_key")
                
    ## Get the indexes for the sorting by Z as an ak array 
    tracks_index_by_z = ak.argsort(poca_z)

    ## Sort all input tracks information by Z 
    poca_x_sorted        = poca_x[tracks_index_by_z]
    poca_y_sorted        = poca_y[tracks_index_by_z]
    poca_z_sorted        = poca_z[tracks_index_by_z]
    major_axis_x_sorted  = major_axis_x[tracks_index_by_z]
    major_axis_y_sorted  = major_axis_y[tracks_index_by_z]
    major_axis_z_sorted  = major_axis_z[tracks_index_by_z]
    minor_axis1_x_sorted = minor_axis1_x[tracks_index_by_z]
    minor_axis1_y_sorted = minor_axis1_y[tracks_index_by_z]
    minor_axis1_z_sorted = minor_axis1_z[tracks_index_by_z]
    minor_axis2_x_sorted = minor_axis2_x[tracks_index_by_z]
    minor_axis2_y_sorted = minor_axis2_y[tracks_index_by_z]
    minor_axis2_z_sorted = minor_axis2_z[tracks_index_by_z]
    recon_tx_sorted      = recon_tx[tracks_index_by_z]
    recon_ty_sorted      = recon_ty[tracks_index_by_z]
    ## Let us use a temporary ak array for the tracks to PV keys 
    t2pv_key_sorted_tmp  = t2pv_key[tracks_index_by_z]

    ## We now want to modify the tracks to PV keys: 
    ## To do that we need to convert the ak array into a np array as 
    ## it is not possible to change ak array values.
    ## Let start by flattening the ak array and saving the shape of 
    ## the original ak array to allow unflattening it at the end of the process.
    t2pv_key_sorted_flattened, t2pv_key_shape = ak.flatten(t2pv_key_sorted_tmp), ak.num(t2pv_key_sorted_tmp)
    ## Create an empty np array of same legnth as the flattened ak array.
    t2pv_key_sorted_flattened_np = np.empty(ak.to_numpy(t2pv_key_sorted_flattened).shape)

    ## Here starts the loop on the tracks to PV keys.
    ## Since we stored a dictionnary with the correspondance between the 
    ## old and updated PV keys, we need to loop over the tracks to PV keys (evt by evt)
    ## and for each evt loop over all track to create the updated tracks to PV keys
    ## NOTE: this is going to be slow...might want to consider having a better piece of code here
    ##
    ## The counter 'track_flat_i' is used to identiy the tracks position in the (flattened) np array.
    track_flat_i=0
    d_tracks_not_associated = {}
    ## Loop over the temporary ak arrays for the tracks to PV keys (size N_events)
    for evt, t2pv_keys in enumerate(t2pv_key_sorted_tmp):
        d_tracks_not_associated[evt] = 0
        ## Loop over each temporary ak array for the tracks to PV keys (size N_reco_tracks)
        for i, t2pv_key in enumerate(t2pv_keys):
            ## Check if the t2pv_key is in the dictionnary with the correspondance 
            ## between the  old and updated PV keys. Indeed, we removed PVs with 0 tracks, 
            ## but it seems some reconstructed tracks are nevertheless associated to some 
            ## of the removed PVs. If so the final t2pv_key is set to -1, i.e. no PV associated
            if t2pv_key in d_pv_key[evt]:
                ## Fill the np array with the updated t2pv_key
                t2pv_key_sorted_flattened_np[track_flat_i] = d_pv_key[evt][t2pv_key]
            else:
                ## Let us keep track of how often this happens using 
                t2pv_key_sorted_flattened_np[track_flat_i] = -1
                d_tracks_not_associated[evt] += 1
                
            track_flat_i+=1
    ## Convert back the flattened np array into a flat ak array
    t2pv_key_sorted_flattened = ak.from_numpy(t2pv_key_sorted_flattened_np)
    ## Unflattened the final ak array to the original shape
    t2pv_key_sorted = ak.unflatten(t2pv_key_sorted_flattened,t2pv_key_shape)

    ## Print out the non associated tracks for information. 
    for evt in d_tracks_not_associated.keys():
        if not d_tracks_not_associated[evt]==0:
            print("---> Event %s has %s tracks finally not associated to a PV"%(evt,d_tracks_not_associated[evt]))
              
    ## Since only tracks information changed, create a new namedtuple to be returned
    ## with all other information from the input_tuple
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

        poca_x_sorted,               
        poca_y_sorted,              
        poca_z_sorted,              
        major_axis_x_sorted,        
        major_axis_y_sorted,        
        major_axis_z_sorted,        
        minor_axis1_x_sorted,       
        minor_axis1_y_sorted,       
        minor_axis1_z_sorted,       
        minor_axis2_x_sorted,       
        minor_axis2_y_sorted,       
        minor_axis2_z_sorted,       

        t2pv_key_sorted, 

        recon_tx_sorted, 
        recon_ty_sorted, 

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
    