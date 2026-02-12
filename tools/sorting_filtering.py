'''
Methods to sort and manipulate data for PVs and tracks information.
'''

import awkward as ak
import numpy as np
from tools.named_tuples_def import InputData_tuple, InputData_tuple_SV_OV_updated, InputData_tuple_GNN, InputData_tuple_SV_OV_updated_GNN
from utils.utilities import Timer

def Find_TrueSVs_and_TracksOriginVertex(input_tuple, GNN_preprocess=False):

    ## ================================================================================
    ## ================================================================================
    ## ************************************************************    
    ## Get the relevant PV information from the input tuple
    pv_x = getattr(input_tuple, "pv_x")
    pv_y = getattr(input_tuple, "pv_y")
    pv_z = getattr(input_tuple, "pv_z")
    pv_key = getattr(input_tuple, "pv_key")
    
    ## ************************************************************    
    ## Get the relevant SV information from the input tuple
    sv_x = getattr(input_tuple, "sv_x")
    sv_y = getattr(input_tuple, "sv_y")
    sv_z = getattr(input_tuple, "sv_z")
    sv_key = getattr(input_tuple, "sv_key")
    sv_pv_key = getattr(input_tuple, "sv_pv_key")
    
    ## ************************************************************    
    ## Get the relevant track information from the input tuple
    recon_pv_key      = getattr(input_tuple, "recon_pv_key")
    recon_ov_key      = getattr(input_tuple, "recon_ov_key")

    ## ************************************************************    
    ## Define the plain python lists that will be used to create the awkward arrays
    real_SV_x      = []
    real_SV_y      = []
    real_SV_z      = []
    real_SV_n      = []

    tracks_OV_x = []
    tracks_OV_y = []
    tracks_OV_z = []
    tracks_cat  = []

    ## ************************************************************    
    ## Define the tracks categories

    ## -----------------    
    ## prompt track:
    ## tracks originating from a PV, which can originates from a fake SV 
    ## that has the same coordinates than the PV it is associated with
    prompt_ID = 0  
    ## -----------------    
    ## secondary track:
    ## tracks originating from a true SV that has different same coordinates 
    ## than the PV it is associated with
    second_ID = 1
    ## -----------------    
    ## non associated track:
    ## tracks that has no PV association (failed to be associated with a MC track)
    none_ID   = -1


    ## ************************************************************    
    ## ************************************************************    
    ## We need to loop over each event, which is slow...
    
    ## Get the total number of events from the size of the pv ak array
    nEvts = len(pv_x)
    
    for iEvt in range(nEvts):

        ## ------------------------------------------------
        ## Define the list of true SV indices from which tracks originates from
        ## It will be used after the loop over tracks...
        true_SV_idx_Evt = []

        ## ------------------------------------------------
        ## Define the tracks originating vertex coordinates 
        ## and category lists within the current event
        tracks_OV_x_Evt = []
        tracks_OV_y_Evt = []
        tracks_OV_z_Evt = []
        tracks_cat_Evt  = []

        ## ------------------------------------------------
        ## Get the current event PV info
        pv_x_Evt         = np.array(pv_x[iEvt])
        pv_y_Evt         = np.array(pv_y[iEvt])
        pv_z_Evt         = np.array(pv_z[iEvt])
        pv_key_Evt       = np.array(pv_key[iEvt])
        ## ------------------------------------------------
        ## Get the current event SV info
        sv_x_Evt         = np.array(sv_x[iEvt])
        sv_y_Evt         = np.array(sv_y[iEvt])
        sv_z_Evt         = np.array(sv_z[iEvt])
        sv_pv_key_Evt    = np.array(sv_pv_key[iEvt])
        sv_key_Evt       = np.array(sv_key[iEvt])
        ## ------------------------------------------------
        ## Get the current event tracks <-> vertex association info
        recon_pv_key_Evt = np.array(recon_pv_key[iEvt])
        recon_ov_key_Evt = np.array(recon_ov_key[iEvt])

        ## ------------------------------------------------
        ## Get the current event number of tracks
        n_Tracks = len(recon_pv_key_Evt)
        
        ## ========================================================================
        ## Start looping over all tracks within the current event
        for iTrack in range(n_Tracks):
            ## Check if the current track is associated to any PV:
            ## - if yes: continue the investigation
            ## - if no:  set the track category to 'none_ID' value 
            ##           and the originating vertex coordinates to -999.
            if int(recon_pv_key_Evt[iTrack])!=-999:

                ## Check if the current track origin vertex is a PV:
                ## - if yes: set the track category to 'prompt_ID' value 
                ##           and the originating vertex coordinates to 
                ##           the corresponding PV values
                ## - if no:  continue the investigation 
                if recon_ov_key_Evt[iTrack] in pv_key_Evt:
                    pv_idx = np.where(recon_ov_key_Evt[iTrack]==pv_key_Evt)
                    tracks_OV_x_Evt.append(pv_x_Evt[pv_idx][0])
                    tracks_OV_y_Evt.append(pv_y_Evt[pv_idx][0])
                    tracks_OV_z_Evt.append(pv_z_Evt[pv_idx][0])
                    tracks_cat_Evt.append(prompt_ID)
                else:
                    ## Check if the current track origin vertex is a SV: 
                    ##   it should always be the case, but just to be sure,
                    ##   if it is not the case, a warning will be printed out
                    if recon_ov_key_Evt[iTrack] in sv_key_Evt:       

                        ## Get the z coordinate of the SV associated to the track  
                        sv_idx = np.where(recon_ov_key_Evt[iTrack]==sv_key_Evt)                       
                        z_sv_asso       = sv_z_Evt[sv_idx][0]

                        ## Get the z coordinate of the PV associated to the track  
                        pv_key_sv_asso  = int(sv_pv_key_Evt[sv_idx][0])                    
                        pv_idx = np.where(pv_key_sv_asso==pv_key_Evt)                    
                        z_pv_asso       = pv_z_Evt[pv_idx][0]

                        ## Check if the z coordinates are the same for the SV and PV:
                        ## - if no:  set the track category to 'secondary_ID' value 
                        ##           and the originating vertex coordinates to 
                        ##           the corresponding SV values
                        ## - if yes: set the track category to 'prompt_ID' value 
                        ##           and the originating vertex coordinates to 
                        ##           the corresponding PV values                        
                        if z_pv_asso!=z_sv_asso:
                            ## Append the SV coordinates to the tracks OV coordinates list
                            tracks_OV_x_Evt.append(sv_x_Evt[sv_idx][0])
                            tracks_OV_y_Evt.append(sv_y_Evt[sv_idx][0])
                            tracks_OV_z_Evt.append(z_sv_asso)
                            ## Append the tracks category 'second_ID' value to the tracks category list
                            tracks_cat_Evt.append(second_ID)
                            ## Append the SV index to the tracks true SV index list 
                            ## It will be used after the loop over tracks
                            true_SV_idx_Evt.append(sv_idx[0][0])

                        else:
                            ## Append the PV coordinates to the tracks OV coordinates list
                            tracks_OV_x_Evt.append(pv_x_Evt[pv_idx][0])
                            tracks_OV_y_Evt.append(pv_y_Evt[pv_idx][0])
                            tracks_OV_z_Evt.append(z_pv_asso)
                            ## Append the tracks category 'prompt_ID' value to the tracks category list
                            tracks_cat_Evt.append(prompt_ID)
                    else:
                        print("WARNING :: No SV association!")

            else:
                tracks_OV_x_Evt.append(-999.)
                tracks_OV_y_Evt.append(-999.)
                tracks_OV_z_Evt.append(-999.)
                tracks_cat_Evt.append(none_ID)

        ## ====================================================================================================
        ## Let's append the current event tracks true origin vertex coordinates and there category
        tracks_OV_x.append(tracks_OV_x_Evt)
        tracks_OV_y.append(tracks_OV_y_Evt)
        tracks_OV_z.append(tracks_OV_z_Evt)
        tracks_cat.append(tracks_cat_Evt)

        ## ====================================================================================================
        ## We now want to construct the SV info (x,y,z,n) based on the tracks that originates from true SVs.
        ## Given that it happens sometime that multiple true "SV" actually have the same x,y,z values we want to 
        ## only store one SV and count the total number of tracks as the sum of tracks originating from
        ## these "identical" true SV.

        ## Start by checking that there actually are some tracks that originates from SV:
        if true_SV_idx_Evt:

            ## Modify the true SV index python list into a np array
            true_SV_idx_Evt = np.array(true_SV_idx_Evt)

            ## Get the coordinates of the true SV index np array
            true_SV_x_Evt = sv_x_Evt[true_SV_idx_Evt]
            true_SV_y_Evt = sv_y_Evt[true_SV_idx_Evt]
            true_SV_z_Evt = sv_z_Evt[true_SV_idx_Evt]

            ## Let's sort them by ascending z values
            true_SV_z_Evt_sortIdx = np.argsort(true_SV_z_Evt)
            true_SV_x_Evt_sorted = true_SV_x_Evt[true_SV_z_Evt_sortIdx]
            true_SV_y_Evt_sorted = true_SV_y_Evt[true_SV_z_Evt_sortIdx]
            true_SV_z_Evt_sorted = true_SV_z_Evt[true_SV_z_Evt_sortIdx]

            ## Now, we can check how many tracks originates from the same SV based on there coordinates
            unique_true_SV_z_Evt_sorted, unique_true_SV_nTracks_Evt_sorted = np.unique(true_SV_z_Evt_sorted, return_counts=True)
            unique_true_SV_x_Evt_sorted, _ = np.unique(true_SV_x_Evt_sorted, return_counts=True)
            unique_true_SV_y_Evt_sorted, _ = np.unique(true_SV_y_Evt_sorted, return_counts=True)

            ## Because we assume that SV for a given z coordinate would have the same x and y values, 
            ## we can check that the number of unique x, y, and z values are idential
            len_unique_true_SV_x_Evt_sorted = len(unique_true_SV_x_Evt_sorted)
            len_unique_true_SV_y_Evt_sorted = len(unique_true_SV_y_Evt_sorted)
            len_unique_true_SV_z_Evt_sorted = len(unique_true_SV_z_Evt_sorted)

            ## If not, let's print out some warning. It happens on rare occasions, and should not be an issue,
            ## given that we store one unique SV per z values with the corresponding x and y coordinates.
            
            if len_unique_true_SV_z_Evt_sorted!=len_unique_true_SV_x_Evt_sorted or len_unique_true_SV_z_Evt_sorted!=len_unique_true_SV_y_Evt_sorted:
                print("*"*50)
                print("WARNING::")
                print("  At Event: ",iEvt)
                print("  Different number of SV with unique Z and {X or Y} values !!")
                print("   - N(SV) with unique X values:",len_unique_true_SV_x_Evt_sorted)
                print("   - N(SV) with unique Y values:",len_unique_true_SV_y_Evt_sorted)
                print("   - N(SV) with unique Z values:",len_unique_true_SV_z_Evt_sorted)
                print("")
                print("  `--> Will account for %s SV assuming they need to have different Z values"%(len_unique_true_SV_z_Evt_sorted))
                print("")
                print("*"*50)
                #print("true_SV_x_Evt_sorted",true_SV_x_Evt_sorted)
                #print("true_SV_y_Evt_sorted",true_SV_y_Evt_sorted)
                #print("true_SV_z_Evt_sorted",true_SV_z_Evt_sorted)

            ## Since the output of the np.unique method automatically sort the values
            ## we still need to loop over the unique true z values and get back the 
            ## corresponding x and y values.
            ## Let's start by setting the containers for the x and y values
            true_SV_x_vals   = np.zeros(len_unique_true_SV_z_Evt_sorted)
            true_SV_y_vals   = np.zeros(len_unique_true_SV_z_Evt_sorted)
            ## While the z and nTracks are already the output of the np.unique for the z values
            true_SV_z_vals   = unique_true_SV_z_Evt_sorted
            true_SV_n_tracks = unique_true_SV_nTracks_Evt_sorted

            ## Loop over the unique SV z values
            for iSV, SV_z_val in enumerate(unique_true_SV_z_Evt_sorted):
                ## Get the indices in the non-unique z coordinates that match with the
                ## current z value. The size of filt is equal to the number of tracks
                ## originating from the unique SV
                filt = np.where(true_SV_z_Evt_sorted==SV_z_val)
                ## Because we assume that the x and y for identical z values should be identical,
                ## Let's take the first entry in the list of x and y and assign them as the unique SV
                ## x and y coordinates
                true_SV_x_vals[iSV]   = true_SV_x_Evt_sorted[filt][0]
                true_SV_y_vals[iSV]   = true_SV_y_Evt_sorted[filt][0]

            ## We can now append the final unique SV coordinates and associated number of tracks
            real_SV_x.append(true_SV_x_vals)
            real_SV_y.append(true_SV_y_vals)
            real_SV_z.append(true_SV_z_vals)
            real_SV_n.append(true_SV_n_tracks)
        else:
            ## In case no track is associated to a SV (i.e. an event with all tracks being prompt),
            ## we append an empty list to the SV coordinates.
            real_SV_x.append([])
            real_SV_y.append([])
            real_SV_z.append([])
            real_SV_n.append([])

    ## We can finally create the ak arrays from the nested python lists
    ## For the unique SV info:
    real_SV_x_ak = ak.Array(real_SV_x)
    real_SV_y_ak = ak.Array(real_SV_y)
    real_SV_z_ak = ak.Array(real_SV_z)
    real_SV_n_ak = ak.Array(real_SV_n)

    ## For the tracks info:
    tracks_OV_x_ak = ak.Array(tracks_OV_x)
    tracks_OV_y_ak = ak.Array(tracks_OV_y)
    tracks_OV_z_ak = ak.Array(tracks_OV_z)
    tracks_cat_ak  = ak.Array(tracks_cat)
    
    
    if GNN_preprocess:
        output_tuple = InputData_tuple_SV_OV_updated_GNN(

            pv_x,
            pv_y, 
            pv_z,
            getattr(input_tuple, "pv_ntracks"),
            getattr(input_tuple, "pv_cat"),
            pv_key, 

            real_SV_x_ak,
            real_SV_y_ak,
            real_SV_z_ak,
            real_SV_n_ak,

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
            tracks_cat_ak,
            tracks_OV_x_ak,
            tracks_OV_y_ak,
            tracks_OV_z_ak,

            getattr(input_tuple, "recon_tx"), 
            getattr(input_tuple, "recon_ty"), 
        )        
    else:
        output_tuple = InputData_tuple_SV_OV_updated(

            pv_x,
            pv_y, 
            pv_z,
            getattr(input_tuple, "pv_ntracks"),
            getattr(input_tuple, "pv_cat"),
            pv_key, 

            real_SV_x_ak,
            real_SV_y_ak,
            real_SV_z_ak,
            real_SV_n_ak,

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
            tracks_cat_ak,
            tracks_OV_x_ak,
            tracks_OV_y_ak,
            tracks_OV_z_ak,

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
            
        
    return output_tuple
    
def Sort_PVs_by_Z_and_update_PV_key(input_tuple, GNN_preprocess=False, nTracks_ValidPVs = 0):

    ## ================================================================================
    ## ================================================================================
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
            
        
    ## ================================================================================
    ## ================================================================================
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
                       
        
    ## ================================================================================
    ## ================================================================================
    ## Since only PV information changed, create a new namedtuple to be returned
    ## with all other information from the input_tuple
    if GNN_preprocess:
        output_tuple = InputData_tuple_SV_OV_updated_GNN(

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
            getattr(input_tuple, "recon_cat"), 
            getattr(input_tuple, "recon_ov_x"), 
            getattr(input_tuple, "recon_ov_y"), 
            getattr(input_tuple, "recon_ov_z"), 

            getattr(input_tuple, "recon_tx"), 
            getattr(input_tuple, "recon_ty"), 
        )        
    else:
        output_tuple = InputData_tuple_SV_OV_updated(

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
        ) 
            
        
    return output_tuple, d_pv_k


def Sort_tracks_by_Z_and_update_T2PV_key(input_tuple, d_pv_key, GNN_preprocess=False):

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
    recon_cat     = getattr(input_tuple, "recon_cat")
    recon_ov_x    = getattr(input_tuple, "recon_ov_x")
    recon_ov_y    = getattr(input_tuple, "recon_ov_y")
    recon_ov_z    = getattr(input_tuple, "recon_ov_z")
    
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
    recon_cat_sorted     = recon_cat[tracks_index_by_z]
    recon_ov_x_sorted    = recon_ov_x[tracks_index_by_z]
    recon_ov_y_sorted    = recon_ov_y[tracks_index_by_z]
    recon_ov_z_sorted    = recon_ov_z[tracks_index_by_z]
    ## Let us use a temporary ak array for the tracks to PV keys 
    t2pv_key_sorted_tmp  = t2pv_key[tracks_index_by_z]

    ## We now want to modify the tracks to PV keys: 
    ## To do that we need to convert the ak array into a np array as 
    ## it is not possible to change ak array values.
    ## Let start by flattening the ak array and saving the shape of 
    ## the original ak array to allow unflattening it at the end of the process.
    t2pv_key_sorted_flattened, t2pv_key_shape = ak.flatten(t2pv_key_sorted_tmp), ak.num(t2pv_key_sorted_tmp)
    ## Create an empty np array of same length as the flattened ak array.
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
    if GNN_preprocess:
        return InputData_tuple_SV_OV_updated_GNN(

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
            recon_cat_sorted, 
            recon_ov_x_sorted, 
            recon_ov_y_sorted, 
            recon_ov_z_sorted, 

            recon_tx_sorted, 
            recon_ty_sorted, 
        ) 
    else:
        return InputData_tuple_SV_OV_updated(

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
            recon_cat_sorted, 
            recon_ov_x_sorted, 
            recon_ov_y_sorted, 
            recon_ov_z_sorted, 

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


def Filter_valid_tracks(input_tuple, config, GNN_preprocess=False):

    
    ## ************************************************************    
    ## Get the PVs and SVs vertexes information from the input tuple
    pv_x       = getattr(input_tuple, "pv_x")
    pv_y       = getattr(input_tuple, "pv_y")
    pv_z       = getattr(input_tuple, "pv_z")
    pv_ntracks = getattr(input_tuple, "pv_ntracks")
    pv_cat     = getattr(input_tuple, "pv_cat")
    pv_key     = getattr(input_tuple, "pv_key")

    sv_x       = getattr(input_tuple, "sv_x")
    sv_y       = getattr(input_tuple, "sv_y")
    sv_z       = getattr(input_tuple, "sv_z")
    sv_ntracks = getattr(input_tuple, "sv_ntracks")
    sv_cat     = getattr(input_tuple, "sv_cat")
    sv_key     = getattr(input_tuple, "sv_key")
    sv_pv_key  = getattr(input_tuple, "sv_pv_key")
    
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
    t2ov_key      = getattr(input_tuple, "recon_ov_key")
    
    if not GNN_preprocess:
        ## Get the KDE information from the input tuple
        IP_KDE          = getattr(input_tuple, "IP_KDE")
        IP_KDE_xMax     = getattr(input_tuple, "IP_KDE_xMax")
        IP_KDE_yMax     = getattr(input_tuple, "IP_KDE_yMax")

        poca_KDE_A      = getattr(input_tuple, "poca_KDE_A")
        poca_KDE_A_xMax = getattr(input_tuple, "poca_KDE_A_xMax")
        poca_KDE_A_yMax = getattr(input_tuple, "poca_KDE_A_yMax")

        poca_KDE_B      = getattr(input_tuple, "poca_KDE_B")
        poca_KDE_B_xMax = getattr(input_tuple, "poca_KDE_B_xMax")
        poca_KDE_B_yMax = getattr(input_tuple, "poca_KDE_B_yMax")

    ## ************************************************************        
    ## Get the total number of events
    nEvts = len(pv_x)
    
    ## ************************************************************    
    ## Retrieve filtering parameters from config file:
    MaxPOCA = config['MaxPOCA'] 
    MinPOCA = config['MinPOCA'] 
    
    ## First let's filter out tracks with too large POCA z values
    # Define the filter as tracks to accept...
    print("")
    print("Filtering tracks with: POCA_major_axis_z >",MaxPOCA)
    filter_Max = major_axis_z<MaxPOCA

    ## Second let's filter out tracks with too small minor1 POCA axis values
    # Define the filter as tracks to accept...
    print("Filtering tracks with: sqrt(POCA_minor_axis1_x^2 + POCA_minor_axis1_y^2 + POCA_minor_axis1_z^2)<",MinPOCA)
    filter_Min1 = np.sqrt(minor_axis1_x*minor_axis1_x + minor_axis1_y*minor_axis1_y + minor_axis1_z*minor_axis1_z)>MinPOCA

    ## Third let's filter out tracks with too small minor2 POCA axis values
    # Define the filter as tracks to accept...
    print("Filtering tracks with: sqrt(POCA_minor_axis2_x^2 + POCA_minor_axis2_y^2 + POCA_minor_axis2_z^2)<",MinPOCA)
    filter_Min2 = np.sqrt(minor_axis2_x*minor_axis2_x + minor_axis2_y*minor_axis2_y + minor_axis2_z*minor_axis2_z)>MinPOCA

    ## Combine all filters of tracks to be accepted // reject tracks when filt_comb==False
    filter_tracks = filter_Max*filter_Min1*filter_Min2
        
    ## Store in a dict with tracks from which events are being filtered out:
    filter_events = [True] * nEvts
    d_tracks_filt = {}
    # For this let's just loop over events (probably there is a better way of doing this...) 
    for iEvt in range(len(poca_z)):
        # ak.where(filter_tracks[iEvt]) returns a tuple with one array 
        # corresponding to the list of tracks filtered in each event (if any)
        # So using "[0]" returns the corresponding array
        iFilt = ak.where(filter_tracks[iEvt]==False)[0]
        if not len(iFilt)==0:
            filter_events[iEvt] = False
            d_tracks_filt[iEvt] = iFilt
            
    # Now display which tracks are being filtered out 
    if len(d_tracks_filt)==0:
        print("")
        print("")
        print("No tracks matching filtering conditions found. All tracks to be used.")        
    else:        
        print("")
        print("")
        print("A total of %s events (out of %s) are now going to be filtered out."%(len(d_tracks_filt),nEvts))
        print("")
        print("In details...")
        print("")
        for evtID in d_tracks_filt.keys():
            # ak.where(filter_tracks[iEvt]) returns a tuple with one array 
            # corresponding to the list of tracks filtered in each event (if any)
            # So using "[0]" returns the corresponding array
            print("  ------------")
            print("  Event %s"%(evtID))
            for iTrack, trackID in enumerate(d_tracks_filt[evtID]):
                print("")
                print("    containing badly behaving track (%s) with values:"%trackID)
                print("        POCA_x  = %.15f ; POCA_y  = %.15f ; POCA_z  = %.15f"%(poca_x[evtID][trackID],poca_y[evtID][trackID],poca_z[evtID][trackID]))
                print("        POCA_major_axis_x  = %.3f ; POCA_major_axis_y  = %.3f ; POCA_major_axis_z  = %.3f"%(major_axis_x[evtID][trackID],major_axis_y[evtID][trackID],major_axis_z[evtID][trackID]))
                print("        POCA_minor_axis1_x = %.15f ; POCA_minor_axis1_y = %.15f ; POCA_minor_axis1_z = %.15f"%(minor_axis1_x[evtID][trackID],minor_axis1_y[evtID][trackID],minor_axis1_z[evtID][trackID]))
                print("        POCA_minor_axis2_x = %.15f ; POCA_minor_axis2_y = %.15f ; POCA_minor_axis2_z = %.15f"%(minor_axis2_x[evtID][trackID],minor_axis2_y[evtID][trackID],minor_axis2_z[evtID][trackID]))

                
    ## ************************************************************    
    # ...and apply the event filter    
    # -----------------------------------
    pv_x_filt       = pv_x[filter_events]
    pv_y_filt       = pv_y[filter_events]
    pv_z_filt       = pv_z[filter_events]
    pv_ntracks_filt = pv_ntracks[filter_events]
    pv_cat_filt     = pv_cat[filter_events]
    pv_key_filt     = pv_key[filter_events]

    sv_x_filt       = sv_x[filter_events]
    sv_y_filt       = sv_y[filter_events]
    sv_z_filt       = sv_z[filter_events]
    sv_ntracks_filt = sv_ntracks[filter_events]
    sv_cat_filt     = sv_cat[filter_events]
    sv_key_filt     = sv_key[filter_events]
    sv_pv_key_filt  = sv_pv_key[filter_events]
    
    # -----------------------------------
    poca_x_filt        = poca_x[filter_events]
    poca_y_filt        = poca_y[filter_events]
    poca_z_filt        = poca_z[filter_events]
    major_axis_x_filt  = major_axis_x[filter_events]
    major_axis_y_filt  = major_axis_y[filter_events]
    major_axis_z_filt  = major_axis_z[filter_events]
    minor_axis1_x_filt = minor_axis1_x[filter_events]
    minor_axis1_y_filt = minor_axis1_y[filter_events]
    minor_axis1_z_filt = minor_axis1_z[filter_events]
    minor_axis2_x_filt = minor_axis2_x[filter_events]
    minor_axis2_y_filt = minor_axis2_y[filter_events]
    minor_axis2_z_filt = minor_axis2_z[filter_events]
    recon_tx_filt      = recon_tx[filter_events]
    recon_ty_filt      = recon_ty[filter_events]
    t2pv_key_filt      = t2pv_key[filter_events]
    t2ov_key_filt      = t2ov_key[filter_events]
    
    
    if not GNN_preprocess:
        # -----------------------------------
        IP_KDE_filt          = IP_KDE[filter_events]
        IP_KDE_xMax_filt     = IP_KDE_xMax[filter_events]
        IP_KDE_yMax_filt     = IP_KDE_yMax[filter_events]

        poca_KDE_A_filt      = poca_KDE_A[filter_events]
        poca_KDE_A_xMax_filt = poca_KDE_A_xMax[filter_events]
        poca_KDE_A_yMax_filt = poca_KDE_A_yMax[filter_events]

        poca_KDE_B_filt      = poca_KDE_B[filter_events]
        poca_KDE_B_xMax_filt = poca_KDE_B_xMax[filter_events]
        poca_KDE_B_yMax_filt = poca_KDE_B_yMax[filter_events]

    
    ## Create a new namedtuple to be returned
    if GNN_preprocess:
        return InputData_tuple_GNN(

            # -----------------------------------
            pv_x_filt,
            pv_y_filt,
            pv_z_filt,
            pv_ntracks_filt,
            pv_cat_filt,
            pv_key_filt,

            sv_x_filt,
            sv_y_filt,
            sv_z_filt,
            sv_ntracks_filt,
            sv_cat_filt,
            sv_key_filt,
            sv_pv_key_filt,

            # -----------------------------------
            poca_x_filt,
            poca_y_filt,
            poca_z_filt,

            major_axis_x_filt,
            major_axis_y_filt,
            major_axis_z_filt,
            minor_axis1_x_filt,
            minor_axis1_y_filt,
            minor_axis1_z_filt,
            minor_axis2_x_filt,
            minor_axis2_y_filt,
            minor_axis2_z_filt,

            t2pv_key_filt,
            t2ov_key_filt,

            recon_tx_filt,
            recon_ty_filt
        )         
    else:
        return InputData_tuple(

            # -----------------------------------
            pv_x_filt,
            pv_y_filt,
            pv_z_filt,
            pv_ntracks_filt,
            pv_cat_filt,
            pv_key_filt,

            sv_x_filt,
            sv_y_filt,
            sv_z_filt,
            sv_ntracks_filt,
            sv_cat_filt,
            sv_key_filt,
            sv_pv_key_filt,

            # -----------------------------------
            poca_x_filt,
            poca_y_filt,
            poca_z_filt,

            major_axis_x_filt,
            major_axis_y_filt,
            major_axis_z_filt,
            minor_axis1_x_filt,
            minor_axis1_y_filt,
            minor_axis1_z_filt,
            minor_axis2_x_filt,
            minor_axis2_y_filt,
            minor_axis2_z_filt,

            t2pv_key_filt,
            t2ov_key_filt,

            recon_tx_filt,
            recon_ty_filt,

            # -----------------------------------
            IP_KDE_filt,
            IP_KDE_xMax_filt,
            IP_KDE_yMax_filt,

            poca_KDE_A_filt,
            poca_KDE_A_xMax_filt,
            poca_KDE_A_yMax_filt,

            poca_KDE_B_filt,
            poca_KDE_B_xMax_filt,
            poca_KDE_B_yMax_filt
        ) 
    