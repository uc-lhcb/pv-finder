'''
This is meant to be the main method where all the necessary input data are extracted from the original root files 
and where the computation of all the necessary additional information is done.
The output is in the format of a namedtuple containing all the variables. 
This output namedtuple will be concatenated in the script 'scripts.Make_HDF5_files_from_ROOT_files.py' where
'process_root_file' is called before the 'raw' input variables are transformed in higher level variables: 
- KDEs 
- target histograms
- six ellipsoid parameters
- ...
'''

import uproot as uproot
import awkward as ak
import numpy as np
from utils.utilities import Timer, Set_ak_values_to_zero_condition
from tools.named_tuples_def import InputData_tuple, InputData_tuple_GNN

def Process_root_file(filepath, configs, GNN_preprocess=False):

    name = filepath.stem
    with Timer(start=f"Loading file: {name}.root"):
        
        tree = uproot.open(str(filepath))["kernel"]

        ## --------------------------------------------------------------------------------------------------------
        ## All PV variables are of shape [N_PVs * N_evts]
        ## 
        ## Note that all the PV are accounted for, also those with 0 tracks. The reason is that when creating 
        ## the input root files, the MC tracks are 'filtered' using a reconstructability criteria:  
        ## only long tracks are considered (i.e. tracks with hits in all of LHCb's tracking sub-detectors)
        ## 
        ## These can be filtered using the variable 'pv_cat = {'-1': pv_ntrks < 2, '0': [2 <= pv_ntrks < 5], '1': pv_ntrks > 5}
        ## Howerever the definition of the number of long tracks forming a PV to be considered could be changed, 
        ## for example from 5 to 4, later on in the code...
        ###
        pv_x     = tree["pv_loc_x"].array() # True x PV 
        pv_y     = tree["pv_loc_y"].array() # True y PV 
        pv_z     = tree["pv_loc"].array()   # True z PV 
        pv_ntrks = tree["pv_ntrks"].array() # Number of tracks in PV
        pv_cat   = tree["pv_cat"].array()   # PV category (LHCb or not)
        pv_key   = tree["pv_key"].array()   # Unique ID identifying each PV 

        ## --------------------------------------------------------------------------------------------------------
        ## All SV variables are of shape [N_SVs * N_evts]
        ## 
        ## Note that all the SV are accounted for, also those from bremsstrahlung (i.e. with sv_ntrks = 1)
        ## These can be filtered using the variable 'sv_cat = {'-1': sv_ntrks = 0, '0': sv_ntrks = 1, '1': sv_ntrks >1}
        ##
        sv_x       = tree["sv_loc_x"].array() # SVs like above
        sv_y       = tree["sv_loc_y"].array()
        sv_z       = tree["sv_loc"].array()   
        sv_ntrks   = tree["sv_ntrks"].array()
        sv_cat     = tree["sv_cat"].array()
        sv_key     = tree["svr_key"].array()   
        sv_pv_key  = tree["svr_pv_key"].array() # PV key to which the SV is related to 

        ## --------------------------------------------------------------------------------------------------------
        ## All tracks variables are of shape [N_tracks * N_evts]
        ## 
        ## Note that N_tracks != N_PVs * pv_ntrks, as pv_ntrks is from MC truth information and N_tracks corresponds to 
        ## the number of reconstructed in LHCb reconstruction software (from hits in the detector)
        ## 
        poca_x        = tree["POCA_center_x"].array()       ## poca ellipsoid center, i.e. tracks x coordinate
        poca_y        = tree["POCA_center_y"].array()       ## poca ellipsoid center, i.e. tracks y coordinate
        poca_z        = tree["POCA_center_z"].array()       ## poca ellipsoid center, i.e. tracks z coordinate
        major_axis_x  = tree["POCA_major_axis_x"].array()   ## poca ellipsoid major axis vector x coordinate (@1 sigma)
        major_axis_y  = tree["POCA_major_axis_y"].array()   ## poca ellipsoid major axis vector y coordinate (@1 sigma)
        major_axis_z  = tree["POCA_major_axis_z"].array()   ## poca ellipsoid major axis vector z coordinate (@1 sigma)
        minor_axis1_x = tree["POCA_minor_axis1_x"].array()  ## poca ellipsoid minor axis 1 vector x coordinate (@1 sigma)
        minor_axis1_y = tree["POCA_minor_axis1_y"].array()  ## poca ellipsoid minor axis 1 vector y coordinate (@1 sigma)
        minor_axis1_z = tree["POCA_minor_axis1_z"].array()  ## poca ellipsoid minor axis 1 vector z coordinate (@1 sigma)
        minor_axis2_x = tree["POCA_minor_axis2_x"].array()  ## poca ellipsoid minor axis 2 vector x coordinate (@1 sigma)
        minor_axis2_y = tree["POCA_minor_axis2_y"].array()  ## poca ellipsoid minor axis 2 vector y coordinate (@1 sigma)
        minor_axis2_z = tree["POCA_minor_axis2_z"].array()  ## poca ellipsoid minor axis 2 vector z coordinate (@1 sigma)
        ##
        recon_pv_key  = tree["recon_pv_key"].array()        ## Unique ID identifying the PV from which the track originates 
        recon_ov_key  = tree["recon_ov_key"].array()        ## Unique ID identifying the Origin vertex (PV or SV) from which the track originates 
        ##
        recon_tx  = tree["recon_tx"].array()                ## Tracks slope projected on X-axis (for poca_KDE computation)
        recon_ty  = tree["recon_ty"].array()                ## Tracks slope projected on Y-axis (for poca_KDE computation)
        
        if not GNN_preprocess:
            ## --------------------------------------------------------------------------------------------------------
            ## All KDE variables are of shape [4000 * N_evts]
            ## 
            ## While no method is currently implemented to recompute the old KDE (from tracks IP) as using brut force minuit 
            ## in the cpp code generating the ROOT files, a method is available to recompute the poca_KDE variables taking 
            ## the tracks poca info. This is useful to create poca_KDE variables with a different binning scheme 
            ## -------------
            #IP_KDE      = tree["oldzdata"].array() / 2500.0 # Old KDE in Z := [(Sum tracks_IP^2 - Sum tracks_IP) / Sum tracks_IP]
            IP_KDE      = tree["oldzdata"].array()          # Old KDE in Z := [(Sum tracks_IP^2 - Sum tracks_IP) / Sum tracks_IP]
            # Rescale IP_KDE 
            f_IP_KDE    = configs["scaleFactor_IP_KDE"]
            IP_KDE      = IP_KDE / f_IP_KDE
            IP_KDE_xMax = tree["oldxmax"].array()           # Location in X of max(IP_KDE)  
            IP_KDE_yMax = tree["oldymax"].array()           # Location in X of max(IP_KDE)  
            ## 
            ## Set the IP_KDE_xMax and IP_KDE_yMax histograms to 0 where IP_KDE is also 0
            IP_KDE_xMax = Set_ak_values_to_zero_condition(IP_KDE,IP_KDE_xMax)
            IP_KDE_yMax = Set_ak_values_to_zero_condition(IP_KDE,IP_KDE_yMax)        
            ## -------------
            #poca_KDE_A      = tree["POCAzdata"].array() / 1000.0     # New KDE in Z := Sum tracks_POCA 
            poca_KDE_A      = tree["POCAzdata"].array()              # New KDE in Z := Sum tracks_POCA 
            # Rescale poca_KDE_A 
            f_poca_KDE_A    = configs["scaleFactor_poca_KDE_A"]       # Get the scale factor from config file
            poca_KDE_A      = poca_KDE_A/f_poca_KDE_A                 # Recaling of poca_KDE_A
            ##         
            poca_KDE_A_xMax = tree["POCAxmax"].array()                # Location in X of max(poca_KDE_A)  
            poca_KDE_A_yMax = tree["POCAymax"].array()                # Location in Y of max(poca_KDE_A)
            ## -------------
            #poca_KDE_B      = tree["POCA_sqzdata"].array() / 10000.0  # New "KDE^2" in Z := Sum tracks_POCA^2 
            poca_KDE_B      = tree["POCA_sqzdata"].array()            # New "KDE^2" in Z := Sum tracks_POCA^2 
            # Rescale poca_KDE_B 
            f_poca_KDE_B    = configs["scaleFactor_poca_KDE_B"]       # Get the scale factor from config file
            poca_KDE_B      = poca_KDE_B / f_poca_KDE_B               # Recaling of poca_KDE_B
            ##         
            poca_KDE_B_xMax = tree["POCA_sqxmax"].array()             # Location in X of max(poca_KDE_B)  
            poca_KDE_B_yMax = tree["POCA_sqymax"].array()             # Location in Y of max(poca_KDE_B)  
            ## 
            ## Set the poca_KDE_(A/B)_xMax and poca_KDE_(A/B)_yMax histograms to 0 where poca_KDE_(A/B) is also 0
            ## 
            poca_KDE_A_xMax = Set_ak_values_to_zero_condition(poca_KDE_A,poca_KDE_A_xMax)
            poca_KDE_A_yMax = Set_ak_values_to_zero_condition(poca_KDE_A,poca_KDE_A_yMax)
            poca_KDE_B_xMax = Set_ak_values_to_zero_condition(poca_KDE_B,poca_KDE_B_xMax)
            poca_KDE_B_yMax = Set_ak_values_to_zero_condition(poca_KDE_B,poca_KDE_B_yMax)

            ## Shift all KDE values by an epsilon value such that they are never equal to 0
            ## This is usefull from a ML approach since 0 values introduces divergences
            ## Get the epsilon value from the config file
            epsilon_KDE = configs["epsilon_KDE"]       
            ## Add epsilon to all KDE (z) variables
            IP_KDE     = IP_KDE + epsilon_KDE
            poca_KDE_A = poca_KDE_A + epsilon_KDE
            poca_KDE_B = poca_KDE_B + epsilon_KDE
        
        ## --------------------------------------------------------------------------------------------------------
        ## Change some of the input arrays from the default types (float32) to suited types (e.g. int16,...) 
        ## 
        pv_ntrks     = ak.values_astype(pv_ntrks,     "uint16")
        sv_ntrks     = ak.values_astype(sv_ntrks,     "uint16")
        ##
        pv_key       = ak.values_astype(pv_key,       "int16")
        sv_key       = ak.values_astype(sv_key,       "int16")
        sv_pv_key    = ak.values_astype(sv_pv_key,    "int16")
        ##
        recon_pv_key = ak.values_astype(recon_pv_key, "int16")
        recon_ov_key = ak.values_astype(recon_ov_key, "int16")
        ##
        pv_cat       = ak.values_astype(pv_cat,       "int8")
        sv_cat       = ak.values_astype(sv_cat,       "int8")


    if not GNN_preprocess:
        ## Return the input arrays as 
        return InputData_tuple(

            pv_x,
            pv_y,
            pv_z,
            pv_ntrks,
            pv_cat,
            pv_key, 

            sv_x,
            sv_y,
            sv_z,
            sv_ntrks,
            sv_cat,
            sv_key, 
            sv_pv_key, 

            poca_x,               
            poca_y,              
            poca_z,              
            major_axis_x,        
            major_axis_y,        
            major_axis_z,        
            minor_axis1_x,       
            minor_axis1_y,       
            minor_axis1_z,       
            minor_axis2_x,       
            minor_axis2_y,       
            minor_axis2_z,       

            recon_pv_key, 
            recon_ov_key, 

            recon_tx, 
            recon_ty, 

            IP_KDE,
            IP_KDE_xMax,
            IP_KDE_yMax,

            poca_KDE_A,
            poca_KDE_A_xMax,
            poca_KDE_A_yMax,
            poca_KDE_B,
            poca_KDE_B_xMax,
            poca_KDE_B_yMax,

        ) 
    else:
        ## Return the input arrays as 
        return InputData_tuple_GNN(

            pv_x,
            pv_y,
            pv_z,
            pv_ntrks,
            pv_cat,
            pv_key, 

            sv_x,
            sv_y,
            sv_z,
            sv_ntrks,
            sv_cat,
            sv_key, 
            sv_pv_key, 

            poca_x,               
            poca_y,              
            poca_z,              
            major_axis_x,        
            major_axis_y,        
            major_axis_z,        
            minor_axis1_x,       
            minor_axis1_y,       
            minor_axis1_z,       
            minor_axis2_x,       
            minor_axis2_y,       
            minor_axis2_z,       

            recon_pv_key, 
            recon_ov_key, 

            recon_tx, 
            recon_ty

        ) 
        