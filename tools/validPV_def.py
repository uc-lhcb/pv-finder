'''
Method to update the definition of valid PVs. Default from the input ROOT file is 

valid_PV == PV(nTracks>5)

'''

import awkward as ak
import numpy as np
from tools.named_tuples_def import InputData_tuple_SV_OV_updated, InputData_tuple_SV_OV_updated_GNN
from utils.utilities import Timer

def Update_validPV_def(input_tuple, n_tracks_valid_PVs, z_min, z_max, GNN_preprocess=False):

    ## ------------------------------------------------------------
    ## Retrieve the ntracks information as well as the z position of the PV 
    pv_z        = getattr(input_tuple, "pv_z")
    pv_ntracks  = getattr(input_tuple, "pv_ntracks")
    
    ## ------------------------------------------------------------
    ## Construct an updated category ak array 
    ## with value 0 if pv_ntracks<n_tracks_valid_PVs
    ## and value 1 otherwise (i.e. for valid PVs)
    
    ## NOTE::This will have as effect to remove all PVs category -1 which correspond to PVs with no LHCb long tracks.
    ## Looking into the input file, this seem to occur only for PVs with 1 associated track!!
    pv_cat_updated = ak.where((pv_ntracks<n_tracks_valid_PVs)|(pv_z<z_min)|(pv_z>z_max), 0, 1)

    ## ------------------------------------------------------------
    ## Return the updated tuple
    if GNN_preprocess:
        return InputData_tuple_SV_OV_updated_GNN(

            getattr(input_tuple, "pv_x"),
            getattr(input_tuple, "pv_y"),
            getattr(input_tuple, "pv_z"),
            getattr(input_tuple, "pv_ntracks"),

            pv_cat_updated,

            getattr(input_tuple, "pv_key"), 

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
        return InputData_tuple_SV_OV_updated(

            getattr(input_tuple, "pv_x"),
            getattr(input_tuple, "pv_y"),
            getattr(input_tuple, "pv_z"),
            getattr(input_tuple, "pv_ntracks"),

            pv_cat_updated,

            getattr(input_tuple, "pv_key"), 

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
    