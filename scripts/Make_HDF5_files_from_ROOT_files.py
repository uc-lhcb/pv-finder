#!/usr/bin/env python3
'''
Main script to transform input ROOT files into HDF5 files, meant to be as generic as possible, allowing to save all necessary information. 

This script calls external methods saving in HDF5 format some of the following quantities present in the orignial ROOT files:

One needs to export the python path of the package:

$ export PYTHONPATH='/home/sakar/pv-finder_v2/'

======================================================================================================
## True PVs infos       SHAPE ([N_pvs] * N_evts)
======================================================================================================
# PV positions: 
  "pv_loc_x" in x
  "pv_loc_y" in y
  "pv_loc"   in z

# "pv_ntracks": 
        number of long LHCb tracks originating from the PVs: N_tracks ; 
        Definition of long LHCb tracks is based on a crieria of "reconstructability" of the tracks, 
        i.e. number of hits in the different tracking sub-detectors of LHCb

# "pv_cat":      
        category of the PVs based on the number of tracks: 
         >> "-1": zero long LHCb track
         >> "0" : N_tracks < 5
         >> "1" : N_tracks >= 5
        
# "pv_key":
        unique identifier (i.e. different for each PV) as a key from MC
    
    
======================================================================================================
## True SVs infos: 
======================================================================================================
# !!! CURRENTLY NOT PROPAGATED ---> NEED TO BE IMPLEMENTED !!!


======================================================================================================
## KDE like quantities      SHAPE (N_KDE_bins * N_evts) , where N_KDE_bins == 4000 by default
======================================================================================================

# -----------------------
# Latest version of KDEs 
# -----------------------
# Based on the tracks Point Of Closest Approach (POCA), 
# instead of the tracks Impact Parameter (IP)

- "poca_KDE_A": 
        reconstructed KDE (KDE in z from the SUM over tracks POCA params) 
- "poca_KDE_A_xMax":
        maximum of X for which poca_KDE_A is maximum in Z
- "poca_KDE_A_yMax":
        maximum of Y for which poca_KDE_A is maximum in Z

- "poca_KDE_B": 
        reconstructed KDE "squared" (KDE in z from the SUM over !!squared!! tracks POCA params) 
- "poca_KDE_B_xMax":
        maximum of X for which poca_KDE_B (in z) is maximum
- "poca_KDE_B_yMax":
        maximum of Y for which poca_KDE_B (in z) is maximum

# -----------------------
# Old version of KDEs 
# -----------------------
# Based on the tracks IP, 
# somewhat deprecated and replaced by KDE from POCA, see above

- "X": 
        reconstructed KDE_IP (KDE in z from the tracks IP params) 
- "Y": 
        true KDE_IP  
- "Xmax": 
        maximum of X for which KDE_IP is maximum in Z
- "Ymax": 
        maximum of Y for which KDE_IP is maximum in Z


======================================================================================================
## Target histograms    SHAPE (N_KDE_bins * N_evts)
======================================================================================================
# The target histograms used in the training of the PV-finder models (KDE-to-hist or tracks-to-hist)
are generated from an empirical function which relates the width of the PVs peaks to the number of 
tracks that originates from them. The target histograms are generated from the method Compute_target_poca_KDE
and are independant from the input KDE (KDE_POCA or KDE_IP). They are generated using the same number 
of bins as for the KDE histograms.

- "target_hist"

======================================================================================================
## Tracks like quantities    SHAPE ([N_tracks] * N_evts)
======================================================================================================

# -----------------------
# Latest tracks parameters -- 12 parameters ; 3 for center and 9 for uncertainty --
# -----------------------
# Based on the POCA for each track

# Center of POCA ellipsoid 
- "POCA_center_x"
- "POCA_center_y"
- "POCA_center_z"

# Parameters related to "size" of the ellipsoid describing the uncertainty on the tracks POCA center and obtained from reconstructed tracks' covariance matrix 
- "POCA_major_axis_x"
- "POCA_major_axis_y"
- "POCA_major_axis_z"

- "POCA_minor_axis1_x"
- "POCA_minor_axis1_y"
- "POCA_minor_axis1_z"

- "POCA_minor_axis2_x"
- "POCA_minor_axis2_y"
- "POCA_minor_axis2_z"

# The 9 POCA_axis parameters can be reduced into a set of six parameters, which are used in the training of tracks to KDE or tracks to hist models. These tracks ellipsoid parameters are computed from the 9 POCA_axis parameters from the method Compute_tracks_ellipsoid:
- "poca_A"
- "poca_B"
- "poca_C"
- "poca_D"
- "poca_E"
- "poca_F"

# -----------------------
# Old tracks parameters -- 5 parameters -- (could still be useful) 
# -----------------------

# Point in space where IP is minimal (point where track is closest to beam line): 
# !!!! >>>>>>>> Present in the ROOT file, but NOT PERSISTED in the HDF5 file for the moment !!!
- "recon_x" 
- "recon_y" 
- "recon_z"

# Slope of the tracks:
!!!! >>>>>>>> Present in the ROOT file, AND ARE PERSISTED in the HDF5 file as these info comes handy when needing to recompute the POCA KDE !!!
- "recon_tx"
- "recon_ty"

'''


import argparse
from pathlib import Path, PosixPath
#import numpy as np
import warnings
import yaml
import glob
import time
import os

from utils.utilities            import Timer, Save_tuple_to_hdf5, getDataTagName
from tools.named_tuples_def     import *
from tools.concatenate          import Concatenate_tuples
from tools.process_root_files   import Process_root_file
from tools.sorting_filtering    import Filter_valid_tracks, Find_TrueSVs_and_TracksOriginVertex
from tools.sorting_filtering    import Sort_PVs_by_Z_and_update_PV_key, Sort_tracks_by_Z_and_update_T2PV_key
from tools.POCA_KDE             import Compute_updated_poca_KDE
from tools.target_hists         import Compute_target_hists
from tools.ellipsoid            import Compute_tracks_ellipsoid
from tools.validPV_def          import Update_validPV_def
from tools.split_data_intervals import split_data_intervals

default_n_bins_poca_kde = {"LHCb":4000,"ATLAS":8000}
default_n_tracks_valid_PVs = {"LHCb":5,"ATLAS":2}

def main(output_fname, files, Exp, configs):

        
    ## Measure full time to process data:
    global_start_time = time.time()

    
    ## Get the global configuration (one for each experiment, selected based on the --experiment argument)    
    Exp = configs['Experiment']
    global_configs = configs['global_configs'][Exp]
    
    ## Get the data_processing configuration
    data_processing_configs = configs["data_processing_configs"]
    
    ## -------------------------------------------------------------------------------------------
    ## Get the number of bins for the KDE and target hist from configuration file 
    n_bins_poca_kde    = global_configs['n_bins_poca_kde']
    ## Get the number of originating tracks for valid PVs to default value (i.e. PV(nTracks>=5) in LHCb)
    n_tracks_valid_PVs = global_configs['n_tracks_valid_PVs']
    ## Get the flag to split or not the data into intervals
    doIntervals = global_configs['doIntervals']
    ## Get the valid Z interval:
    z_min = global_configs['z_min']
    z_max = global_configs['z_max']
    
    ## ------------------------------------------------------------
    ## Check that the list of input ROOT files do actually exist
    ## before processing them
    for f in files:
        assert f.exists(), f"{f} must be an existing file"

    ## ------------------------------------------------------------
    ## One line to get all the input ROOT files as a list of namedtuple 
    ## containers, each one conatining the awkward arrays of the variables 
    ## from the input ROOT files
    print("-"*100)
    print("Processing input ROOT files...")
    l_tuples = [Process_root_file(f, global_configs) for f in files if not "reduced" in str(f)]

    ## ------------------------------------------------------------
    ## Convert list of namedtuples to one namedtuple
    print("-"*100)
    with Timer(start="\n Concatenating... \n"):
        cTuple = Concatenate_tuples("InputData_tuple", l_tuples)

    ## -------------------------------------------------------------------------------------------
    ## Below are all the data transformation methods 
    ## -------------------------------------------------------------------------------------------

    ## ------------------------------------------------------------
    ## Start by filtering tracks with badly behaving POCA values
    print("-"*100)
    with Timer(start="\n Filtering events where at least one track has badly behaving POCA values... \n"):
        cTuple = Filter_valid_tracks(cTuple, global_configs)

    ## ------------------------------------------------------------
    ## Find real SVs and get the origin vertex coordinates of tracks
    print("-"*100)
    with Timer(start="\n Finding real SVs storing the tracks Origin Vertex coordinates and category... \n"):
        cTuple = Find_TrueSVs_and_TracksOriginVertex(cTuple)

    ## ------------------------------------------------------------
    ## Sort PVs by Z and update the keys (from 'random numbers' to ordered 0,1,2,3...)
    ## Updated the PV keys might not be important, but it seems just easier to read.
    print("-"*100)
    with Timer(start="\n Sorting PVs by Z and updating PV keys... \n"):
        cTuple, dict_PV_keys = Sort_PVs_by_Z_and_update_PV_key(cTuple)

    ## ------------------------------------------------------------
    ## Sort tracks by Z and update the tracks to PV keys according to the update PV keys
    print("-"*100)
    with Timer(start="\n Sorting tracks by Z and updating T2PV keys... \n"):
        cTuple = Sort_tracks_by_Z_and_update_T2PV_key(cTuple, dict_PV_keys)

    ## ------------------------------------------------------------
    ## Compute updated poca_KDE (with updated binning scheme) 
    print("-"*100)
    if n_bins_poca_kde==default_n_bins_poca_kde[Exp]:
        print("\n ")
        print(" Requested computing POCA KDE with %s bins..."%(default_n_bins_poca_kde[Exp]))
        print(" This information is already available in the input tuples.")
        print(" It will be simply copied to the hdf5 output file.")
        print("\n ")
    else:
        print("\n Requested updating POCA KDE with N_bins = %s... \n"%(n_bins_poca_kde))
        with Timer(start="\n Computing poca_KDE with updated binning scheme: ... \n"):
            cTuple = Compute_updated_poca_KDE(cTuple, n_bins_poca_kde)

    ## ------------------------------------------------------------
    ## Compute updated PV category, i.e. the "pv_cat" variable (with updated minimal nTracks value) 
    print("-"*100)
    #if n_tracks_valid_PVs==default_n_tracks_valid_PVs[Exp]:
    #    print("\n Defining valid PVs with default values: PV(nTracks>=%s) \n"%default_n_tracks_valid_PVs[Exp])
    #else:
    #    print("\n Requested updating valid PVs definition from valid_PV == PV(nTracks>=%s) to valid_PV == PV(nTracks>=%s) \n"%(default_n_tracks_valid_PVs[Exp], n_tracks_valid_PVs))
    with Timer(start="\n Updating the valid PVs category now... \n"):
        print("\n Valid PVs are defined as: PV(nTracks) >= %d  &  PV(z) >= %.1f  &  PV(z) <= %.1f] \n"%(n_tracks_valid_PVs, z_min, z_max))
        cTuple = Update_validPV_def(cTuple, n_tracks_valid_PVs, z_min, z_max)

    ## -------------------------------------------------------------------------------------------
    ## Below all the new data variables methods are called 
    ## -------------------------------------------------------------------------------------------
    ## Compute tracks Ellipsoid parameters
    ## Since the shape of the tuple is modified when calling Compute_tracks_ellipsoid 
    ## the 6 ellipsoid params being added to the inputTuple!!!
    ## This means the inputTuple for the following methods will need to correspond to the output tuple 
    ## of Compute_tracks_ellipsoid!!!
    print("-"*100)
    with Timer(start="\n Computing tracks' ellipsoid parameters from major, minor1 and minor2 axes... \n"):
        cTuple = Compute_tracks_ellipsoid(cTuple)

    ## ------------------------------------------------------------
    ## Compute target histograms for poca_KDE (with binning scheme corresponding to the one used for the input poca_KDE)
    ## It will return a tuple with the new target histogram variable.
    ## This means the inputTuple for the following methods will need to correspond to the output tuple 
    ## of Compute_target_poca_KDE!!!
    print("-"*100)
    with Timer(start="\n Computing target histograms for poca_KDE with binning scheme: N_bins = %s... \n"%(n_bins_poca_kde)):
        cTuple = Compute_target_hists(cTuple, n_bins_poca_kde, n_tracks_valid_PVs, global_configs)    

    ## ------------------------------------------------------------
    ## Apply splitting of the data into intervals: KDE, target histograms as well as the tracks need to be split
    if doIntervals:
        print("-"*100)
        with Timer(start="\n Splitting the data into intervals requested. Splitting now...\n"):
            cTuple = split_data_intervals(cTuple, global_configs)    

    ## ------------------------------------------------------------
    ## At last, save the final tuple with the high level variables into hdf5 format
    print("-"*100)
    with Timer(start=f"\n Saving to {output_fname}... \n"):
        Save_tuple_to_hdf5(output_fname, cTuple)
    
    ## -------------------------------------------------------------------------------------------
    ## Get the global time to process the data
    global_time = time.time() - global_start_time
    
    print("-"*100)
    print("")
    print("Global time to process data: %.4f s"%global_time)
    print("")
    print("-"*100)
    ## -------------------------------------------------------------------------------------------
    
## ------------------------------------------------------------
## ------------------------------------------------------------
## ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This processes files, in multiple processes. "
        "Example of how to run the script: "
        "./Make_HDF5_files_from_ROOT_files.py -c configuration_default.yaml -o yourCleverOutPutFileName.h5 -f pv_HLT1CPU_MinBiasMagUp_*.root"
    )

    ## -------------------------------------------------------------------------------------------
    ## List of required arguments
    parser.add_argument(
        "-c", "--configs", type=str, required=True, nargs="+", help="Name of the configuration file"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Set the output file (.h5)"
    )
    parser.add_argument(
        "-f", "--files", type=str, required=True, nargs="+", help="The files to read in"
    )

    ## -------------------------------------------------------------------------------------------
    ## Retrieve the parsed arguments 
    args = parser.parse_args()
    
    ## -------------------------------------------------------------------------------------------
    ## Retrieve the configuration file name
    configs_file_name = args.configs[0]
    
    ## -------------------------------------------------------------------------------------------
    ## Get the configuration parameters from the considered yaml file
    current_dir = os.getcwd()
    if "scripts" in current_dir:
        current_dir = current_dir[:current_dir.find("pv-finder/")+len("pv-finder/")]
    CONFIG = current_dir + '/configs/' + configs_file_name
    with open(CONFIG, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    ## -------------------------------------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## Get the global configuration (one for each experiment, selected based on the --experiment argument)    
    global_configs = configs['global_configs'][Exp]
    #print("global_config dict:",global_config)
    
    ## Get the dataset configuration defining the locations of the input and output files
    dataset_configs = configs['data_processing_configs']
    #print("dataset_config dict:",dataset_config)
        
    ## -------------------------------------------------------------------------------------------
    ## Edit the list of input files path by adding the root path defined in the dataset_configs
    input_root = PosixPath(dataset_configs['input_files_path'])
    l_inputs_tmp = []

    for this_file in args.files:
        if "*" in this_file:
            l_inputs_tmp += glob.glob('%s/%s'%(input_root,this_file))
        else:
            l_inputs_tmp += [glob.glob('%s/%s'%(input_root,this_file))[0]]
            
    l_inputs = []
    for this_input in l_inputs_tmp:
        l_inputs.append(PosixPath(this_input))
    
    ## -------------------------------------------------------------------------------------------
    ## Edit the output file path by adding the root path defined in the dataset_configs
    output_root = PosixPath(dataset_configs['output_files_path'])
    output_file_name = output_root / args.output 
    
    ## Add some options from the configuration file:
    output_file_name = getDataTagName(str(output_file_name), global_configs)
    output_file_name = PosixPath(str(output_file_name) + ".hdf5")
            
    ## -------------------------------------------------------------------------------------------
    ## -------------------------------------------------------------------------------------------
    ## Call all data preparation methods from the main
    main(output_file_name, l_inputs, Exp, configs)
    
    ## -------------------------------------------------------------------------------------------
    ## END OF FILE
    ## -------------------------------------------------------------------------------------------
    