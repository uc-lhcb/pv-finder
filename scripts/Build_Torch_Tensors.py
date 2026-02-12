#!/usr/bin/env python3
'''
Main script to build the torch tensors from the HDF5 files variables. 

The main goal is to have tensors directly accessible to load saving some time when operating model training.

The following possible variables are currently considered:

- tracks POCA (input tensor for the tracks-to-KDE or tracks-to-hist models)

- KDE         (input to the KDE-to-hist or target of the tracks-to-hist models)

- target hist (target of the tracks-to-hist or KDE-to-hist models)


Note that this method DO NOT use as option the number of events and transforms ALL EVENTS from the HDF5 file to be saved into torch tensors. 

The number of events to be considered is used when loading the data when actually performing the training. This allows not to have to repeat the tensor creation. However, the downside is the this duplicates data to be stored on disk, which can be a limiting factor if dealing with very large number of input events.

# -------------------------------------------
  TO RUN CODE, from terminal:

  ```
  cd pv-finder_v2/
  ```
  ./scripts/Build_Torch_Tensors.py -c configuration_default.yaml -i $FILENAME
  ```
  
  where:
  
  - "configuration_default.yaml" is the configuration file name
    it can be any configuration file name
    
  - "$FILENAME" is the HDF5 file (without the .hdf5 extension) 
    built from the scripts/Make_HDF5_files_from_ROOT_files.py script
  
# -------------------------------------------

'''

import awkward as ak
import numpy as np
import argparse
import warnings
import yaml
import h5py
from pathlib import Path, PosixPath

import torch
from torch.utils.data import TensorDataset

from utils.utilities import Get_ak_from_hdf5, Timer, getDataTagName

def main(input_fname, output_fname, data_configs):

    ## ------------------------------------------------------------
    ## Open input HDF5 file
    print("-"*100)
    with Timer(start="\n Opening input hdf5 file \n"):
        print("   `--> ",input_fname)
        input_file = h5py.File(input_fname, "r")
        
    ## ------------------------------------------------------------
    ## Define float32 precision
    ## To be applied to all variables
    FP_TYPE = np.float32
    ## ------------------------------------------------------------
        
    '''
    Note: Every input AWKWARD array from the input HDF5 file is transformed into NUMPY array 
          as the time to build a pytorch tensor from awkward arrays is slower by a factor 1000! 
    '''
    
    
    ########################################################################################    
    ## ------------------------------------------------------------
    ## Persisting TRACKS data (split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_configs["persist_tracks_inter_tensor"]:
        
        tracksData_name = output_fname + "_tracksData_inter.pt"        
        with Timer(start="\n Persisting tracks data (split in intervals) into pytorch tensor... This can take a couple of minutes depending on the input file size! \n"):
            poca_x = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_x_inter")).astype(FP_TYPE)
            poca_y = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_y_inter")).astype(FP_TYPE)
            poca_z = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_z_inter")).astype(FP_TYPE)
            poca_A = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_A_inter")).astype(FP_TYPE)
            poca_B = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_B_inter")).astype(FP_TYPE)
            poca_C = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_C_inter")).astype(FP_TYPE)
            poca_D = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_D_inter")).astype(FP_TYPE)
            poca_E = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_E_inter")).astype(FP_TYPE)
            poca_F = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_F_inter")).astype(FP_TYPE)

            '''
            IMPORTANT !!!!! 
            
            Tracks input are returned with Z in first position. 
            This comes from historical reasons and to keep backward compatibility.
            
            IMPORTANT !!!!!             
            '''

            tracksData = np.concatenate((poca_z,
                                         poca_x,
                                         poca_y,
                                         poca_A,
                                         poca_B,
                                         poca_C,
                                         poca_D,
                                         poca_E,
                                         poca_F),axis=1)
            
            tracksData_pt   = torch.tensor(tracksData)
            torch.save(tracksData_pt, tracksData_name)
            print("Saved tracks tensor in:",tracksData_name)
            print("-"*100)
            
    ## ------------------------------------------------------------
    ## Persisting KDE data (split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_configs["persist_KDE_inter_tensor"]:

        poca_KDE_A_name = output_fname + "_poca_KDE_A_inter.pt"        
        with Timer(start="\n Persisting KDE_A data (split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_A
            poca_KDE_A      = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A_inter")).astype(FP_TYPE)
            poca_KDE_A_pt   = torch.tensor(poca_KDE_A)
            torch.save(poca_KDE_A_pt, poca_KDE_A_name)
            print("Saved KDE_A tensor in:",poca_KDE_A_name)
            print("-"*100)
            
        poca_KDE_A_xMax_name = output_fname + "_poca_KDE_A_xMax_inter.pt"
        with Timer(start="\n Persisting KDE_A_xMax data (split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_A_xMax
            poca_KDE_A_xMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A_xMax_inter")).astype(FP_TYPE)
            poca_KDE_A_xMax_pt   = torch.tensor(poca_KDE_A_xMax)
            torch.save(poca_KDE_A_xMax_pt, poca_KDE_A_xMax_name)
            print("Saved KDE_A_xMax tensor in:",poca_KDE_A_xMax_name)
            print("-"*100)

        poca_KDE_A_yMax_name = output_fname + "_poca_KDE_A_yMax_inter.pt"
        with Timer(start="\n Persisting KDE_A_yMax data (split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_A_yMax
            poca_KDE_A_yMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A_yMax_inter")).astype(FP_TYPE)
            poca_KDE_A_yMax_pt   = torch.tensor(poca_KDE_A_yMax)
            torch.save(poca_KDE_A_yMax_pt, poca_KDE_A_yMax_name)
            print("Saved KDE_A_yMax tensor in:",poca_KDE_A_yMax_name)
            print("-"*100)

        poca_KDE_B_name = output_fname + "_poca_KDE_B_inter.pt"
        with Timer(start="\n Persisting KDE_B data (split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_B
            poca_KDE_B      = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B_inter")).astype(FP_TYPE)
            poca_KDE_B_pt   = torch.tensor(poca_KDE_B)
            torch.save(poca_KDE_B_pt, poca_KDE_B_name)
            print("Saved KDE_B tensor in:",poca_KDE_B_name)
            print("-"*100)
            
        poca_KDE_B_xMax_name = output_fname + "_poca_KDE_B_xMax_inter.pt"
        with Timer(start="\n Persisting KDE_B_xMax data (split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_B_xMax
            poca_KDE_B_xMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B_xMax_inter")).astype(FP_TYPE)
            poca_KDE_B_xMax_pt   = torch.tensor(poca_KDE_B_xMax)
            torch.save(poca_KDE_B_xMax_pt, poca_KDE_B_xMax_name)
            print("Saved KDE_B_xMax tensor in:",poca_KDE_B_xMax_name)
            print("-"*100)
            
        poca_KDE_B_yMax_name = output_fname + "_poca_KDE_B_yMax_inter.pt"
        with Timer(start="\n Persisting KDE_B_yMax data (split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_B_yMax
            poca_KDE_B_yMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B_yMax_inter")).astype(FP_TYPE)
            poca_KDE_B_yMax_pt   = torch.tensor(poca_KDE_B_yMax)
            torch.save(poca_KDE_B_yMax_pt, poca_KDE_B_yMax_name)
            print("Saved KDE_B_yMax tensor in:",poca_KDE_B_yMax_name)
            print("-"*100)
            
            
    ## ------------------------------------------------------------
    ## Persisting TARGET HIST data (split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_configs["persist_targetHist_inter_tensor"]:

        targetHists_name = output_fname + "_targetHists_inter.pt"
        with Timer(start="\n Persisting target hist data (split in intervals) into pytorch tensor... \n"):
            targetHists = ak.to_numpy(Get_ak_from_hdf5(input_file,"targetHists_inter")).astype(FP_TYPE)
            # Since targetHists is a [4xN] structure with 
            #
            # [0] for valid PVs, 
            # [1] for invalid PVs, 
            # [2] for valid SVs,
            # [3] for invalid SVs,
            #
            # we only want to get [0] for the moment... 
            # Could be updated in the future if needed 
            targetHists_pt   = torch.tensor(targetHists[0])
            torch.save(targetHists_pt, targetHists_name)
            print("Saved target hist tensor in:",targetHists_name)
            print("-"*100)
            
    ########################################################################################    
    ########################################################################################    

    ## ------------------------------------------------------------
    ## Persisting TRACKS data (** NOT ** split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_configs["persist_tracks_tensor"]:
        
        tracksData_name = output_fname + "_tracksData.pt"        
        with Timer(start="\n Persisting tracks data (** NOT ** split in intervals) into pytorch tensor... This can take a couple of minutes depending on the input file size! \n"):
            poca_x = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_x")).astype(FP_TYPE)
            poca_y = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_y")).astype(FP_TYPE)
            poca_z = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_z")).astype(FP_TYPE)
            poca_A = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_A")).astype(FP_TYPE)
            poca_B = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_B")).astype(FP_TYPE)
            poca_C = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_C")).astype(FP_TYPE)
            poca_D = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_D")).astype(FP_TYPE)
            poca_E = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_E")).astype(FP_TYPE)
            poca_F = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_F")).astype(FP_TYPE)

            '''
            IMPORTANT !!!!! 
            
            Tracks input are returned with Z in first position. 
            This comes from historical reasons and to keep backward compatibility.
            
            IMPORTANT !!!!!             
            '''

            tracksData = np.concatenate((poca_z,
                                         poca_x,
                                         poca_y,
                                         poca_A,
                                         poca_B,
                                         poca_C,
                                         poca_D,
                                         poca_E,
                                         poca_F),axis=1)
            
            tracksData_pt   = torch.tensor(tracksData)
            torch.save(tracksData_pt, tracksData_name)
            print("Saved tracks tensor in:",tracksData_name)
            print("-"*100)
            
    ## ------------------------------------------------------------
    ## Persisting KDE data (** NOT ** split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_configs["persist_KDE_tensor"]:

        poca_KDE_A_name = output_fname + "_poca_KDE_A.pt"        
        with Timer(start="\n Persisting KDE_A data (** NOT ** split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_A
            poca_KDE_A      = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A")).astype(FP_TYPE)
            poca_KDE_A_pt   = torch.tensor(poca_KDE_A)
            torch.save(poca_KDE_A_pt, poca_KDE_A_name)
            print("Saved KDE_A tensor in:",poca_KDE_A_name)
            print("-"*100)
            
        poca_KDE_A_xMax_name = output_fname + "_poca_KDE_A_xMax.pt"
        with Timer(start="\n Persisting KDE_A_xMax data (** NOT ** split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_A_xMax
            poca_KDE_A_xMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A_xMax")).astype(FP_TYPE)
            poca_KDE_A_xMax_pt   = torch.tensor(poca_KDE_A_xMax)
            torch.save(poca_KDE_A_xMax_pt, poca_KDE_A_xMax_name)
            print("Saved KDE_A_xMax tensor in:",poca_KDE_A_xMax_name)
            print("-"*100)

        poca_KDE_A_yMax_name = output_fname + "_poca_KDE_A_yMax.pt"
        with Timer(start="\n Persisting KDE_A_yMax data (** NOT ** split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_A_yMax
            poca_KDE_A_yMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A_yMax")).astype(FP_TYPE)
            poca_KDE_A_yMax_pt   = torch.tensor(poca_KDE_A_yMax)
            torch.save(poca_KDE_A_yMax_pt, poca_KDE_A_yMax_name)
            print("Saved KDE_A_yMax tensor in:",poca_KDE_A_yMax_name)
            print("-"*100)

        poca_KDE_B_name = output_fname + "_poca_KDE_B.pt"
        with Timer(start="\n Persisting KDE_B data (** NOT ** split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_B
            poca_KDE_B      = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B")).astype(FP_TYPE)
            poca_KDE_B_pt   = torch.tensor(poca_KDE_B)
            torch.save(poca_KDE_B_pt, poca_KDE_B_name)
            print("Saved KDE_B tensor in:",poca_KDE_B_name)
            print("-"*100)
            
        poca_KDE_B_xMax_name = output_fname + "_poca_KDE_B_xMax.pt"
        with Timer(start="\n Persisting KDE_B_xMax data (** NOT ** split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_B_xMax
            poca_KDE_B_xMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B_xMax")).astype(FP_TYPE)
            poca_KDE_B_xMax_pt   = torch.tensor(poca_KDE_B_xMax)
            torch.save(poca_KDE_B_xMax_pt, poca_KDE_B_xMax_name)
            print("Saved KDE_B_xMax tensor in:",poca_KDE_B_xMax_name)
            print("-"*100)
            
        poca_KDE_B_yMax_name = output_fname + "_poca_KDE_B_yMax.pt"
        with Timer(start="\n Persisting KDE_B_yMax data (** NOT ** split in intervals) into pytorch tensor... \n"):
            ## -------------------
            ## poca_KDE_B_yMax
            poca_KDE_B_yMax = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B_yMax")).astype(FP_TYPE)
            poca_KDE_B_yMax_pt   = torch.tensor(poca_KDE_B_yMax)
            torch.save(poca_KDE_B_yMax_pt, poca_KDE_B_yMax_name)
            print("Saved KDE_B_yMax tensor in:",poca_KDE_B_yMax_name)
            print("-"*100)
            
    ## ------------------------------------------------------------
    ## Persisting TARGET HIST data (** NOT ** split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_configs["persist_targetHist_tensor"]:

        targetHists_name = output_fname + "_targetHists.pt"
        with Timer(start="\n Persisting target hist data (** NOT ** split in intervals) into pytorch tensor... \n"):
            targetHists = ak.to_numpy(Get_ak_from_hdf5(input_file,"targetHists")).astype(FP_TYPE)
            
            # Since targetHists is a [4xN] structure with 
            #
            # [0] for valid PVs, 
            # [1] for invalid PVs, 
            # [2] for valid SVs,
            # [3] for invalid SVs,
            #
            # we only want to get [0] for the moment... 
            # Could be updated in the future if needed 
            targetHists_pt   = torch.tensor(targetHists[0])
            torch.save(targetHists_pt, targetHists_name)
            print("Saved target hist tensor in:",targetHists_name)
            print("-"*100)
    
## ------------------------------------------------------------
## ------------------------------------------------------------
## ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This processes files, in multiple processes. "
        "Example of how to run the script: "
        "./scripts/Build_Torch_Tensors.py -c configuration_default.yaml -i $FILENAME"
    )

    ## -------------------------------------------------------------------------------------------
    ## List of required arguments
    parser.add_argument(
        "-c", "--configs", type=str, required=True, nargs="+", help="Name of the configuration file"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Set the input file dataTag (same as used for Make_HDF5_files_from_ROOT_files.py)"
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
        current_dir = current_dir[:current_dir.find("pv-finder_v2/")+len("pv-finder_v2")]    
    CONFIG = current_dir + '/configs/' + configs_file_name
    with open(CONFIG, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    ## -------------------------------------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## Get the global configuration (one for each experiment, selected based on the --experiment argument)    
    global_configs = configs['global_configs'][Exp]
    
    ## Get the dataset configuration defining the locations of the input and output files
    dataset_configs = configs['data_processing_configs']
    #print("dataset_config dict:",dataset_config)
        
    ## -------------------------------------------------------------------------------------------
    ## Edit the input file path by adding the root path defined in the dataset_configs
    root = dataset_configs['output_files_path']
    
    input_file = root + "/" + args.input    
    input_file = getDataTagName(input_file, global_configs)
    input_file += ".hdf5"
    
    output_file_tag = input_file[:-5]

    ## -------------------------------------------------------------------------------------------
    ## -------------------------------------------------------------------------------------------
    ## Call all data preparation methods from the main
    main(input_file, output_file_tag, dataset_configs)

    ## -------------------------------------------------------------------------------------------
    ## END OF FILE
    ## -------------------------------------------------------------------------------------------

