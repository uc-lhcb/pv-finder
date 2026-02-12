'''
File containing most of the methods usefull accross many scripts
'''

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
import sys
import time
from datetime import datetime
from pathlib import Path, PosixPath
import awkward as ak
import h5py as h5py
import json
import numpy as np
import pandas as pd
from collections import namedtuple
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

import scipy
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import cosine as sp_cos

from sklearn.preprocessing  import normalize
from torch_geometric.utils  import dense_to_sparse, to_edge_index, to_torch_coo_tensor, to_dense_adj
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader

from tools.ToolBox import norm_cdf, get_res

## ========================================================================================
def checkFileExists(path,tag=""):
    
    if tag=="":
        if not os.path.exists(path):
            print("ERROR:: file %s does not exist!"%(path))
            sys.exit()
    if tag=="dataOnDisk":
        if not os.path.exists(path):
            print("ERROR:: file %s does not exist!"%(path))
            print("")
            print("ERROR:: Requested accessing input data from persisted tensors,")
            print("ERROR:: but it seems these have not been created.")
            print("")
            print("ERROR:: Run scripts/Build_Torch_Tensors.py to save tensors on disk before requesting dataOnDemand=True")
            sys.exit()

## ========================================================================================
class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""

    __slots__ = ("file", "progress")

    def __init__(self, file, progress):
        self.file = file
        self.progress = progress

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            self.progress.write(x.strip(), file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


## ========================================================================================
@contextmanager
def tqdm_redirect(progress):
    old_out = sys.stdout

    if hasattr(progress, "postfix"):
        with redirect_stdout(DummyTqdmFile(sys.stdout, progress)), redirect_stderr(
            DummyTqdmFile(sys.stderr, progress)
        ):
            yield old_out
    else:
        yield old_out


## ========================================================================================
def select_gpu(selection=None):
    """
    Select a GPU if availale.

    Selection can be set to get a specific GPU. If left unset, it will REQUIRE that a GPU be selected by environment variable. 
    If -1, the CPU will be selected.
    """

    if str(selection) == "-1":
        return torch.device("cpu")

    ## ----------------------------------------------------------------------------------------
    # This must be done before any API calls to Torch that touch the GPU
    if selection is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selection)

    if not torch.cuda.is_available():
        print("Selecting CPU (CUDA not available)")
        return torch.device("cpu")
    elif selection is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is *required* when running with CUDA available"
        )

    print(torch.cuda.device_count(), "available GPUs (initially using device 0):")
    for i in range(torch.cuda.device_count()):
        print(" ", i, torch.cuda.get_device_name(i))

    return torch.device("cuda:0")

## ========================================================================================
def pv_isolation(pv_z, min_dz=2.5):
    pv_iso = np.ones(ak.to_numpy(pv_z).shape)
    for i_pv in range(0,len(pv_z)):
        dz_l = 10
        dz_r = 10
        if (i_pv-1) >= 0:
            dz_l = pv_z[i_pv]-pv_z[i_pv-1]
            
        if (i_pv+1) < len(pv_z):
            dz_r = pv_z[i_pv+1]-pv_z[i_pv]
        
        if dz_l<min_dz or dz_r<min_dz:
            pv_iso[i_pv] = 0
                        
    return pv_iso        

## ========================================================================================
def tracks_isolation_only(tracks_z, min_dz=0.5):
    tracks_iso = np.ones(ak.to_numpy(tracks_z).shape)
    for iTrack in range(0,len(tracks_z)):
        dz_l = 1
        dz_r = 1
        if (iTrack-1) >= 0:
            dz_l = tracks_z[iTrack]-tracks_z[iTrack-1]
            
        if (iTrack+1) < len(tracks_z):
            dz_r = tracks_z[iTrack+1]-tracks_z[iTrack]
        
        if dz_l<min_dz or dz_r<min_dz:
            tracks_iso[iTrack] = 0
                        
    return tracks_iso


## ========================================================================================
def tracks_weighted_pos(pv_key, 
                        tracks_x, tracks_y, tracks_z, 
                        tracks_x_sigma, tracks_y_sigma, tracks_z_sigma,
                        tracks_pv_key
                       ):

    w_x = np.power(tracks_x_sigma,-2)
    w_y = np.power(tracks_y_sigma,-2)
    w_z = np.power(tracks_z_sigma,-2)
    
    sum_w_x = np.sum(w_x[tracks_pv_key==pv_key])
    sum_w_y = np.sum(w_y[tracks_pv_key==pv_key])
    sum_w_z = np.sum(w_z[tracks_pv_key==pv_key])

    sum_w_x_sq = np.power(sum_w_x,2)
    sum_w_y_sq = np.power(sum_w_y,2)
    sum_w_z_sq = np.power(sum_w_z,2)

    mu_x = np.sum(w_x[tracks_pv_key==pv_key] * tracks_x[tracks_pv_key==pv_key])/sum_w_x
    mu_y = np.sum(w_y[tracks_pv_key==pv_key] * tracks_y[tracks_pv_key==pv_key])/sum_w_y
    mu_z = np.sum(w_z[tracks_pv_key==pv_key] * tracks_z[tracks_pv_key==pv_key])/sum_w_z

    s_x = 1./np.sqrt(sum_w_x)
    s_y = 1./np.sqrt(sum_w_y)
    s_z = 1./np.sqrt(sum_w_z)
        
    return mu_x, mu_y, mu_z, s_x, s_y, s_z


## ========================================================================================
def import_progress_bar(notebook):
    """Set up notebook or regular progress bar.

    If None or if piping to a file, just provide an empty do-nothing function."""

    def progress(iterator, **kargs):
        return iterator

    if notebook is None:
        pass
    elif notebook:
        from tqdm import tqdm_notebook as progress
    elif sys.stdout.isatty():
        from tqdm import tqdm as progress
    else:
        # Don't display progress if this is not a
        # notebook and not connected to the terminal
        pass

    return progress


## ========================================================================================
class Timer(object):
    __slots__ = "message verbose start_time".split()

    def __init__(self, message=None, start=None, verbose=True):
        """
        If message is None, add a default message.
        If start is not None, then print start then message.
        Turn off all printing with verbose.
        """

        if verbose and start is not None:
            print(start, end="", flush=True)
        if message is not None:
            self.message = message
        elif start is not None:
            self.message = "\n ==> took {time:.4} s"
        else:
            self.message = "\n ==> Operation took {time:.4} s"

        self.verbose = verbose
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.verbose:
            print(self.message.format(time=self.elapsed_time()))


## ========================================================================================
def get_device_from_model(model):
    if hasattr(model, "weight"):
        return model.weight.device
    else:
        return get_device_from_model(list(model.children())[0])

    
## ========================================================================================
def Save_tuple_to_hdf5(hdf5_file_name,named_tuple):
    """
    Method to save a named tuple with awkward arrays into hdf5 format.
    This method can be used when processing the original input files in ROOT format.
    Documentation can be found here:
    https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#saving-awkward-arrays-to-hdf5
    """
    ## Create the output hdf5 file
    file = h5py.File(hdf5_file_name, "w")
    ## Looping over the variables, a.k.a the "fields"
    for field in named_tuple._fields:
        group = file.create_group(field)
        form, length, container = ak.to_buffers(ak.to_packed(getattr(named_tuple, field)), container=group)
        group.attrs["form"] = form.to_json()
        group.attrs["length"] = json.dumps(length)
    file.close()
    
    
## ========================================================================================
def Get_ak_from_hdf5(hdf5_file,var):
    """
    Method to read an hdf5 and get the stored awkward arrays by explicit name: "var" 
    
    - Example: 
    
    # file = h5py.File("WolfgangPauli.hdf5", "r")
    # pv_loc_x = Get_ak_from_hdf5(file,"pv_loc_x")
    # poca_x = Get_ak_from_hdf5(file,"poca_x")
     
    This assumes the hdf5 file was created using the 'save_tuple_to_hdf5' 
    method or any other method creating groups with labels (or field)

    Documentation can be found here:
    https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#reading-awkward-arrays-from-hdf5

    """
    group = hdf5_file[var]
    var_ak = ak.from_buffers(
        ak.forms.from_json(group.attrs["form"]),
        json.loads(group.attrs["length"]),
        {k: np.asarray(v) for k, v in group.items()},
    )
    return var_ak

## ========================================================================================
def Set_ak_values_to_zero_condition(ak_ref,ak_mod):

    '''
    Since awkward arrays cannot have the values changed directly, one can trasnform the array 
    into a numpy array to do the computation, then trasmform it back to an awkward array
    ''' 
    # Flatten the reference ak array and get the shape 
    # (will be used to unflatten the modified ak array after)
    ak_ref_flat, shape = ak.flatten(ak_ref), ak.num(ak_ref)
    # Flatten the ak array to be modified
    ak_mod_flat = ak.flatten(ak_mod)
    # Transform the ak array to a np array
    ak_mod_flat = ak.to_numpy(ak_mod_flat)
    # Modify the np array entries based on the values of the ref array
    ak_mod_flat[0 == ak_ref_flat] = 0
    # Transform the np array back to an ak array
    ak_mod_flat = ak.from_numpy(ak_mod_flat)
    # Unflatten the ak array using the original shape
    ak_mod = ak.unflatten(ak_mod_flat,shape)
    
    return ak_mod

## ========================================================================================
def makeOutPutFolders(data_tag, training_tag, configs, configsName):

    ## Make the global output directory first (if not already done)
    output_dir = Path(configs["training_configs"]["output_files_path"])
    output_dir.mkdir(exist_ok=True)
    
    ## Make the data tag specific output directory (if not already done)
    output_dir = output_dir / data_tag
    output_dir.mkdir(exist_ok=True)    
    
    ## Make the training specific output directory (if not already done)
    output_dir = output_dir / training_tag
    output_dir.mkdir(exist_ok=True)    
    
    ## Copy the configuration file used into this directory so one can
    ## double check what parameters values were used for this specific training 
    os.system("cp %s %s/."%(configsName,output_dir))

    ## Make the output directory for the training weights (if not already done)
    output_dir_w = output_dir / "weights"
    output_dir_w.mkdir(exist_ok=True)    

    ## Make the output directory for the training plots (if not already done)
    output_dir_p = output_dir / "plots"
    output_dir_p.mkdir(exist_ok=True)    

    ## Return the final output_dir (the ones for the weights) to be used to store the weights
    return output_dir_w, output_dir_p

## ========================================================================================
def getTrainTagName(configs):
        
    ## Add some options from the config file:
    tag = "%s"%(configs["model_class"])
    tag = tag + "__Intervals%s"%(configs["training_configs"]["doIntervals"])
    tag = tag + "__%sEpochs"%(configs["training_configs"]["n_epochs"])
    tag = tag + "__BatchSize%s"%(configs["training_configs"]["batch_size"])
    if configs["model_class"]=="tracks-to-hist" or configs["model_class"]=="KDE-to-hist":
        coeff = str(configs["training_configs"]["asym_coeff"])
        coeff = coeff.replace(".","p")
        tag = tag + "__AsymCoeff%s"%(coeff)
        
    return tag
    
## ========================================================================================
def getDataTagName(tag, configs, do_GNN_preprocess=False):
    
    ## Add some options from the configuration file:
    if do_GNN_preprocess:
        tag = tag + "__GNN_preprocessed"        
    else:
        tag = tag + "__validPVs_%stracks"%(configs["n_tracks_valid_PVs"])
        if configs["doIntervals"]:
            tag = tag + "__Intervals_%sbins"%(configs["nBinsPerInterval"])

    return tag
    
## ========================================================================================
def get_split_indices(full_length, split_fractions, split_seed, shuffle):
    
    ## --------------------------------------------------------
    sum_fracs = np.sum(split_fractions)
    if not sum_fracs==1:
        print("Sum of split_fractions (= %.2f) different from 1.0!"%(sum_fracs))
        print("Please provide a list of split fractions that sums to one!")
        sys.exit()
        
    ## --------------------------------------------------------    
    print("Splitting dataset of initialize length",full_length)
    print("according to the following fractions: %.2f (train) / %.2f (validation) / %.2f (test)"%(split_fractions[0],split_fractions[1],split_fractions[2]))
    ## --------------------------------------------------------    
    ## Define a list of indices of size full_length==len(full_dataset)
    indices = list(range(full_length))
    if shuffle:
        ## Set a seed with a given value for reproducibility. 
        ## By default split_seed is set to 1
        np.random.seed(split_seed)
        np.random.shuffle(indices)
    ## --------------------------------------------------------    
    ## Define the amount of events for each split data
    n_train = int(np.floor(split_fractions[0] * full_length))
    n_val   = int(np.floor(split_fractions[1] * full_length))
    n_test  = int(np.floor(split_fractions[2] * full_length))
    print("")
    print("Sizes of split samples: %s (train) / %s (validation) / %s (test)"%(n_train,n_val,n_test))
    ## --------------------------------------------------------    
    if (n_train+n_val+n_test)<full_length:
        print("Last %s event(s) discarded due to rounding..."%(full_length-(n_train+n_val+n_test)))
        print("This corresponds to %.3f %% percent of full data."%(100*(1.-(n_train+n_val+n_test)/float(full_length))))
    ## --------------------------------------------------------    
    return indices[:n_train], indices[n_train:(n_train+n_val)], indices[(n_train+n_val):(n_train+n_val+n_test)]


## ========================================================================================
def get_split_slices(slice_evts, split_fractions, doIntervals=True, nIntervals_perEvt=40):

    print("")
    print("="*100)
    ## --------------------------------------------------------
    sum_fracs = np.sum(split_fractions)
    if not sum_fracs==1:
        print("")
        print("Sum of split_fractions (= %.2f) different from 1.0!"%(sum_fracs))
        print("Please provide a list of split fractions that sums to one!")
        print("")
        sys.exit()
    
    ## --------------------------------------------------------    
    ## Define the starting and stopping points of the slice [in events]
    iEvt_start  = slice_evts.start
    iEvt_stop   = slice_evts.stop
    nEvts_total = iEvt_stop-iEvt_start
    
    ## --------------------------------------------------------    
    print("")
    print("Splitting dataset with original slice_evts(%s,%s)"%(int(iEvt_start),int(iEvt_stop)))
    print("according to the following fractions: %.2f (train) / %.2f (validation) / %.2f (test)"%(split_fractions[0],split_fractions[1],split_fractions[2]))
    ## --------------------------------------------------------    
    ## Define the amount of events for each split data
    nEvts_train = int(np.floor(split_fractions[0] * nEvts_total))
    nEvts_val   = int(np.floor(split_fractions[1] * nEvts_total))
    nEvts_test  = int(np.floor(split_fractions[2] * nEvts_total))    
    print("")
    print("Sizes of split samples [events]: %s (train) / %s (validation) / %s (test)"%(nEvts_train,nEvts_val,nEvts_test))
    ## --------------------------------------------------------    
    if (nEvts_train+nEvts_val+nEvts_test)<nEvts_total:
        print("")
        print("Last %s event(s) discarded due to rounding..."%(nEvts_total-(nEvts_train+nEvts_val+nEvts_test)))
        print("This corresponds to %.3f %% percent of full data."%(100*(1.-(nEvts_train+nEvts_val+nEvts_test)/float(nEvts_total))))
        print("")
    ## --------------------------------------------------------    
    if doIntervals:
        ## Define the amount of intervals for each split data
        nInts_train = nEvts_train*nIntervals_perEvt
        nInts_val   = nEvts_val  *nIntervals_perEvt
        nInts_test  = nEvts_test *nIntervals_perEvt
        print("")
        print("Sizes of split samples [intervals]: %s (train) / %s (validation) / %s (test)"%(nInts_train,nInts_val,nInts_test))
    
    ## --------------------------------------------------------    
    ## Define slices for the training, validation and test samples
    ## in terms of events...
    slice_entries_train = slice(iEvt_start,               iEvt_start+nEvts_train)
    slice_entries_val   = slice(slice_entries_train.stop, slice_entries_train.stop+nEvts_val)
    slice_entries_test  = slice(slice_entries_val.stop,   slice_entries_val.stop+nEvts_test)
    
    if doIntervals:
        ## ... or in terms of intervals of using intervals
        slice_entries_train = slice(slice_entries_train.start*nIntervals_perEvt, slice_entries_train.stop*nIntervals_perEvt)
        slice_entries_val   = slice(slice_entries_val.start  *nIntervals_perEvt, slice_entries_val.stop  *nIntervals_perEvt)
        slice_entries_test  = slice(slice_entries_test.start *nIntervals_perEvt, slice_entries_test.stop *nIntervals_perEvt)
        
    return slice_entries_train, slice_entries_val, slice_entries_test
    print("="*100)
    print("")
        

## ========================================================================================
def from_bins_to_z(data_bins, configs, doInterval = False, i_interval=-1, nBinsPerInterval=-1):
    ## Scaling factor from bin value to z value
    scale  = configs["n_bins_poca_kde"]/(configs["z_max"]-configs["z_min"])
    ## Shift value in case the z minimum is different from 0
    shift = configs["z_min"]
    
    if doInterval:
        if i_interval<0 or nBinsPerInterval<0:
            print("ERROR:: Incorrect argument values for either:")
            print("         - i_interval       = %s (interval id within a given event)"%i_interval)
            print("         - nBinsPerInterval = %s (constant for all intervals)"%nBinsPerInterval)
            sys.exit(1)
        else:
            data_z = [(data_bin_i+i_interval*nBinsPerInterval)/scale+shift for data_bin_i in data_bins]    
    else:
        data_z = [data_bin_i/scale+shift for data_bin_i in data_bins]
        
    return data_z
    

## ========================================================================================
def getTensor(file_name, slice=slice(0,9999999999), verbose=False):

    if verbose:
        print("-"*100)
    
        print("Loading pytorch tensor from\n")
        print("   `--> ",file_name)
        
    tensor = torch.load(file_name)
    tensor = tensor[slice]
    return tensor
    
    if verbose:
        print("-"*100)


## ========================================================================================
def buildTensor(file, data_type, doIntervals, slice=slice(0,9999999999), verbose=False):
    
    if verbose:
        print("-"*100)
    
    ## ------------------------------------------------------------
    ## Open input HDF5 file
    input_file = h5py.File(file, "r")
        
    ## ------------------------------------------------------------
    ## Define float32 precision
    ## To be applied to all variables
    FP_TYPE = np.float32
    ## ------------------------------------------------------------
        
    '''
    Note: Every input AWKWARD array from the input HDF5 file is transformed into NUMPY array 
          as the time to build a pytorch tensor from awkward arrays is slower by a factor 1000! 
    '''
    
    var_intervals_tag = ""
    if doIntervals:
        var_intervals_tag = "_inter"

    print_interval_tag = {True:"(split in intervals)", False:"(** NOT ** split in intervals)"}
    ########################################################################################    
    ## ------------------------------------------------------------
    ## Persisting TRACKS data (split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    if data_type == "tracks":
        
        if verbose:
            print("Building tracks data %s into pytorch tensor... This can take a couple of minutes depending on the input file size! \n"%(print_interval_tag[doIntervals]))
            
        poca_x = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_x%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_y = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_y%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_z = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_z%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_A = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_A%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_B = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_B%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_C = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_C%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_D = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_D%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_E = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_E%s"%(var_intervals_tag))).astype(FP_TYPE)
        poca_F = ak.to_numpy(Get_ak_from_hdf5(input_file, "poca_F%s"%(var_intervals_tag))).astype(FP_TYPE)

        ## ------------------------------------------------------------
        ## If a subset of the data is requested !
        ## Note that nEntries is automatically converted into intervals in
        ## case doIntervals == True
        poca_x = poca_x[slice]
        poca_y = poca_y[slice]
        poca_z = poca_z[slice]
        poca_A = poca_A[slice]
        poca_B = poca_B[slice]
        poca_C = poca_C[slice]
        poca_D = poca_D[slice]
        poca_E = poca_E[slice]
        poca_F = poca_F[slice]

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

        tensor   = torch.tensor(tracksData)
    
    ## ------------------------------------------------------------
    ## Persisting KDE data (split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    elif data_type == "KDE_A":

        if verbose:
            print("Building KDE_A data %s into pytorch tensor... \n"%(print_interval_tag[doIntervals]))
            
        ## -------------------
        ## poca_KDE_A
        poca_KDE_A      = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_A%s"%(var_intervals_tag))).astype(FP_TYPE)

        ## ------------------------------------------------------------
        ## If a subset of the data is requested !
        poca_KDE_A = poca_KDE_A[slice]

        tensor   = torch.tensor(poca_KDE_A)
                
    elif data_type == "KDE_B":
        
        if verbose:
            print("Building KDE_B data %s into pytorch tensor... \n"%(print_interval_tag[doIntervals]))
            
        ## -------------------
        ## poca_KDE_B
        poca_KDE_B      = ak.to_numpy(Get_ak_from_hdf5(input_file,"poca_KDE_B%s"%(var_intervals_tag))).astype(FP_TYPE)

        ## ------------------------------------------------------------
        ## If a subset of the data is requested !
        poca_KDE_B = poca_KDE_B[slice]

        tensor   = torch.tensor(poca_KDE_B)
                        
    ## ------------------------------------------------------------
    ## Persisting TARGET HIST data (split in intervals) into pytorch tensor
    ## ------------------------------------------------------------
    elif data_type == "targetHists":

        if verbose:
            print("Building target hist data %s into pytorch tensor... \n"%(print_interval_tag[doIntervals]))
            
        targetHists = ak.to_numpy(Get_ak_from_hdf5(input_file,"targetHists%s"%(var_intervals_tag))).astype(FP_TYPE)

        # Since targetHists is a [4xN] structure with 
        #
        # [0] for valid PVs, 
        # [1] for invalid PVs, 
        # [2] for valid SVs,
        # [3] for invalid SVs,
        #
        # we only want to get [0] for the moment... 
        # Could be updated in the future if needed 
        targetHists = targetHists[0]

        ## ------------------------------------------------------------
        ## If a subset of the data is requested !
        targetHists = targetHists[slice]

        tensor   = torch.tensor(targetHists)
    
    else:
        print("ERROR:: data_type argument (%s) not currently handled !"%(data_type))
        print("ERROR:: Please select from available options."%(data_type))
        sys.exit()

    if verbose:
        print("-"*100)
        
    ## ------------------------------------------------------------
    return tensor
    
## ========================================================================================
def collectData_fromHDF(configs, l_files, l_slice_evts, split=True, verbose=False, **kargs):
    """
    Method to load data from from HDF5 file, from which pytorch tensors are created on the flight 
    
    Arguments:
    
    - configs: configuration dictionnary
    
    - files: list of HDF5 files to be used for the construction of the tracks, KDE and target hists tensors
    
    - split: boolean to switch between returning either a split dataset for training, validation and testing purposes
        
    - slice_evts: slice in terms of number of events. Is automatically converted into number of intervals if using them
               
    After retriving the individual tensors, the method build a pytorch TensorDataset 
    before embedding it into a pytorch DataLoader.
        
    """

    ## -----------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## -----------------------------------------------------------------
    ## Get the global configuration 
    global_configs = configs['global_configs'][Exp]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # First check if the data to be collected is using intervals or not
    intervals_tag = ""
    intervals_verbose = ""
    if configs["training_configs"]["doIntervals"]:
        intervals_tag = "_inter"
        intervals_verbose = "(split in intervals)"
        
        # -------------------------------------------------------------------------
        # Before loading (or building) the data tensor, we need to compute the number of 
        # intervals corresponding to the requested number of events! 
        # Start by computing the number of intervals per event
        nBinsKDE_perEvt           = global_configs["n_bins_poca_kde"]
        nBinsPerInterval_perEvt   = global_configs["nBinsPerInterval"]
        nIntervals_perEvt         = int(nBinsKDE_perEvt/nBinsPerInterval_perEvt)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## List of tensors for each HDF5 file
    ## Will be used to concatenate the tensors 
    ## after the loop over the list of files
    X_list = []
    Y_list = []
    
    ## In case spliting the data into train, validation and test samples
    X_list_train = []
    X_list_val   = []
    X_list_test  = []    
    Y_list_train = []
    Y_list_val   = []
    Y_list_test  = []
        
        
    nEvts_total = 0
    nEvts_train = 0
    nEvts_val   = 0
    nEvts_test  = 0
        
    for idx, file in enumerate(l_files):

        print("")
        print("*"*100)
        print("*"*100)
        with Timer(start="\n Collecting data from input hdf5 file \n"):
            print("   `--> ",file)
            print("")
            print("-"*100)
            print("")
            
            slice_evts = l_slice_evts[idx]
            
            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # Get the total number of entries (could by events or intervals) in the input 
            # HDF5 file (or stored tensor) from the KDE_A tensor 
            # Here since no slice argument is given, "all" entries are returned: 
            # by default, slice=slice(0,9999999999)
            KDE_A_pt = buildTensor(file, 
                                   "KDE_A", 
                                   global_configs["doIntervals"],
                                   slice(0,9999999999),
                                   verbose)

            # -------------------------------------------------------------------------
            # Get the number of entries in the data file (can be either events or intervals)
            nEntries_data = KDE_A_pt.shape[0]
            # Check that the number of intervals per event is the same between the input data 
            # and the configuration file
            if configs["training_configs"]["doIntervals"]:                     
                if not nBinsPerInterval_perEvt==int(KDE_A_pt.shape[1]):
                    print("ERROR:: Number of intervals in input data (%s) different from the one defined in the configuration file (%s)"%(KDE_A_pt.shape[1],nBinsPerInterval_perEvt))
                    sys.exit()
            print("nEntries_data",nEntries_data)
            # -------------------------------------------------------------------------
            # Get the corresponding total number of events in the input data
            nEvts_data = int(nEntries_data)
            
            # If data split in intervals, then we need to divide the number of entries 
            # by the number of intervals per event to get the number of events
            if configs["training_configs"]["doIntervals"]:
                nEvts_data = int(nEntries_data/nIntervals_perEvt)
            print("nEvts_data",nEvts_data)

            # -------------------------------------------------------------------------
            # Check if the requested number of events to be used is not larger than the actual
            # total number of events in the input file
            if slice_evts.start > nEvts_data:
                print("Slice starting event (%s) larger than the maximum number of available events in the input file (%s)"%(slice_evts.start,nEvts_data))
                sys.exit()

            if slice_evts.stop > nEvts_data:
                print("WARNING:: Slice stopping event (%s) larger than the maximum number of available events in the input file (%s)"%(slice_evts.stop,nEvts_data))
                print("Slice stopping event set back to %s"%(nEvts_data))
                slice_evts = slice(slice_evts.start,nEvts_data)

            if slice_evts.stop == -1:
                slice_evts = slice(slice_evts.start,nEvts_data)

            print("slice_evts",slice_evts)
            nEvts_total += (slice_evts.stop - slice_evts.start)
            print("nEvts_total",nEvts_total)
            # -------------------------------------------------------------------------
            # Now to allow using the same code for training using intervals or full event, 
            # let's define number of entries to use from the input data that can either
            # correspond to number of events or intervals
            slice_entries   = slice_evts
            print("slice_entries",slice_entries)
            if configs["training_configs"]["doIntervals"]:
                # Compute the number of intervals to be used from the corresponfing number of events
                slice_entries   = slice(slice_entries.start*nIntervals_perEvt, slice_entries.stop*nIntervals_perEvt)

            if split:
                slice_entries_train, slice_entries_val, slice_entries_test = get_split_slices(slice_evts,
                                                                                              configs["training_configs"]["train_split"],
                                                                                              configs["training_configs"]["doIntervals"],
                                                                                              nIntervals_perEvt
                                                                                             )
                nEvts_train += (slice_entries_train.stop - slice_entries_train.start) / nIntervals_perEvt
                nEvts_val   += (slice_entries_val.stop   - slice_entries_val.start)   / nIntervals_perEvt
                nEvts_test  += (slice_entries_test.stop  - slice_entries_test.start)  / nIntervals_perEvt

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # Now build the needed tensors depending on the model class
            # -------------------------------------------------------------------------    
            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            if configs["model_class"]=="tracks-to-KDE":                
                # -------------------------------------------------------------------------
                # Get the tracks input data (X) from the HDF5 file
                if split:
                    X_train = buildTensor(file, 
                                          "tracks", 
                                          global_configs["doIntervals"], 
                                          slice_entries_train,
                                          verbose)
                    X_val   = buildTensor(file, 
                                          "tracks", 
                                          global_configs["doIntervals"], 
                                          slice_entries_val,
                                          verbose)
                    X_test  = buildTensor(file, 
                                          "tracks", 
                                          global_configs["doIntervals"], 
                                          slice_entries_test,
                                          verbose)

                else:
                    X = buildTensor(file, 
                                    "tracks", 
                                    global_configs["doIntervals"], 
                                    slice_entries,
                                          verbose)                    
                # -------------------------------------------------------------------------
                # Get the labels (Y) corresponding to the KDE here
                if split:
                    Y_train = KDE_A_pt[slice_entries_train]
                    Y_val   = KDE_A_pt[slice_entries_val]
                    Y_test  = KDE_A_pt[slice_entries_test]
                else:
                    Y       = KDE_A_pt[slice_entries]

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            elif configs["model_class"]=="tracks-to-hist":

                # -------------------------------------------------------------------------
                # Get the tracks input data (X) from the HDF5 file
                if split:
                    X_train = buildTensor(file, 
                                          "tracks", 
                                          global_configs["doIntervals"], 
                                          slice_entries_train,
                                          verbose)
                    X_val   = buildTensor(file, 
                                          "tracks", 
                                          global_configs["doIntervals"], 
                                          slice_entries_val,
                                          verbose)
                    X_test  = buildTensor(file, 
                                          "tracks", 
                                          global_configs["doIntervals"], 
                                          slice_entries_test,
                                          verbose)

                else:
                    X = buildTensor(file, 
                                    "tracks", 
                                    global_configs["doIntervals"], 
                                    slice_entries,
                                          verbose)

                # -------------------------------------------------------------------------
                # Get the labels (Y) corresponding to the target hists from the HDF5 file
                if split:
                    Y_train = buildTensor(file, 
                                          "targetHists", 
                                          global_configs["doIntervals"], 
                                          slice_entries_train,
                                          verbose)
                    Y_val   = buildTensor(file, 
                                          "targetHists", 
                                          global_configs["doIntervals"], 
                                          slice_entries_val,
                                          verbose)
                    Y_test  = buildTensor(file, 
                                          "targetHists", 
                                          global_configs["doIntervals"], 
                                          slice_entries_test,
                                          verbose)

                else:
                    Y = buildTensor(file, 
                                    "targetHists", 
                                    global_configs["doIntervals"], 
                                    slice_entries,
                                          verbose)

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            elif configs["model_class"]=="KDE-to-hist":
                print("WARNING:::dataLoader for KDE-to-hist models not implemented yet!!!")

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            else:
                print("ERROR::: Model class (%s) not in list of available model classes!!!"%(configs["model_class"]))
                sys.exit()

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            ## Append the X and Y tensors into lsit of tensors:
            if split:
                X_list_train.append(X_train)
                X_list_val.append(X_val)
                X_list_test.append(X_test)
                Y_list_train.append(Y_train)
                Y_list_val.append(Y_val)
                Y_list_test.append(Y_test)
            else:
                X_list.append(X)
                Y_list.append(Y)
            
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Concatenate all tensors into one unique tensor
    if verbose:
        print("")
        print("="*100)
        print("Concatenating input X and labels Y tensors...")
    if split:
        X_train = torch.cat(X_list_train)
        X_val   = torch.cat(X_list_val)
        X_test  = torch.cat(X_list_test)
        Y_train = torch.cat(Y_list_train)
        Y_val   = torch.cat(Y_list_val)
        Y_test  = torch.cat(Y_list_test)
    else:
        X = torch.cat(X_list)
        Y = torch.cat(Y_list)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## Create the pytorch TensorDataset
    if split:
        if verbose:
            print("")
            print("="*100)
            print("Constructing training   TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_train))
        dataset_train = TensorDataset(X_train, Y_train)                            
        if verbose:
            print("Constructing validation TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_val))
        dataset_val   = TensorDataset(X_val,   Y_val)                            
        if verbose:
            print("Constructing test       TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_test))
        dataset_test  = TensorDataset(X_test,  Y_test)                                            
    else:
        if verbose:
            print("")
            print("="*100)
            print("Constructing TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_total))
        dataset       = TensorDataset(X, Y)                            
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## Get the batch size from the configs
    batchSize = configs["training_configs"]["batch_size"]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## By default, split the dataset in three data sets for training, validation and testing
    if split:        
        # -------------------------------------------------------------------------
        ## Transform the split datasets into three Data loader objects
        if verbose:
            print("")
            print("="*100)
            print("Constructing trainging, validation and testing DataLoaders from TensorDatasets")
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batchSize, **kargs)
        val_loader   = torch.utils.data.DataLoader(dataset=dataset_val,   batch_size=batchSize, **kargs)
        test_loader  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=batchSize, **kargs)

        # -------------------------------------------------------------------------
        ## Return the split input Data loaders
        return train_loader, val_loader, test_loader
        
    else:
        # -------------------------------------------------------------------------
        ## Create the data loader with the batch_size from configuration file
        if verbose:
            print("")
            print("="*100)
            print("Constructing DataLoader from TensorDataset")
        loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, **kargs)        
        ## Return the entire input data. This way one can use different data files for training, validation or testing  
        return loader        

## ========================================================================================
def collectData_fromTensor(configs, l_files_prefix, l_slice_evts, split=True, verbose=False, **kargs):
    """
    Method to load data from persisted pytorch tensors from Build_Torch_Tensors.py 
    
    Arguments:
    
    - configs:     configuration dictionnary
    
    - files_prefix: list of files (only the prefix to be filled with final suffix for
                    retriving the tracks, KDE and target hists tensors
        
    - split:      boolean to switch between returning either a split dataset for training, 
                  validation and testing purposes
        
    - slice_evts: slice in terms of number of events. Is automatically converted into number 
                  of intervals if using them
               
    After retriving the individual tensors, the method build a pytorch TensorDataset 
    before embedding it into a pytorch DataLoader, which is returned,
        
    """

    ## -----------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## -----------------------------------------------------------------
    ## Get the global configuration 
    global_configs = configs['global_configs'][Exp]
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # First check if the data to be collected is using intervals or not
    intervals_tag = ""
    intervals_verbose = ""
    if configs["training_configs"]["doIntervals"]:
        intervals_tag = "_inter"
        intervals_verbose = "(split in intervals)"
        
        # -------------------------------------------------------------------------
        # Before loading (or building) the data tensor, we need to compute the number of 
        # intervals corresponding to the requested number of events! 
        # Start by computing the number of intervals per event
        nBinsKDE_perEvt           = global_configs["n_bins_poca_kde"]
        nBinsPerInterval_perEvt   = global_configs["nBinsPerInterval"]
        nIntervals_perEvt         = int(nBinsKDE_perEvt/nBinsPerInterval_perEvt)
    
    
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## List of tensors for each HDF5 file
    ## Will be used to concatenate the tensors 
    ## after the loop over the list of files
    X_list = []
    Y_list = []
    
    ## In case spliting the data into train, validation and test samples
    X_list_train = []
    X_list_val   = []
    X_list_test  = []    
    Y_list_train = []
    Y_list_val   = []
    Y_list_test  = []
        
    nEvts_total = 0
    nEvts_train = 0
    nEvts_val   = 0
    nEvts_test  = 0
        
    for idx, file_prefix in enumerate(l_files_prefix):

        print("")
        print("*"*100)
        print("*"*100)
        with Timer(start="\n Collecting data from input tensors"):
            print("")
            
            slice_evts = l_slice_evts[idx]
    
            # -------------------------------------------------------------------------
            # Get the tensors location
            KDE_A_loc       = file_prefix + "_poca_KDE_A%s.pt"%(intervals_tag)
            tracksData_loc  = file_prefix + "_tracksData%s.pt"%(intervals_tag)
            targetHists_loc = file_prefix + "_targetHists%s.pt"%(intervals_tag)

            checkFileExists(KDE_A_loc,      "dataOnDisk")
            checkFileExists(tracksData_loc, "dataOnDisk")
            checkFileExists(targetHists_loc,"dataOnDisk")

            if configs["model_class"]=="tracks-to-KDE" or configs["model_class"]=="tracks-to-hist":
                print("   `--> ",tracksData_loc)
            if configs["model_class"]=="KDE-to-hist" or configs["model_class"]=="tracks-to-KDE":
                print("   `--> ",KDE_A_loc)
            if configs["model_class"]=="KDE-to-hist" or configs["model_class"]=="tracks-to-hist":
                print("   `--> ",targetHists_loc)
            print("")
            print("-"*100)
            print("")

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # Get the total number of entries (could by events or intervals) in the input 
            # stored tensor from the KDE_A tensor 
            #
            # Here since no slice argument is given, "all" entries are returned: 
            # by default, slice_evts=slice(0,9999999999)
            #
            KDE_A_pt = getTensor(KDE_A_loc, slice(0,9999999999), verbose)

            # -------------------------------------------------------------------------
            # Get the number of entries in the data file (can be either events or intervals)
            nEntries_data = KDE_A_pt.shape[0]
            # Check that the number of intervals per event is the same between the input data 
            # and the configuration file
            if configs["training_configs"]["doIntervals"]:                     
                if not nBinsPerInterval_perEvt==int(KDE_A_pt.shape[1]):
                    print("ERROR:: Number of intervals in input data (%s) different from the one defined in the configuration file (%s)"%(KDE_A_pt.shape[1],nBinsPerInterval_perEvt))
                    sys.exit()

            # -------------------------------------------------------------------------
            # Get the corresponding total number of events in the input data
            nEvts_data = nEntries_data
            # If data split in intervals, then we need to divide the number of entries 
            # by the number of intervals per event to get the number of events
            if configs["training_configs"]["doIntervals"]:
                nEvts_data = nEntries_data/nIntervals_perEvt

            # -------------------------------------------------------------------------
            # Check if the requested number of events to be used is not larger than the actual
            # total number of events in the input file
            if slice_evts.start > nEvts_data:
                print("Slice starting event (%s) larger than the maximum number of available events in the input file (%s)"%(slice_evts.start,nEvts_data))
                sys.exit()

            if slice_evts.stop > nEvts_data:
                print("WARNING:: Slice stopping event (%s) larger than the maximum number of available events in the input file (%s)"%(slice_evts.stop,nEvts_data))
                print("Slice stopping event set back to %s"%(nEvts_data))
                slice_evts = slice(slice_evts.start,nEvts_data)

            if slice_evts.stop == -1:
                slice_evts = slice(slice_evts.start,nEvts_data)
                
            nEvts_total += (slice_evts.stop - slice_evts.start)

            # -------------------------------------------------------------------------
            # Now to allow using the same code for training using intervals or full event, 
            # let's define number of entries to use from the input data that can either
            # correspond to number of events or intervals
            slice_entries   = slice_evts
            if configs["training_configs"]["doIntervals"]:
                # Compute the number of intervals to be used from the corresponfing number of events
                slice_entries   = slice(slice_entries.start*nIntervals_perEvt, slice_entries.stop*nIntervals_perEvt)

            if split:
                slice_entries_train, slice_entries_val, slice_entries_test = get_split_slices(slice_evts,
                                                                                              configs["training_configs"]["train_split"],
                                                                                              configs["training_configs"]["doIntervals"],
                                                                                              nIntervals_perEvt
                                                                                             )
                nEvts_train += (slice_entries_train.stop - slice_entries_train.start) / nIntervals_perEvt
                nEvts_val   += (slice_entries_val.stop   - slice_entries_val.start)   / nIntervals_perEvt
                nEvts_test  += (slice_entries_test.stop  - slice_entries_test.start)  / nIntervals_perEvt
            

            # -------------------------------------------------------------------------
            # Now load (or build) the needed tensors depending on the model class
            # -------------------------------------------------------------------------    

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            if configs["model_class"]=="tracks-to-KDE":                
                # -------------------------------------------------------------------------
                # Get the tracks input data (X) directly from the tensors saved on disk
                if split:
                    X_train = getTensor(tracksData_loc, slice_entries_train, verbose)
                    X_val   = getTensor(tracksData_loc, slice_entries_val, verbose)
                    X_test  = getTensor(tracksData_loc, slice_entries_test, verbose)                
                else:
                    X       = getTensor(tracksData_loc, slice_entries, verbose)               
                # -------------------------------------------------------------------------
                # Get the labels (Y) corresponding to the KDE here
                if split:
                    Y_train = KDE_A_pt[slice_entries_train]
                    Y_val   = KDE_A_pt[slice_entries_val]
                    Y_test  = KDE_A_pt[slice_entries_test]
                else:
                    Y       = KDE_A_pt[slice_entries]

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            elif configs["model_class"]=="tracks-to-hist":
                # -------------------------------------------------------------------------
                # Get the tracks input data (X)
                if split:
                    X_train = getTensor(tracksData_loc, slice_entries_train, verbose)
                    X_val   = getTensor(tracksData_loc, slice_entries_val, verbose)
                    X_test  = getTensor(tracksData_loc, slice_entries_test, verbose)                
                else:
                    X       = getTensor(tracksData_loc, slice_entries, verbose)               

                # -------------------------------------------------------------------------
                # Get the labels (Y) corresponding to the target hists
                if split:
                    Y_train = getTensor(targetHists_loc, slice_entries_train, verbose)
                    Y_val   = getTensor(targetHists_loc, slice_entries_val, verbose)
                    Y_test  = getTensor(targetHists_loc, slice_entries_test, verbose)                
                else:
                    Y       = getTensor(targetHists_loc, slice_entries, verbose)               

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            elif configs["model_class"]=="KDE-to-hist":
                print("WARNING:::dataLoader for KDE-to-hist models not implemented yet!!!")

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            else:
                print("ERROR::: Model class (%s) not in list of available model classes!!!"%(configs["model_class"]))
                sys.exit()

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            ## Append the X and Y tensors into lsit of tensors:
            if split:
                X_list_train.append(X_train)
                X_list_val.append(X_val)
                X_list_test.append(X_test)
                Y_list_train.append(Y_train)
                Y_list_val.append(Y_val)
                Y_list_test.append(Y_test)
            else:
                X_list.append(X)
                Y_list.append(Y)

        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Concatenate all tensors into one unique tensor
    if verbose:
        print("")
        print("="*100)
        print("Concatenating input X and labels Y tensors...")
    if split:
        X_train = torch.cat(X_list_train)
        X_val   = torch.cat(X_list_val)
        X_test  = torch.cat(X_list_test)
        Y_train = torch.cat(Y_list_train)
        Y_val   = torch.cat(Y_list_val)
        Y_test  = torch.cat(Y_list_test)
    else:
        X = torch.cat(X_list)
        Y = torch.cat(Y_list)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## Create the pytorch TensorDataset
    if split:
        if verbose:
            print("")
            print("="*100)
            print("Constructing training   TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_train))
        dataset_train = TensorDataset(X_train, Y_train)                            
        if verbose:
            print("Constructing validation TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_val))
        dataset_val   = TensorDataset(X_val,   Y_val)                            
        if verbose:
            print("Constructing test       TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_test))
        dataset_test  = TensorDataset(X_test,  Y_test)                                            
    else:
        if verbose:
            print("")
            print("="*100)
            print("Constructing TensorDataset from input X and labels Y tensors %s corresponding to %s events"%(intervals_verbose,nEvts_total))
        dataset       = TensorDataset(X, Y)                            
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## Get the batch size from the configs
    batchSize = configs["training_configs"]["batch_size"]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## By default, split the dataset in three data sets for training, validation and testing
    if split:        
        # -------------------------------------------------------------------------
        ## Transform the split datasets into three Data loader objects
        if verbose:
            print("")
            print("="*100)
            print("Constructing trainging, validation and testing DataLoaders from TensorDatasets")
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batchSize, **kargs)
        val_loader   = torch.utils.data.DataLoader(dataset=dataset_val,   batch_size=batchSize, **kargs)
        test_loader  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=batchSize, **kargs)

        # -------------------------------------------------------------------------
        ## Return the split input Data loaders
        return train_loader, val_loader, test_loader
        
    else:
        # -------------------------------------------------------------------------
        ## Create the data loader with the batch_size from configuration parameters file
        if verbose:
            print("")
            print("="*100)
            print("Constructing DataLoader from TensorDataset")
        loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, **kargs)        
        ## Return the entire input data. This way one can use different data files for training, validation or testing  
        return loader        

