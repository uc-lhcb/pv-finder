##
## mds July 19,2020
##
## basic structure derived from collectdata_mdsA.py
## but adapted so that X will return a tensor built from
## the awkward arrays of track parameters
## to be used as the feature set for algorithm and Y will return the 
## KDE and associate Xmax and Ymax values to be used in the cost function.
##
##  building X requires "padding" the awkward array content so it
##  fits into a well-defined tensor structure.
##

import torch
from torch.utils.data import TensorDataset

import numpy as np
from pathlib import Path
from functools import partial
import warnings
from collections import namedtuple

from .utilities import Timer
from .jagged import concatenate


# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward

ja = awkward.JaggedArray

dtype_X = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU
dtype_Y = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU


def collect_t2kde_data(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    slice=None,
    **kargs,
):
    """
    This function collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data('a.h5', 'b.h5')

    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    Xlist = []
    Ylist = []

    print("Loading data...")

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as f:
            ## [:,np.newaxis,:] makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
            ## a is *probably* 4000 and b is *probably* N, but it could be the other
            ## way around;  check iwth .shape

## Here we read in the KDE itself plus the values of x and y where the KDE is maximal for 
## each bin of z. It appears that in the test file the original KDE values .AND. the values 
## of Xmax and Ymax have been divided by 2500. This should have been done only for the 
## KDE values, so Xmax and Ymax are re-scaled to better use the dynamic range available 
## using np.float16

            
            kernel = np.asarray(f["kernel"])
            Xmax = 2500.*np.asarray(f["Xmax"])
            Ymax = 2500.*np.asarray(f["Ymax"]) 
            
            Y = ja.concatenate((kernel,Xmax,Ymax),axis=1).astype(dtype_Y)
            
## now build the feature set from the relevant tracks' parameters
## we need to usse "afile" to account for the variable length
## structure of the awkward arrays
        
            afile = awkward.hdf5(f)
            
            pocaz = np.asarray(0.001*afile["recon_pocaz"].astype(dtype_Y))
            pocax = np.asarray(afile["recon_pocax"].astype(dtype_Y))
            pocay = np.asarray(afile["recon_pocay"].astype(dtype_Y))
            pocaTx = np.asarray(afile["recon_tx"].astype(dtype_Y))
            pocaTy = np.asarray(afile["recon_ty"].astype(dtype_Y))
            pocaSigmapocaxy = np.asarray(afile["recon_sigmapocaxy"].astype(dtype_Y))
            nEvts = len(pocaz)

## mds for testing only            for i in range(nEvts-1):
## mds for testing only                maxLen = max(maxLen,len(pocaz[i]))
## mds for testing only            print("maxLen = ",maxLen)
            

##  mark non-track data with -99 as a flag
            maxLen = 600 ## for safety:  600 >> 481, which is what was seen for 100 evts
            padded_pocaz = np.zeros((nEvts,maxLen))-99.
            padded_pocax = np.zeros((nEvts,maxLen))-99.
            padded_pocay = np.zeros((nEvts,maxLen))-99.
            padded_tx    = np.zeros((nEvts,maxLen))-99.
            padded_ty    = np.zeros((nEvts,maxLen))-99.
            padded_sigma = np.zeros((nEvts,maxLen))-99.

            for i, e in enumerate(pocaz):
                fillingLength = min(len(e),maxLen)
                padded_pocaz[i,:fillingLength] = pocaz[i][:fillingLength].astype(dtype_Y)
                padded_pocax[i,:fillingLength] = pocax[i][:fillingLength].astype(dtype_Y)
                padded_pocay[i,:fillingLength] = pocay[i][:fillingLength].astype(dtype_Y)
                padded_tx[i,:fillingLength] = pocaTx[i][:fillingLength].astype(dtype_Y)
                padded_ty[i,:fillingLength] = pocaTy[i][:fillingLength].astype(dtype_Y)
                padded_sigma[i,:fillingLength] = pocaSigmapocaxy[i][:fillingLength].astype(dtype_Y)

            padded_pocaz = padded_pocaz[:,np.newaxis,:]
            padded_pocax = padded_pocax[:,np.newaxis,:]
            padded_pocay = padded_pocay[:,np.newaxis,:]
            padded_tx = padded_tx[:,np.newaxis,:]
            padded_ty = padded_ty[:,np.newaxis,:]
            padded_sigma = padded_sigma[:,np.newaxis,:]

            X = ja.concatenate((padded_pocaz,padded_pocax,padded_pocay,padded_tx,padded_ty,padded_sigma),axis=1).astype(dtype_X)

## mds            print("X = ",X)
            print("len(X) = ",len(X))
            Xlist.append(X)
            Ylist.append(Y)
            print("len(Xlist) = ",len(Xlist))
    X = np.concatenate(Xlist, axis=0)
    Y = np.concatenate(Ylist, axis=0)
    print("outer loop X.shape = ", X.shape)

    if slice:
        X = X[slice, :]
        Y = Y[slice, :]

    with Timer(start=f"Constructing {X.shape[0]} event dataset"):
        x_t = torch.tensor(X)
        y_t = torch.tensor(Y)

        if device is not None:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

        dataset = TensorDataset(x_t, y_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)
    print("x_t.shape = ",x_t.shape)
    print("x_t.shape[0] = ", x_t.shape[0])
    print("x_t.shape[1] = ", x_t.shape[1])
    nFeatures = 6
    x_t.view(x_t.shape[0],nFeatures,-1)
    print("x_t.shape = ",x_t.shape)
    
    
    return loader
