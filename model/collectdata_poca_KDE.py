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

import awkward

VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n", "cat"))


def collect_truth(*files, pvs=True):
    """
    This function collects the truth information from files as
    awkward arrays (JaggedArrays). Give it the same files as collect_data.

    pvs: Collect PVs or SVs (default True: PVs)
    """

    x_list = []
    y_list = []
    z_list = []
    n_list = []
    c_list = []

    p = "p" if pvs else "s"

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as XY:
            afile = awkward.hdf5(XY)
            x_list.append(afile[f"{p}v_loc_x"])
            y_list.append(afile[f"{p}v_loc_y"])
            z_list.append(afile[f"{p}v_loc"])
            n_list.append(afile[f"{p}v_ntracks"])
            c_list.append(afile[f"{p}v_cat"])

    return VertexInfo(
        concatenate(x_list),
        concatenate(y_list),
        concatenate(z_list),
        concatenate(n_list),
        concatenate(c_list),
    )


def collect_data(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    masking=False,
    slice=None,
    load_xy=False,
    load_A_and_B=False,
    **kargs,
):
    """
    This function collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data('a.h5', 'b.h5')

    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    Xlist = []
    Ylist = []

    print("Loading data...")

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as XY:
            ## [:,np.newaxis,:] makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
            ## a is *probably* 4000 and b is *probably* N, but it could be the other
            ## way around;  check iwth .shape

## X_A is the KDE from summing probabilities
            X_A = np.asarray(XY["poca_KDE_A"])[:, np.newaxis, :].astype(dtype)
            X = X_A   ##  default is to use only the KDE from summing probabilities

## X_B is the KDE from summing probability square values; can be used to augment X_A
            X_B = np.asarray(XY["poca_KDE_B"])[:, np.newaxis, :].astype(dtype)
 
##  restore "old name" for consistency when using with old KDE data           
##            Y = np.asarray(XY["pv_target"]).astype(dtype)
            Y = np.asarray(XY["pv"]).astype(dtype)
            Y_target = Y[:,0,:]
            Y_other = Y[:,1,:]

            if load_A_and_B and (not load_xy):
                X = np.concatenate((X, X_B), axis=1)

            elif load_A_and_B and load_xy:
                ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
                ##  just as the KDE value was divided by 2500. But the range is (nominally)
                ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
                x = np.asarray(XY["poca_KDE_A_xMax"])[:, np.newaxis, :].astype(dtype)
                x = 5000.0 * x
                y = np.asarray(XY["poca_KDE_A_yMax"])[:, np.newaxis, :].astype(dtype)
                y = 5000.0 * y
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate(
                    (X, X_B, x, y), axis=1
                )  ## filling in axis with (X,X_B,x,y)

            elif load_xy and (not load_A_and_B):
                x = np.asarray(XY["poca_KDE_A_xMax"])[:, np.newaxis, :].astype(dtype)
                y = np.asarray(XY["poca_KDE_A_yMax"])[:, np.newaxis, :].astype(dtype)
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate((X, x, y), axis=1)  ## filling in axis with (X,x,y)

            if masking:
                print(Y.shape)
                # Set the result to nan if the array is above
                # threshold and the current array is below threshold
                Y_target[(Y_other > 0.01) & (Y_target < 0.01)] = dtype(np.nan)

            Xlist.append(X)
            Ylist.append(Y_target)

    X = np.concatenate(Xlist, axis=0)
    Y = np.concatenate(Ylist, axis=0)

    if slice:
        X = X[slice, :, :]
        Y = Y[slice, :]

    with Timer(start=f"Constructing {X.shape[0]} event dataset"):
        x_t = torch.tensor(X)
        y_t = torch.tensor(Y)

        if device is not None:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

        dataset = TensorDataset(x_t, y_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)
    return loader

def read_data(
    *files,
    dtype=np.float32,
    masking=False,
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
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    Xlist = []
    Ylist = []

    print("Loading data...")

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as XY:
            ## [:,np.newaxis,:] makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
            ## a is *probably* 4000 and b is *probably* N, but it could be the other
            ## way around;  check iwth .shape

## X_A is the KDE from summing probabilities
            X_A = np.asarray(XY["poca_KDE_A"])[:, np.newaxis, :].astype(dtype)

## X_B is the KDE from summing probability square values; can be used to augment X_A
            X_B = np.asarray(XY["poca_KDE_B"])[:, np.newaxis, :].astype(dtype)
##  no doubt, we will want a re-scaling here as well to get the range roughly 0 - 1
            
##            Y = np.asarray(XY["pv_target"]).astype(dtype)
            Y = np.asarray(XY["pv"]).astype(dtype)


            ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
            ##  just as the KDE value was divided by 2500. But the range is (nominally)
            ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
            poca_KDE_A_xMax = np.asarray(XY["poca_KDE_A_xMax"])[:, np.newaxis, :].astype(dtype)
            poca_KDE_A_xMax = 5000*poca_KDE_A_xMax 
            poca_KDE_A_yMax = np.asarray(XY["poca_KDE_A_yMax"])[:, np.newaxis, :].astype(dtype)
            poca_KDE_A_yMax = 5000*poca_KDE_A_yMax 

            if masking:
                # Set the result to nan if the "other" array is above
                # threshold and the current array is below threshold
                Y[(np.asarray(XY["pv_other"]) > 0.01) & (Y < 0.01)] = dtype(np.nan)

##  let's get the other data stored in the files

## the numpy array "kernel" has the original kernel (perhaps scaled down by 100)
            kernel = np.asarray(XY["kernel"])[:, np.newaxis, :].astype(dtype)
    return X_A, X_B, kernel, poca_KDE_A_xMax, poca_KDE_A_yMax


def collect_data_poca(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    masking=False,
    slice=None,
    load_xy=False,
    load_A_and_B=False,
    load_XandXsq=False,
    **kargs,
):
    """
    This function collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data_poca('a.h5', 'b.h5')
    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    Xlist = []
    Ylist = []

    print("Loading data...")

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as XY:
            ## [:,np.newaxis,:] makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
            ## a is *probably* 4000 and b is *probably* N, but it could be the other
            ## way around;  check iwth .shape

## X_A is the KDE from summing probabilities
            X_A = np.asarray([list(XY["POCAzdata"][f'Event{i}']) 
                              for i in range(len(XY["POCAzdata"]))])[:,np.newaxis,:].astype(dtype)
            X = X_A   ##  default is to use only the KDE from summing probabilities
            Xsq = X ** 2  ## this simply squares each element of X

## X_B is the KDE from summing probability square values; can be used to augment X_A
            X_B = np.asarray([list(XY["POCA_sqzdata"][f'Event{i}']) 
                              for i in range(len(XY["POCA_sqzdata"]))])[:,np.newaxis,:].astype(dtype)
 
            Y = np.asarray([list(XY["pv"][f'Event{i}'])[0] for i in range(len(XY["pv"]))]).astype(dtype)
            Y_other = np.asarray([list(XY["pv"][f'Event{i}'])[1] for i in range(len(XY["pv"]))]).astype(dtype)


##  if we want to treat new KDE as input for old KDE infrerence engine, use
##  load_XandXsq
##  we will not want to use this moving forward, but it is necessary for
##  testing with some old inference engines
            if load_XandXsq and (not load_xy):
                X = np.concatenate((X, Xsq), axis=1)

            elif load_XandXsq and load_xy:
                ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
                ##  just as the KDE value was divided by 2500. But the range is (nominally)
                ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
                x = np.asarray([list(XY["POCAxmax"][f'Event{i}']) for i in 
                                range(len(XY["POCAxmax"]))])[:, np.newaxis, :].astype(dtype)
                x = 5000.0 * x
                y = np.asarray([list(XY["POCAymax"][f'Event{i}']) for i in 
                                range(len(XY["POCAymax"]))])[:, np.newaxis, :].astype(dtype)
                y = 5000.0 * y
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate(
                    (X, Xsq, x, y), axis=1
                )  ## filling in axis with (X,Xsq,x,y)

##  end of treating new KDE and input for old algs
            if load_A_and_B and (not load_xy):
                X = np.concatenate((X, X_B), axis=1)

            elif load_A_and_B and load_xy:
                ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
                ##  just as the KDE value was divided by 2500. But the range is (nominally)
                ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
                x = np.asarray([list(XY["POCAxmax"][f'Event{i}']) for i in 
                                range(len(XY["POCAxmax"]))])[:, np.newaxis, :].astype(dtype)
                x = 5000.0 * x
                y = np.asarray([list(XY["POCAymax"][f'Event{i}']) for i in 
                                range(len(XY["POCAymax"]))])[:, np.newaxis, :].astype(dtype)
                y = 5000.0 * y
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate(
                    (X, X_B, x, y), axis=1
                )  ## filling in axis with (X,X_B,x,y)

            elif load_xy and (not load_A_and_B) and (not load_XandXsq):
                x = np.asarray([list(XY["POCAxmax"][f'Event{i}']) for i in 
                                range(len(XY["POCAxmax"]))])[:, np.newaxis, :].astype(dtype)
                np.asarray([list(XY["POCAymax"][f'Event{i}']) for i in 
                                range(len(XY["POCAymax"]))])[:, np.newaxis, :].astype(dtype)
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate((X, x, y), axis=1)  ## filling in axis with (X,x,y)

            if masking:
                # Set the result to nan if the "other" array is above
                # threshold and the current array is below threshold
                Y[(Y_other > 0.01) & (Y < 0.01)] = dtype(np.nan)

            Xlist.append(X)
            Ylist.append(Y)

    X = np.concatenate(Xlist, axis=0)
    Y = np.concatenate(Ylist, axis=0)

    if slice:
        X = X[slice, :, :]
        Y = Y[slice, :]

    with Timer(start=f"Constructing {X.shape[0]} event dataset"):
        x_t = torch.tensor(X)
        y_t = torch.tensor(Y)

        if device is not None:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

        dataset = TensorDataset(x_t, y_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)
    return loader


def collect_data_poca_ATLAS(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    masking=False,
    slice=None,
    load_xy=False,
    load_A_and_B=False,
    load_XandXsq=False,
    **kargs,
):
    """
    This function collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data_poca('a.h5', 'b.h5')
    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    Xlist = []
    Ylist = []

    print("Loading data...")

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as XY:
            ## [:,np.newaxis,:] makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
            ## a is *probably* 4000 and b is *probably* N, but it could be the other
            ## way around;  check iwth .shape

## X_A is the KDE from summing probabilities
            X_A = np.asarray([list(XY["poca_KDE_A_zdata"][f'Event{i}']) 
                              for i in range(len(XY["poca_KDE_A_zdata"]))])[:,np.newaxis,:].astype(dtype)
            X = X_A   ##  default is to use only the KDE from summing probabilities
            Xsq = X ** 2  ## this simply squares each element of X

## X_B is the KDE from summing probability square values; can be used to augment X_A
            X_B = np.asarray([list(XY["poca_KDE_B_zdata"][f'Event{i}']) 
                              for i in range(len(XY["poca_KDE_B_zdata"]))])[:,np.newaxis,:].astype(dtype)
 
            Y = np.asarray([list(XY["Target_Y"][f'Event{i}'])[0] for i in range(len(XY["Target_Y"]))]).astype(dtype)
            Y_other = np.asarray([list(XY["Target_Y"][f'Event{i}'])[1] for i in range(len(XY["Target_Y"]))]).astype(dtype)

            #Y = np.asarray([list(XY["Target_Y"][f'Event{i}']) for i in range(len(XY["Target_Y"]))]).astype(dtype)


##  if we want to treat new KDE as input for old KDE infrerence engine, use
##  load_XandXsq
##  we will not want to use this moving forward, but it is necessary for
##  testing with some old inference engines
            if load_XandXsq and (not load_xy):
                X = np.concatenate((X, Xsq), axis=1)

            elif load_XandXsq and load_xy:
                ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
                ##  just as the KDE value was divided by 2500. But the range is (nominally)
                ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
                x = np.asarray([list(XY["poca_KDE_A_xmax"][f'Event{i}']) for i in 
                                range(len(XY["poca_KDE_A_xmax"]))])[:, np.newaxis, :].astype(dtype)
                #x = 5000.0 * x
                y = np.asarray([list(XY["poca_KDE_A_ymax"][f'Event{i}']) for i in 
                                range(len(XY["poca_KDE_A_ymax"]))])[:, np.newaxis, :].astype(dtype)
                #y = 5000.0 * y
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate(
                    (X, Xsq, x, y), axis=1
                )  ## filling in axis with (X,Xsq,x,y)

##  end of treating new KDE and input for old algs
            if load_A_and_B and (not load_xy):
                X = np.concatenate((X, X_B), axis=1)

            elif load_A_and_B and load_xy:
                ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
                ##  just as the KDE value was divided by 2500. But the range is (nominally)
                ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
                x = np.asarray([list(XY["poca_KDE_A_xmax"][f'Event{i}']) for i in 
                                range(len(XY["poca_KDE_A_xmax"]))])[:, np.newaxis, :].astype(dtype)
                #x = 5000.0 * x
                y = np.asarray([list(XY["poca_KDE_A_ymax"][f'Event{i}']) for i in 
                                range(len(XY["poca_KDE_A_ymax"]))])[:, np.newaxis, :].astype(dtype)
                #y = 5000.0 * y
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate(
                    (X, X_B, x, y), axis=1
                )  ## filling in axis with (X,X_B,x,y)

            elif load_xy and (not load_A_and_B) and (not load_XandXsq):
                x = np.asarray([list(XY["poca_KDE_A_xmax"][f'Event{i}']) for i in 
                                range(len(XY["poca_KDE_A_xmax"]))])[:, np.newaxis, :].astype(dtype)
                np.asarray([list(XY["poca_KDE_A_ymax"][f'Event{i}']) for i in 
                                range(len(XY["poca_KDE_A_ymax"]))])[:, np.newaxis, :].astype(dtype)
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate((X, x, y), axis=1)  ## filling in axis with (X,x,y)

            if masking:
#                 Set the result to nan if the "other" array is above
#                 threshold and the current array is below threshold
                Y[(Y_other > 0.01) & (Y < 0.01)] = dtype(np.nan)

            Xlist.append(X)
            Ylist.append(Y)

    X = np.concatenate(Xlist, axis=0)
    Y = np.concatenate(Ylist, axis=0)

    if slice:
        X = X[slice, :, :]
        Y = Y[slice, :]

    with Timer(start=f"Constructing {X.shape[0]} event dataset"):
        x_t = torch.tensor(X)
        y_t = torch.tensor(Y)

        if device is not None:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

        dataset = TensorDataset(x_t, y_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)
    return loader