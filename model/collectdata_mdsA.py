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
    load_XandXsq=False,
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
            X = np.asarray(XY["kernel"])[:, np.newaxis, :].astype(dtype)
            Xsq = X ** 2  ## this simply squares each element of X

            Y = np.asarray(XY["pv"]).astype(dtype)

            if load_XandXsq and (not load_xy):
                X = np.concatenate((X, Xsq), axis=1)

            elif load_XandXsq and load_xy:
                ##  the code which wrote the files divided the Xmax and Ymax values by 2500,
                ##  just as the KDE value was divided by 2500. But the range is (nominally)
                ##  -0.4 - 0.4.  So multiply by 5000 so the feature range is ~ -1 to +1
                x = np.asarray(XY["Xmax"])[:, np.newaxis, :].astype(dtype)
                x = 5000.0 * x
                y = np.asarray(XY["Ymax"])[:, np.newaxis, :].astype(dtype)
                y = 5000.0 * y
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate(
                    (X, Xsq, x, y), axis=1
                )  ## filling in axis with (X,Xsq,x,y)

            elif load_xy and (not load_XandXsq):
                x = np.asarray(XY["Xmax"])[:, np.newaxis, :].astype(dtype)
                y = np.asarray(XY["Ymax"])[:, np.newaxis, :].astype(dtype)
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate((X, x, y), axis=1)  ## filling in axis with (X,x,y)

            if masking:
                # Set the result to nan if the "other" array is above
                # threshold and the current array is below threshold
                Y[(np.asarray(XY["pv_other"]) > 0.01) & (Y < 0.01)] = dtype(np.nan)

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
