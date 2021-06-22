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

VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n", "cat", "key"))


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
    key_list = [] # added line to store PV keys

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
            key_list.append(afile[f"pv_key"])

    return VertexInfo(
        concatenate(x_list),
        concatenate(y_list),
        concatenate(z_list),
        concatenate(n_list),
        concatenate(c_list),
        concatenate(key_list),
    )


def collect_data(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    masking=False,
    slice=None,
    load_xy=False,
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
            X = np.asarray(XY["kernel"])[:, np.newaxis, :].astype(dtype)
            Y = np.asarray(XY["pv"]).astype(dtype)

            if load_xy:
                x = np.asarray(XY["Xmax"])[:, np.newaxis, :].astype(dtype)
                y = np.asarray(XY["Ymax"])[:, np.newaxis, :].astype(dtype)
                x[X == 0] = 0
                y[X == 0] = 0
                X = np.concatenate((X, x, y), axis=1)

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
