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


# below added by ekauffma:

def collect_poca(*files):
    
    #initialize lists
    pocax_list = []
    pocay_list = []
    pocaz_list = []

    majoraxisx_list = []
    majoraxisy_list = []
    majoraxisz_list = []

    minoraxis1x_list = []
    minoraxis1y_list = []
    minoraxis1z_list = []
    minoraxis2x_list = []
    minoraxis2y_list = []
    minoraxis2z_list = []

    match_list = []
    
    #iterate through all files
    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with h5py.File(XY_file, mode="r") as XY:

            #print keys in current hdf5 file
            print(XY.keys())

            afile = awkward.hdf5(XY)

            #append to appropriate lists
            pocax_list.append(afile["poca_x"])
            pocay_list.append(afile["poca_y"])
            pocaz_list.append(afile["poca_z"])

            majoraxisx_list.append(afile["major_axis_x"])
            majoraxisy_list.append(afile["major_axis_y"])
            majoraxisz_list.append(afile["major_axis_z"])

            minoraxis1x_list.append(afile["minor_axis1_x"])
            minoraxis1y_list.append(afile["minor_axis1_y"])
            minoraxis1z_list.append(afile["minor_axis1_z"])

            minoraxis2x_list.append(afile["minor_axis2_x"])
            minoraxis2y_list.append(afile["minor_axis2_y"])
            minoraxis2z_list.append(afile["minor_axis2_z"])
            
            match_list.append(afile["recon_pv_key"])
    
    #construct pocas dictionary
    pocas = {}
    pocas["x"] = {"poca": concatenate(pocax_list),
                  "major_axis": concatenate(majoraxisx_list),
                  "minor_axis1": concatenate(minoraxis1x_list),
                  "minor_axis2": concatenate(minoraxis2x_list)}

    pocas["y"] = {"poca": concatenate(pocay_list),
                  "major_axis": concatenate(majoraxisy_list),
                  "minor_axis1": concatenate(minoraxis1y_list),
                  "minor_axis2": concatenate(minoraxis2y_list)}

    pocas["z"] = {"poca": concatenate(pocaz_list),
                  "major_axis": concatenate(majoraxisz_list),
                  "minor_axis1": concatenate(minoraxis1z_list),
                  "minor_axis2": concatenate(minoraxis2z_list)}

    return pocas, concatenate(match_list)