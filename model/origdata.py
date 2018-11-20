import uproot
import numpy as np
from pathlib import Path
import sys
import math
from collections import namedtuple

from .utilities import Timer
from .jagged import concatenate
import awkward

# Bug in version 0.4.1:
awkward.persist.compression[0]['types'] = tuple(awkward.persist.compression[0]['types'])

dtype_X = np.float16
dtype_Y = np.float16

OutputData = namedtuple(
    "OutputData", (
        "X", "Y", "Xmax", "Ymax",
        "pv_loc_x", "pv_loc_y", "pv_loc", "pv_ntracks",
        "sv_loc_x", "sv_loc_y", "sv_loc", "sv_ntracks"))

def concatinate_data(outputs):
    return OutputData(
        X = np.concatenate([o.X for o in outputs]),
        Y = np.concatenate([o.Y for o in outputs], 1),
        Xmax = np.concatenate([o.Xmax for o in outputs]),
        Ymax = np.concatenate([o.Ymax for o in outputs]),
        pv_loc_x = concatenate(o.pv_loc_x for o in outputs),
        pv_loc_y = concatenate(o.pv_loc_y for o in outputs),
        pv_loc = concatenate(o.pv_loc for o in outputs),
        pv_ntracks = concatenate(o.pv_ntracks for o in outputs),
        sv_loc_x = concatenate(o.sv_loc_x for o in outputs),
        sv_loc_y = concatenate(o.sv_loc_y for o in outputs),
        sv_loc = concatenate(o.sv_loc for o in outputs),
        sv_ntracks = concatenate(o.sv_ntracks for o in outputs)
    )

def save_data_hdf5(hf, od, filelist=None, compression='lzf'):
    dset = hf.create_dataset("kernel", data=od.X, compression=compression)
    if filelist:
        dset.attrs["files"] = np.string_(",".join(str(s.stem) for s in filelist))
    
    hf.create_dataset("pv", data=od.Y[0], compression=compression)
    hf.create_dataset("sv", data=od.Y[2], compression=compression)
    hf.create_dataset("pv_other", data=od.Y[1], compression=compression)
    hf.create_dataset("sv_other", data=od.Y[3], compression=compression)
    hf.create_dataset("Xmax", data=od.Xmax, compression=compression)
    hf.create_dataset("Ymax", data=od.Ymax, compression=compression)
    
    akdh5 = awkward.hdf5(hf)
    akdh5["pv_loc_x"] = od.pv_loc_x
    akdh5["pv_loc_y"] = od.pv_loc_y
    akdh5["pv_loc"] = od.pv_loc
    akdh5["pv_ntracks"] = od.pv_ntracks
    akdh5["sv_loc_x"] = od.sv_loc_x
    akdh5["sv_loc_y"] = od.sv_loc_y
    akdh5["sv_loc"] = od.sv_loc
    akdh5["sv_ntracks"] = od.sv_ntracks
    
    return dset

def norm_cdf(mu, sigma, x):
    '''
    Cumulative distribution function for the standard normal distribution.

    Much faster than scipy.stats.norm.cdf even without jit (if added).
    '''
    return 0.5 * (1 + np.erf((x-mu) / (sigma * math.sqrt(2.0))))

def process_root_file(filepath, sd_1 = 0.1):
    
    name = filepath.stem
    
    with Timer(start = f'Loading file: {name}'):
        tree = uproot.open(str(filepath))['kernel']

        X = (tree['zdata'].array() / 2500.).astype(dtype_X)
        Xmax = (tree['xmax'].array() / 2500.).astype(dtype_X)
        Ymax = (tree['ymax'].array() / 2500.).astype(dtype_X)
        pv_loc = tree['pv_loc'].array()
        pv_loc_x = tree['pv_loc_x'].array()
        pv_loc_y = tree['pv_loc_y'].array()
        pv_cat = tree['pv_cat'].array()
        pv_ntrks = tree['pv_ntrks'].array()
        sv_loc = tree['sv_loc'].array()
        sv_loc_x = tree['sv_loc_x'].array()
        sv_loc_y = tree['sv_loc_y'].array()
        sv_cat = tree['sv_cat'].array()
        sv_ntrks = tree['sv_ntrks'].array()

    N_vals = len(X)
    zvals_range = (-99.95, 299.95)
    Y = np.zeros([4, N_vals, 4000], dtype=dtype_Y)
    edges = np.array([-0.05, 0.05])
    bins = np.arange(-3, 4)
    mat = 0.1*bins[np.newaxis,:] + edges[:,np.newaxis] - 99.95
    
    msgs = []
    with Timer(start = f'Processing events: {name}'):
        for i in range(N_vals):
            columns = (
                pv_loc[i][pv_cat[i]==1],
                pv_loc[i][pv_cat[i]!=1],
                sv_loc[i][sv_cat[i]==1],
                sv_loc[i][sv_cat[i]!=1]
            )
            for n, centers in enumerate(columns):
                # Centers of PVs
                centers = centers[(zvals_range[0] < centers) & (centers < zvals_range[1])]

                for mean in centers:
                    # Compute bin number
                    N_bin = int(np.floor((mean - zvals_range[0])*10))
                    values = norm_cdf(mean, sd_1, N_bin/10 + mat)

                    try:
                        Y[n, i, bins + N_bin] += values[1] - values[0]
                    except IndexError:
                        msgs.append(f"Ignored hit at bin {N_bin} at {mean:.4g} in event {i}, column {n}")
                        
    for msg in msgs:
        print(" ", msg)
                
    return OutputData(
        X, Y, Xmax, Ymax,
        pv_loc_x, pv_loc_y, pv_loc, pv_ntrks,
        sv_loc_x, sv_loc_y, sv_loc, sv_ntrks)