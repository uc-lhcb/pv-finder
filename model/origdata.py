import numpy as np
from pathlib import Path
import sys
import math
from collections import namedtuple
import numba

from .utilities import Timer
from .jagged import concatenate

try:
    import uproot3 as uproot
except ModuleNotFoundError:
    import uproot
try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward

dtype_X = np.float16
dtype_Y = np.float16

# Assuming there are N events:
OutputData = namedtuple(
    "OutputData",
    (
        "X",  # Density in Z, 4000xN
        "Y",  #
        "Xmax",
        "Ymax",
        "pv_loc_x",
        "pv_loc_y",
        "pv_loc",
        "pv_ntracks",
        "pv_cat",
        "sv_loc_x",
        "sv_loc_y",
        "sv_loc",
        "sv_ntracks",
        "sv_cat",
    ),
)


def concatinate_data(outputs):
    return OutputData(
        np.concatenate([o.X for o in outputs]),
        np.concatenate([o.Y for o in outputs], 1),
        np.concatenate([o.Xmax for o in outputs]),
        np.concatenate([o.Ymax for o in outputs]),
        concatenate(o.pv_loc_x for o in outputs),
        concatenate(o.pv_loc_y for o in outputs),
        concatenate(o.pv_loc for o in outputs),
        concatenate(o.pv_ntracks for o in outputs),
        concatenate(o.pv_cat for o in outputs),
        concatenate(o.sv_loc_x for o in outputs),
        concatenate(o.sv_loc_y for o in outputs),
        concatenate(o.sv_loc for o in outputs),
        concatenate(o.sv_ntracks for o in outputs),
        concatenate(o.sv_cat for o in outputs),
    )


def save_data_hdf5(hf, od, filelist=None, compression="lzf"):
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
    akdh5["pv_cat"] = od.pv_cat
    akdh5["sv_loc_x"] = od.sv_loc_x
    akdh5["sv_loc_y"] = od.sv_loc_y
    akdh5["sv_loc"] = od.sv_loc
    akdh5["sv_ntracks"] = od.sv_ntracks
    akdh5["sv_cat"] = od.sv_cat

    return dset


@numba.vectorize(nopython=True)
def norm_cdf(mu, sigma, x):
    """
    Cumulative distribution function for the standard normal distribution.

    Much faster than scipy.stats.norm.cdf even without jit (if added). Use
    np.erf for non-vectorized version. (adds ~1 second)
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def process_root_file(filepath, sd_1=0.1):

    name = filepath.stem

    with Timer(start=f"Loading file: {name}"):
        tree = uproot.open(str(filepath))["kernel"]

        X = (tree["zdata"].array() / 2500.0).astype(dtype_X)  # Density in z, 4000xN
        Xmax = (tree["xmax"].array() / 2500.0).astype(
            dtype_X
        )  # Location of max z in x   <OPTIONAL>
        Ymax = (tree["ymax"].array() / 2500.0).astype(
            dtype_X
        )  # Location of max z in y   <OPTIONAL>
        Xmax[X == 0] = 0
        Ymax[X == 0] = 0  # The following is Truth info for training:
        pv_loc = tree["pv_loc"].array()  # z locations of each PV [#pvs]*N
        pv_loc_x = tree[
            "pv_loc_x"
        ].array()  # x                                 <OPTIONAL>
        pv_loc_y = tree[
            "pv_loc_y"
        ].array()  # y                                 <OPTIONAL>
        pv_ntrks = tree[
            "pv_ntrks"
        ].array()  # number of tracks in PV [#pvs]*N   <OPTIONAL>
        pv_cat = tree["pv_cat"].array()  # PV category (LHCb or not) [#pvs]*N
        sv_loc = tree["sv_loc"].array()  # SVs like above                    <OPTIONAL>
        sv_loc_x = tree["sv_loc_x"].array()
        sv_loc_y = tree["sv_loc_y"].array()
        sv_ntrks = tree["sv_ntrks"].array()
        sv_cat = tree["sv_cat"].array()

        pv_ntrks.content = pv_ntrks.content.astype(np.uint16)
        sv_ntrks.content = sv_ntrks.content.astype(np.uint16)

        pv_cat.content = pv_cat.content.astype(np.int8)
        sv_cat.content = sv_cat.content.astype(np.int8)

    N_vals = len(X)
    zvals_range = (-99.95, 299.95)
    Y = np.zeros([4, N_vals, 4000], dtype=dtype_Y)
    edges = np.array([-0.05, 0.05])
    bins = np.arange(-3, 4)
    mat = 0.1 * bins[np.newaxis, :] + edges[:, np.newaxis] - 99.95

    msgs = []
    with Timer(start=f"Processing events: {name}"):
        for i in range(N_vals):
            columns = (
                pv_loc[i][pv_cat[i] == 1],
                pv_loc[i][pv_cat[i] != 1],
                sv_loc[i][sv_cat[i] == 1],
                sv_loc[i][sv_cat[i] != 1],
            )
            for n, centers in enumerate(columns):
                # Centers of PVs
                centers = centers[
                    (zvals_range[0] < centers) & (centers < zvals_range[1])
                ]

                for mean in centers:
                    # Compute bin number
                    N_bin = int(np.floor((mean - zvals_range[0]) * 10))
                    values = norm_cdf(mean, sd_1, N_bin / 10 + mat)

                    try:
                        Y[n, i, bins + N_bin] += values[1] - values[0]
                    except IndexError:
                        msgs.append(
                            f"Ignored hit at bin {N_bin} at {mean:.4g} in event {i}, column {n}"
                        )

    for msg in msgs:
        print(" ", msg)

    return OutputData(
        X,
        Y,
        Xmax,
        Ymax,
        pv_loc_x,
        pv_loc_y,
        pv_loc,
        pv_ntrks,
        pv_cat,
        sv_loc_x,
        sv_loc_y,
        sv_loc,
        sv_ntrks,
        sv_cat,
    )
