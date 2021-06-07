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
        "sv_cat"
    ),
)

VerboseOutputData = namedtuple("VerboseOutputData",OutputData._fields+(
    "recon_x",
    "recon_y",
    "recon_z",
    "recon_tx",
    "recon_ty",
    "recon_pocax",
    "recon_pocay",
    "recon_pocaz",
    "recon_sigmapocaxy"))


def concatenate_data(outputs, verbose_tracking=False):
    od = OutputData(
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
        concatenate(o.sv_cat for o in outputs))
    if not verbose_tracking:
        return od
    else :
      return VerboseOutputData(*(od),
        concatenate(o.recon_x for o in outputs),
        concatenate(o.recon_y for o in outputs),
        concatenate(o.recon_z for o in outputs),
        concatenate(o.recon_tx for o in outputs),
        concatenate(o.recon_ty for o in outputs),
        concatenate(o.recon_pocax for o in outputs),
        concatenate(o.recon_pocay for o in outputs),
        concatenate(o.recon_pocaz for o in outputs),
        concatenate(o.recon_sigmapocaxy for o in outputs),
    )


def save_data_hdf5(hf, od, filelist=None, compression="lzf", verbose_tracking=False):
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
    if verbose_tracking:
        akdh5["recon_x"] = od.recon_x
        akdh5["recon_y"] = od.recon_y
        akdh5["recon_z"] = od.recon_z
        akdh5["recon_tx"] = od.recon_tx
        akdh5["recon_ty"] = od.recon_ty
        akdh5["recon_pocax"] = od.recon_pocax
        akdh5["recon_pocay"] = od.recon_pocay
        akdh5["recon_pocaz"] = od.recon_pocaz
        akdh5["recon_sigmapocaxy"] = od.recon_sigmapocaxy

    return dset


@numba.vectorize(nopython=True)
def norm_cdf(mu, sigma, x):
    """
    Cumulative distribution function for the standard normal distribution.

    Much faster than scipy.stats.norm.cdf even without jit (if added). Use
    np.erf for non-vectorized version. (adds ~1 second)
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def process_root_file(filepath, sd_1=0.1, verbose_tracking=False):

    name = filepath.stem
    ##  take the following  constants used in calculating pvRes from LHCb-PUB-2017-005
    A_res = 926.0
    B_res = 0.84
    C_res = 10.7
    mds_counter = 0

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
        if(verbose_tracking):
            recon_x = tree["recon_x"].array()
            recon_y = tree["recon_y"].array()
            recon_z = tree["recon_z"].array()
            recon_tx = tree["recon_tx"].array()
            recon_ty = tree["recon_ty"].array()
            recon_pocax = tree["recon_pocax"].array()
            recon_pocay = tree["recon_pocay"].array()
            recon_pocaz = tree["recon_pocaz"].array()
            recon_sigmapocaxy = tree["recon_sigmapocaxy"].array()

        pv_ntrks.content = pv_ntrks.content.astype(np.uint16)
        sv_ntrks.content = sv_ntrks.content.astype(np.uint16)

        pv_cat.content = pv_cat.content.astype(np.int8)
        sv_cat.content = sv_cat.content.astype(np.int8)

    N_vals = len(X)
    zvals_range = (-99.95, 299.95)
    Y = np.zeros([4, N_vals, 4000], dtype=dtype_Y)
    edges = np.array([-0.05, 0.05])
    ##    bins = np.arange(-3, 4)
    bins = np.arange(-5, 6)
    mat = 0.1 * bins[np.newaxis, :] + edges[:, np.newaxis] - 99.95

    msgs = []
    with Timer(start=f"Processing events: {name}"):
        for i in range(N_vals):
            columns = (
                (pv_loc[i][pv_cat[i] == 1], pv_ntrks[i][pv_cat[i] == 1]),
                (pv_loc[i][pv_cat[i] != 1], pv_ntrks[i][pv_cat[i] != 1]),
                (sv_loc[i][sv_cat[i] == 1], sv_ntrks[i][sv_cat[i] == 1]),
                (sv_loc[i][sv_cat[i] != 1], sv_ntrks[i][sv_cat[i] != 1]),
            )

            if mds_counter < 10:

                print(" \n \n N_vals = ", N_vals)

                print("len(columns) = ", len(columns))
                print("columns = \n", columns)
                print("columns[0][0] = ", columns[0][0])
                print("columns[0][1] = ", columns[0][1])
                print("columns[1][0] = ", columns[1][0])
                print("columns[1][1] = ", columns[1][1])
                print("columns[2][0] = ", columns[2][0])
                print("columns[2][1] = ", columns[2][1])
                print("columns[3][0] = ", columns[3][0])
                print("columns[3][1] = ", columns[3][1])
            mds_counter += 1
            ##            assert(mds_counter)<10

            for n, elements in enumerate(columns):

                ##  addition 190810 mds
                ##            nTrks = np.where(np.isnan(nTrks), 0, nTrks)
                ## mds            pvRes = A_res*np.power(nTrks, -1*B_res) + C_res   # values taken from LHCb-PUB-2017-005
                ## mds            sd = 0.001*pvRes  ## convert from microns (in TDR) to  mm (units used here)

                entries = len(elements[0])
                if mds_counter < 10:
                    print(" \n \n \n elements =  ", elements)
                    print(" entries = ", entries)
                for v_index in range(entries):
                    centers = elements[0][v_index]
                    nTrks = elements[1][v_index]
                    sd = np.where(
                        nTrks < 4,
                        sd_1,
                        0.001 * (A_res * np.power(nTrks, -1 * B_res) + C_res),
                    )

                    test = (zvals_range[0] < centers) & (centers < zvals_range[1])

                    centers = centers[test]
                    nTrks = nTrks[test]
                    sd = sd[test]

                    if mds_counter < 10:
                        print("\n \n n = ", n)
                        print(" \n centers = ", centers)
                        print(" nTrks  =       ", nTrks)
                        print(" sd     =       ", sd)

                    for mean, ntrk, pv_res in zip(centers, nTrks, sd):
                        if mds_counter < 10:
                            print(" \n iterating  using zip(centers, nTrks, sd)  ")
                            print(" \n  mean = ", mean)
                            print(" ntrk   = ", ntrk)
                            print(" pv_res = ", pv_res)

                        # Compute bin number
                        N_bin = int(np.floor((mean - zvals_range[0]) * 10))
                        values = norm_cdf(mean, pv_res, N_bin / 10 + mat)
                        ##  replace sd_1 with sd computed from nTrks
                        ##  and increase area to be inversely proportional to sd for sd <150 microns
                        ##  (which is roughly the resolution expected for nTrks ~ 10) and with
                        ##  with unit area at sd = 0.150 mm = 150 micron, and larger values (lower nTrk)
                        ##  altogether, this will make the target peaks more narrow and the areas larger as the
                        ##  number of tracks increases.
                        if mds_counter < 10:
                            print("N_bin = ", N_bin)
                            print("mat = ", mat)
                            print("  values = ", values)

                        populate = values[1] - values[0]
                        populate = np.where(
                            (0.15 / pv_res) > 1, (0.15 / pv_res) * populate, populate
                        )
                        if mds_counter < 10:
                            print(" populate = ", populate)

                        try:
                            ## mds                    Y[n, i, bins + N_bin] += values[1] - values[0]
                            Y[n, i, bins + N_bin] += populate
                        except IndexError:
                            msgs.append(
                                f"Ignored hit at bin {N_bin} at {mean:.4g} in event {i}, column {n}"
                            )

    for msg in msgs:
        print(" ", msg)

    od = OutputData(
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
        sv_cat
    )
    if not verbose_tracking:
        return od
    else:
        return VerboseOutputData(*(od),
            recon_x,
            recon_y,
            recon_z,
            recon_tx,
            recon_ty,
            recon_pocax,
            recon_pocay,
            recon_pocaz,
            recon_sigmapocaxy)

