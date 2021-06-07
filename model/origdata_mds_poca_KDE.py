##  derived from origdata_mdsB.py
##
##  200922 mds -- Marian's new .root files include the original KDEs
##  and two poca KDEs, one summing the probabilities and another the probability^2 values.
##  He has also scaled the original KDEs by a 100 (divided them by 100) so they
##  are typically the same magnitude as the new poca KDEs.
##  [the probabilities are calculated using exp(0.5*chisq) where we probably 
##  need to additionally account for sqrt{determinant of the inverse covariance
##  matrix} and (perhaps) the "usual" (2*pi)^{=3/2}.]
##
##  In addition to the new KDEs, the .root files include poca ellipsoid information
##  that should be passed along to the .hf5 files


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
        "recon_x",
        "recon_y",
        "recon_z",
        "recon_tx",
        "recon_ty",
## mds        "recon_pocax",
## mds        "recon_pocay",
## mds        "recon_pocaz",
## mds        "recon_sigmapocaxy",
        "poca_x",		## poca ellipsoid center, x coordinate
	"poca_y",		## poca ellipsoid center, y coordinate
	"poca_z",		## poca ellipsoid center, z coordinate
	"major_axis_x",		## for poca ellipsoid
	"major_axis_y",		## for poca ellipsoid
	"major_axis_z",		## for poca ellipsoid
	"minor_axis1_x",	## for poca ellipsoid
	"minor_axis1_y",	## for poca ellipsoid
	"minor_axis1_z",	## for poca ellipsoid
	"minor_axis2_x",	## for poca ellipsoid
	"minor_axis2_y",	## for poca ellipsoid
	"minor_axis2_z",	## for poca ellipsoid
        "poca_KDE_A",		## KDE calculated from summing probabilities
	"poca_KDE_A_xMax",	## x value where poca_KDE_A was found
	"poca_KDE_A_yMax",	## y value where poca_KDE_A was found
        "poca_KDE_B",		## KDE calculated from summing probability square values
	"poca_KDE_B_xMax",	## x value where poca_KDE_B was found
	"poca_KDE_B_yMax",	## y value where poca_KDE_B was found
    ),
)


def concatenate_data(outputs):
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
        concatenate(o.recon_x for o in outputs),
        concatenate(o.recon_y for o in outputs),
        concatenate(o.recon_z for o in outputs),
        concatenate(o.recon_tx for o in outputs),
        concatenate(o.recon_ty for o in outputs),
## mds        concatenate(o.recon_pocax for o in outputs),
## mds        concatenate(o.recon_pocay for o in outputs),
## mds        concatenate(o.recon_pocaz for o in outputs),
## mds        concatenate(o.recon_sigmapocaxy for o in outputs),
##
##  the following 18 lines added 200922
        concatenate(o.poca_x for o in outputs),
        concatenate(o.poca_y for o in outputs),
        concatenate(o.poca_z for o in outputs),
        concatenate(o.major_axis_x for o in outputs),
        concatenate(o.major_axis_y for o in outputs),
        concatenate(o.major_axis_z for o in outputs),
        concatenate(o.minor_axis1_x for o in outputs),
        concatenate(o.minor_axis1_y for o in outputs),
        concatenate(o.minor_axis1_z for o in outputs),
        concatenate(o.minor_axis2_x for o in outputs),
        concatenate(o.minor_axis2_y for o in outputs),
        concatenate(o.minor_axis2_z for o in outputs),
        np.concatenate([o.poca_KDE_A for o in outputs]),
        np.concatenate([o.poca_KDE_A_xMax for o in outputs]),
        np.concatenate([o.poca_KDE_A_yMax for o in outputs]),
        np.concatenate([o.poca_KDE_B for o in outputs]),
        np.concatenate([o.poca_KDE_B_xMax for o in outputs]),
        np.concatenate([o.poca_KDE_B_yMax for o in outputs]),
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
##  added 200922
    hf.create_dataset("poca_KDE_A", data=od.poca_KDE_A, compression=compression)
    hf.create_dataset("poca_KDE_A_xMax", data=od.poca_KDE_A_xMax, compression=compression)
    hf.create_dataset("poca_KDE_A_yMax", data=od.poca_KDE_A_yMax, compression=compression)
    hf.create_dataset("poca_KDE_B", data=od.poca_KDE_B, compression=compression)
    hf.create_dataset("poca_KDE_B_xMax", data=od.poca_KDE_B_xMax, compression=compression)
    hf.create_dataset("poca_KDE_B_yMax", data=od.poca_KDE_B_yMax, compression=compression)

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
    akdh5["recon_x"] = od.recon_x
    akdh5["recon_y"] = od.recon_y
    akdh5["recon_z"] = od.recon_z
    akdh5["recon_tx"] = od.recon_tx
    akdh5["recon_ty"] = od.recon_ty
## mds    akdh5["recon_pocax"] = od.recon_pocax
## mds    akdh5["recon_pocay"] = od.recon_pocay
## mds    akdh5["recon_pocaz"] = od.recon_pocaz
## mds    akdh5["recon_sigmapocaxy"] = od.recon_sigmapocaxy

##  added 200922
    akdh5["poca_x"] = od.poca_x
    akdh5["poca_y"] = od.poca_y
    akdh5["poca_z"] = od.poca_z
    akdh5["major_axis_x"] = od.major_axis_x
    akdh5["major_axis_y"] = od.major_axis_y
    akdh5["major_axis_z"] = od.major_axis_z
    akdh5["minor_axis1_x"] = od.minor_axis1_x
    akdh5["minor_axis1_y"] = od.minor_axis1_y
    akdh5["minor_axis1_z"] = od.minor_axis1_z
    akdh5["minor_axis2_x"] = od.minor_axis2_x
    akdh5["minor_axis2_y"] = od.minor_axis2_y
    akdh5["minor_axis2_z"] = od.minor_axis2_z

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
    ##  take the following  constants used in calculating pvRes from LHCb-PUB-2017-005
    A_res = 926.0
    B_res = 0.84
    C_res = 10.7
    mds_counter = 0

    with Timer(start=f"Loading file: {name}"):
        tree = uproot.open(str(filepath))["kernel"]

## mds 10 Sept 2020        X = (tree["zdata"].array() / 2500.0).astype(dtype_X)  # Density in z, 4000xN
## mds 10 Sept 2020        Xmax = (tree["xmax"].array() / 2500.0).astype(
##        X = (tree["oldzdata"].array() / 50.0).astype(dtype_X)  # Density in z, 4000xN
## in origdata_mdsA (for original KDEs) we divided by 2500
## Oops! That was wrong scaling factor.  Try dividing by 6500 rather than 2500
        X = (tree["oldzdata"].array() / 2500.0).astype(dtype_X)  # Density in z, 4000xN
        print("at creation, X.shape = ",X.shape)
        Xmax = (tree["oldxmax"].array() / 2500.0).astype(
            dtype_X
        )  # Location of max z in x   <OPTIONAL>
## mds 10 Sept 2020        Ymax = (tree["ymax"].array() / 2500.0).astype(
        Ymax = (tree["oldymax"].array() / 2500.0).astype(
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
        recon_x = tree["recon_x"].array()
        recon_y = tree["recon_y"].array()
        recon_z = tree["recon_z"].array()
        recon_tx = tree["recon_tx"].array()
        recon_ty = tree["recon_ty"].array()
## mds        recon_pocax = tree["recon_pocax"].array()
## mds        recon_pocay = tree["recon_pocay"].array()
## mds        recon_pocaz = tree["recon_pocaz"].array()
## mds        recon_sigmapocaxy = tree["recon_sigmapocaxy"].array()
## 200922 mds  add the following variables; note that the names of the 
##             Python variables differ from those of the ROOT variables
##  the "scaling" factors of 50.0 and 2500.0 may change after Marian
##  updates the KDE calculations to account for the determinants of
##  the inverse covariance matrices. 
        poca_x = tree["POCA_center_x"].array()
        poca_y = tree["POCA_center_y"].array()
        poca_z = tree["POCA_center_z"].array()
        major_axis_x = tree["POCA_major_axis_x"].array()
        major_axis_y = tree["POCA_major_axis_y"].array()
        major_axis_z = tree["POCA_major_axis_z"].array()
        minor_axis1_x = tree["POCA_minor_axis1_x"].array()
        minor_axis1_y = tree["POCA_minor_axis1_y"].array()
        minor_axis1_z = tree["POCA_minor_axis1_z"].array()
        minor_axis2_x = tree["POCA_minor_axis2_x"].array()
        minor_axis2_y = tree["POCA_minor_axis2_y"].array()
        minor_axis2_z = tree["POCA_minor_axis2_z"].array()
        poca_KDE_A = (tree["POCAzdata"].array() / 1000.0).astype(dtype_X) 
        poca_KDE_A_xMax = (tree["POCAxmax"].array() / 2500.0).astype(dtype_X) 
        poca_KDE_A_yMax = (tree["POCAymax"].array() / 2500.0).astype(dtype_X) 
        poca_KDE_B = (tree["POCA_sqzdata"].array() / 10000.0).astype(dtype_X) 
        poca_KDE_B_xMax = (tree["POCA_sqxmax"].array() / 2500.0).astype(dtype_X) 
        poca_KDE_B_yMax = (tree["POCA_sqymax"].array() / 2500.0).astype(dtype_X) 
        poca_KDE_A_xMax[0 == poca_KDE_A] = 0
        poca_KDE_A_yMax[0 == poca_KDE_A] = 0
        poca_KDE_B_xMax[0 == poca_KDE_B] = 0
        poca_KDE_B_yMax[0 == poca_KDE_B] = 0
##  end of 200922 additions 

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

            if mds_counter < 0:

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
                if mds_counter < 0:
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

                    if mds_counter < 0:
                        print("\n \n n = ", n)
                        print(" \n centers = ", centers)
                        print(" nTrks  =       ", nTrks)
                        print(" sd     =       ", sd)

                    for mean, ntrk, pv_res in zip(centers, nTrks, sd):
                        if mds_counter < 0:
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
                        if mds_counter < 0:
                            print("N_bin = ", N_bin)
                            print("mat = ", mat)
                            print("  values = ", values)

                        populate = values[1] - values[0]
                        populate = np.where(
                            (0.15 / pv_res) > 1, (0.15 / pv_res) * populate, populate
                        )
                        if mds_counter < 0:
                            print(" populate = ", populate)

                        try:
                            ## mds                    Y[n, i, bins + N_bin] += values[1] - values[0]
                            Y[n, i, bins + N_bin] += populate
                        except IndexError:
                            msgs.append(
                                f"Ignored hit at bin {N_bin} at {mean:.4g} in event {i}, column {n}"
                            )

## mds    for msg in msgs:
## mds         print(" ", msg)
## mds 
## mds     print("X.shape = ",X.shape)
## mds     print("Y.shape = ",Y.shape)
## mds     print("Xmax.shape = ",Xmax.shape)
## mds     print("Ymax.shape = ",Ymax.shape)
## mds     print("pv_loc_x.shape = ",pv_loc_x.shape)
## mds     print("pv_loc_y.shape = ",pv_loc_y.shape)
## mds     print("pv_loc.shape = ",pv_loc.shape)
## mds     print("pv_ntrks.shape = ",pv_ntrks.shape)
## mds     print("pv_cat.shape = ", pv_cat.shape)
## mds     print("sv_loc_x.shape = ", sv_loc_x.shape)
## mds     print("sv_loc_y.shape = ", sv_loc_y.shape)
## mds     print("sv_loc.shape = ", sv_loc.shape)
## mds     print("sv_ntrks.shape = ", sv_ntrks.shape)
## mds     print("sv_cat.shape = ", sv_cat.shape)
## mds     print("recon_x.shape = ",recon_x.shape)
## mds     print("recon_y.shape = ",recon_y.shape)
## mds     print("recon_z.shape = ",recon_z.shape)
## mds     print("recon_tx.shape = ",recon_tx.shape)
## mds     print("recon_ty.shape = ",recon_ty.shape)
## mds     print("poca_x.shape = ",poca_x.shape)
## mds     print("poca_y.shape = ",poca_y.shape)
## mds     print("poca_z.shape = ",poca_z.shape)
## mds     print("poca_KDE_A.shape = ",poca_KDE_A.shape)
## mds     print("poca_KDE_A_xMax.shape = ",poca_KDE_A_xMax.shape)
## mds     print("poca_KDE_A_yMax.shape = ",poca_KDE_A_yMax.shape)
## mds     print("poca_KDE_B.shape = ",poca_KDE_B.shape)
## mds     print("poca_KDE_B_xMax.shape = ",poca_KDE_B_xMax.shape)
## mds     print("poca_KDE_B_yMax.shape = ",poca_KDE_B_yMax.shape)

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
        recon_x,
        recon_y,
        recon_z,
        recon_tx,
        recon_ty,
## mds        recon_pocax,
## mds        recon_pocay,
## mds        recon_pocaz,
## mds        recon_sigmapocaxy,
        poca_x,               ## poca ellipsoid center, x coordinate
        poca_y,               ## poca ellipsoid center, y coordinate
        poca_z,               ## poca ellipsoid center, z coordinate
        major_axis_x,         ## for poca ellipsoid
        major_axis_y,         ## for poca ellipsoid
        major_axis_z,         ## for poca ellipsoid
        minor_axis1_x,        ## for poca ellipsoid
        minor_axis1_y,        ## for poca ellipsoid
        minor_axis1_z,        ## for poca ellipsoid
        minor_axis2_x,        ## for poca ellipsoid
        minor_axis2_y,        ## for poca ellipsoid
        minor_axis2_z,        ## for poca ellipsoid
        poca_KDE_A,           ## KDE calculated from summing probabilities
        poca_KDE_A_xMax,      ## x value where poca_KDE_A was found
        poca_KDE_A_yMax,      ## y value where poca_KDE_A was found
        poca_KDE_B,           ## KDE calculated from summing probability square values
        poca_KDE_B_xMax,      ## x value where poca_KDE_B was found
        poca_KDE_B_yMax,      ## y value where poca_KDE_B was found
    ) 
