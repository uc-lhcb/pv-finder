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


import uproot
import numpy as np
from pathlib import Path
import sys
import math
from collections import namedtuple
import numba

from .utilities import Timer
from .jagged import concatenate
import awkward
import awkward as ak

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

    scalefactor = 2500.0
    
    name = filepath.stem
    ##  take the following  constants used in calculating pvRes from LHCb-PUB-2017-005
    ## edited by emk using extrapolated data from https://arxiv.org/pdf/1405.6569.pdf
    A_res = 568.7
    B_res = 0.956
    C_res = 10.33
    mds_counter = 0

    with Timer(start=f"Loading file: {name}"):
        tree = uproot.open(str(filepath))["kernel"]

        X = (tree["oldzdata"].array() / scalefactor)  # Density in z, 10000xN # EMK change scaling factor
        Xmax = (tree["oldxmax"].array()).to_numpy()  # Location of max z in x
        Ymax = (tree["oldymax"].array()).to_numpy()
        
        # set X and Y max to 0 where the kernel is 0
        X_zeros = (X == 0).to_numpy()
        Xmax[X_zeros] = 0
        Xmax = ak.Array(Xmax)
        Ymax[X_zeros] = 0
        Ymax = ak.Array(Ymax)
        
        pv_loc = tree["pv_loc"].array()  # z locations of each PV [#pvs]*N
        pv_loc_x = tree["pv_loc_x"].array()  # x  
        pv_loc_y = tree["pv_loc_y"].array()  # y
        pv_ntrks = tree["pv_ntrks"].array()  # number of tracks in PV [#pvs]*N   <OPTIONAL>
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
        
        
        poca_KDE_A = (tree["POCAzdata"].array() / 1000.0)
        poca_KDE_A_xMax = (tree["POCAxmax"].array() / scalefactor).to_numpy()
        poca_KDE_A_yMax = (tree["POCAymax"].array() / scalefactor).to_numpy()
        poca_KDE_B = (tree["POCA_sqzdata"].array() / 10000.0)
        poca_KDE_B_xMax = (tree["POCAxmax"].array() / scalefactor).to_numpy()
        poca_KDE_B_yMax = (tree["POCAymax"].array() / scalefactor).to_numpy()
        
        # set X and Y max to 0 where the kernel is 0
        KDE_A_zeros = (poca_KDE_A == 0).to_numpy()
        poca_KDE_A_xMax[KDE_A_zeros] = 0
        poca_KDE_A_xMax = ak.Array(poca_KDE_A_xMax)
        poca_KDE_A_yMax[KDE_A_zeros] = 0
        poca_KDE_A_yMax = ak.Array(poca_KDE_A_yMax)
        
        KDE_B_zeros = (poca_KDE_B == 0).to_numpy()
        poca_KDE_B_xMax[KDE_B_zeros] = 0
        poca_KDE_B_xMax = ak.Array(poca_KDE_B_xMax)
        poca_KDE_B_yMax[KDE_B_zeros] = 0
        poca_KDE_B_yMax = ak.Array(poca_KDE_B_yMax)
        
#         pv_ntrks.content = pv_ntrks.content.astype(np.uint16)
#         sv_ntrks.content = sv_ntrks.content.astype(np.uint16)

#         pv_cat.content = pv_cat.content.astype(np.int8)
#         sv_cat.content = sv_cat.content.astype(np.int8)

    N_vals = len(X)
    n_bins = 10000
    zmin = -25.0
    zmax = 25.0
    bin_width = (zmax-zmin)/n_bins
    zvals_range = (zmin+bin_width/2, zmax-bin_width/2)
    Y = np.zeros([4, N_vals, n_bins], dtype=dtype_Y)
    edges = np.array([-bin_width/2, bin_width/2])
    bins = np.arange(-10, 11)
    mat = bin_width * bins[np.newaxis, :] + edges[:, np.newaxis] + zvals_range[0]

    msgs = []
    with Timer(start=f"Processing events: {name}"):
        for i in range(N_vals):
            columns = (
                ## below zeros bc currently no category (everything zeros)
                (pv_loc[i][pv_cat[i] >= 0], pv_ntrks[i][pv_cat[i] >= 0]),
                (pv_loc[i][pv_cat[i] < 0], pv_ntrks[i][pv_cat[i] < 0]),
                (sv_loc[i][sv_cat[i] >= 0], sv_ntrks[i][sv_cat[i] >= 0]),
                (sv_loc[i][sv_cat[i] < 0], sv_ntrks[i][sv_cat[i] < 0]),
            )

#             if mds_counter < 10:

#                 print(" \n \n N_vals = ", N_vals)

#                 print("len(columns) = ", len(columns))
#                 print("columns = \n", columns)
#                 print("columns[0][0] = ", columns[0][0])
#                 print("columns[0][1] = ", columns[0][1])
#                 print("columns[1][0] = ", columns[1][0])
#                 print("columns[1][1] = ", columns[1][1])
#                 print("columns[2][0] = ", columns[2][0])
#                 print("columns[2][1] = ", columns[2][1])
#                 print("columns[3][0] = ", columns[3][0])
#                 print("columns[3][1] = ", columns[3][1])
            mds_counter += 1
            ##            assert(mds_counter)<10

            for n, elements in enumerate(columns):

                ##  addition 190810 mds
                ##            nTrks = np.where(np.isnan(nTrks), 0, nTrks)
                ## mds            pvRes = A_res*np.power(nTrks, -1*B_res) + C_res   # values taken from LHCb-PUB-2017-005
                ## mds            sd = 0.001*pvRes  ## convert from microns (in TDR) to  mm (units used here)

                entries = len(elements[0])
#                 if mds_counter < 10:
#                     print(" \n \n \n elements =  ", elements)
#                     print(" entries = ", entries)
                for v_index in range(entries):
        
                    centers = elements[0][v_index]
                    nTrks = elements[1][v_index]
                    sd = np.where(
                        nTrks < 4,
                        sd_1,
                        0.01 * (A_res * np.power(nTrks, -1 * B_res) + C_res),
                    )

                    test = (zvals_range[0] < centers) & (centers < zvals_range[1])
    
                    centers = centers[test]
                    nTrks = nTrks[test]
                    sd = sd[test]

#                     if mds_counter < 10:
#                         print("\n \n n = ", n)
#                         print(" \n centers = ", centers)
#                         print(" nTrks  =       ", nTrks)
#                         print(" sd     =       ", sd)

                    for mean, ntrk, pv_res in zip(centers, nTrks, sd):
#                         if mds_counter < 10:
#                             print(" \n iterating  using zip(centers, nTrks, sd)  ")
#                             print(" \n  mean = ", mean)
#                             print(" ntrk   = ", ntrk)
#                             print(" pv_res = ", pv_res)

                        # Compute bin number
                        N_bin = int(np.floor((mean - zvals_range[0]) /bin_width))
                        values = norm_cdf(mean, pv_res, N_bin *bin_width + mat)
                        ##  replace sd_1 with sd computed from nTrks
                        ##  and increase area to be inversely proportional to sd for sd <150 microns
                        ##  (which is roughly the resolution expected for nTrks ~ 10) and with
                        ##  with unit area at sd = 0.150 mm = 150 micron, and larger values (lower nTrk)
                        ##  altogether, this will make the target peaks more narrow and the areas larger as the
                        ##  number of tracks increases.
#                         if mds_counter < 10:
#                             print("N_bin = ", N_bin)
#                             print("mat = ", mat)
#                             print("  values = ", values)

                        populate = values[1] - values[0]
                        populate = np.where(
                            (0.15 / pv_res) > 1, (0.15 / pv_res) * populate, populate
                        )
#                         if mds_counter < 10:
#                             print(" populate = ", populate)

                        try:
                            ## mds                    Y[n, i, bins + N_bin] += values[1] - values[0]
                            Y[n, i, bins + N_bin] += populate
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
