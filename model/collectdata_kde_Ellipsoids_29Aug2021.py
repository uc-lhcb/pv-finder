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

## this contains the method 
##  six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2)
from model.ellipsoids import six_ellipsoid_parameters

# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

try:
    import awkward0 as awkward
except ModuleNotFoundError:
    print('ModuleNotFoundError when trying to import awkward0 as awkward')
    import awkward

ja = awkward.JaggedArray

dtype_X = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU
dtype_Y = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU

VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n", "cat"))

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

    """
    for LHCb, assume we want to produce a 4000-bin histogram
    spanning the range -100.0 mm to +300.0 mm

    the following constants will be used to extract the bin number
    and the offset from the lower bin edge
   
    add 210829 mds
    """
    z_low = -100.
    z_high = 300.
    n_zBins = 4000
    z_binWidth = (z_high - z_low)/n_zBins

## these unit vectors will be used to convert the elements of 
## the ellipsoid major and minor axis vectors into vectors
    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 1, 0])
    zhat = np.array([0, 0, 1])

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

## mds 200729  the KDE targets have many zeros. Learning zeros using a ratio
## mds         of predicted to target means that overestimating by a small
## mds         amount in the cost function, even adding an epsilon-like parameter## mds         there is difficult. Let's explicitly add epsilon here.
## mds         We might be able to do it equally well in the cost function,
## mds         but doing it here makes plotting easy as well.

            epsilon = 0.001 
## mds 201019            kernel = np.asarray(f["kernel"]) + epsilon
## we want to use the poca KDE, not the original kernel
            kernel = np.asarray(f["poca_KDE_A"]) + epsilon
            Xmax = 2500.*np.asarray(f["Xmax"])
            Ymax = 2500.*np.asarray(f["Ymax"]) 
            
            Y = ja.concatenate((kernel,Xmax,Ymax),axis=1).astype(dtype_Y)
            
## now build the feature set from the relevant tracks' parameters
## we need to use "afile" to account for the variable length
## structure of the awkward arrays

##  201018  use poca ellipsoid parameter rather than "track parameters"
        

            afile = awkward.hdf5(f)
          
## 210901   mds  we want to manipulate the pocaz information from the awkward
##          array to create zBin and zOffset before "wrapping" it as an numpy
##          array so that the structures of the (eventual) zBin and zOffset
##          arrays will be the same as the numpy arrays built directly from
##          the awkward arrays.  Henry says there will be a cleaner way to
##          do this using awkard 1, but works for the moment
            pocaz = (afile["poca_z"].astype(dtype_Y)).astype(dtype_Y)
            zz = pocaz-z_low
            zzz = (zz/z_binWidth)
            zzzz = np.floor(zzz)
            zBin = np.asarray(zzzz.astype(int))
     
            zOffset = (pocaz-((zzzz*z_binWidth)+z_low))
            zOffset = np.asarray(zOffset)
 
            pocaz = np.asarray(pocaz)
##  the original pocaz values range from -100. to 300. and we want them to
##  populate a range with magnitudes < 1 to make the machine learning easier
            pocaz = 0.001*pocaz
            print("pocaz.shape = ",pocaz.shape)

            pocax = np.asarray(afile["poca_x"].astype(dtype_Y))
            pocay = np.asarray(afile["poca_y"].astype(dtype_Y))
            pocaMx = np.asarray(afile["major_axis_x"].astype(dtype_Y))
            print("pocaMx.shape = ", pocaMx.shape)
            pocaMy = np.asarray(afile["major_axis_y"].astype(dtype_Y))
            pocaMz = np.asarray(afile["major_axis_z"].astype(dtype_Y))

            nEvts = len(pocaz)
            print("nEvts = ", nEvts)

            print("len(pocaMx[0]) = ", len(pocaMx[0]))
            print("len(pocaMx[1]) = ", len(pocaMx[1]))
            print("len(pocaMx[2]) = ", len(pocaMx[2]))
            print("len(pocaMx[3]) = ", len(pocaMx[3]))
            print("len(pocaMx[4]) = ", len(pocaMx[4]))

            Mx = np.multiply(pocaMx.reshape(nEvts,1),xhat)
            My = np.multiply(pocaMy.reshape(nEvts,1),yhat)
            Mz = np.multiply(pocaMz.reshape(nEvts,1),zhat)
            majorAxis = Mx+My+Mz
            print("majorAxis.shape = ",majorAxis.shape)


            poca_m1x = np.asarray(afile["minor_axis1_x"].astype(dtype_Y))
            poca_m1y = np.asarray(afile["minor_axis1_y"].astype(dtype_Y))
            poca_m1z = np.asarray(afile["minor_axis1_z"].astype(dtype_Y))

            mx = np.multiply(poca_m1x.reshape(nEvts,1),xhat)
            my = np.multiply(poca_m1y.reshape(nEvts,1),yhat)
            mz = np.multiply(poca_m1z.reshape(nEvts,1),zhat)
            minorAxis_1 = mx+my+mz
            print("minorAxis_1.shape = ",minorAxis_1.shape)

            poca_m2x = np.asarray(afile["minor_axis2_x"].astype(dtype_Y))
            poca_m2y = np.asarray(afile["minor_axis2_y"].astype(dtype_Y))
            poca_m2z = np.asarray(afile["minor_axis2_z"].astype(dtype_Y))


            mx = np.multiply(poca_m2x.reshape(nEvts,1),xhat)
            my = np.multiply(poca_m2y.reshape(nEvts,1),yhat)
            mz = np.multiply(poca_m2z.reshape(nEvts,1),zhat)
            minorAxis_2 = mx+my+mz
            print("minorAxis_2.shape = ",minorAxis_1.shape)


            A, B, C, D, E, F = six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2)

            print("A.shape = ",A.shape)
            for iTrk in range(1):
              print("majorAxis[iTrk][0][0] = ",majorAxis[iTrk][0][0])
              print("majorAxis[iTrk][1][0] = ",majorAxis[iTrk][1][0])
              print("majorAxis[iTrk][2][0] = ",majorAxis[iTrk][2][0])
              print("minorAxis_1[iTrk][0][0] = ",minorAxis_1[iTrk][0][0])
              print("minorAxis_1[iTrk][1][0] = ",minorAxis_1[iTrk][1][0])
              print("minorAxis_1[iTrk][2][0] = ",minorAxis_1[iTrk][2][0])
              print("minorAxis_2[iTrk][0][0] = ",minorAxis_2[iTrk][0][0])
              print("minorAxis_2[iTrk][1][0] = ",minorAxis_2[iTrk][1][0])
              print("minorAxis_2[iTrk][2][0] = ",minorAxis_2[iTrk][2][0])
              print("  ")


            

##  mark non-track data with -99 as a flag
            maxLen = 600 ## for safety:  600 >> 481, which is what was seen for 100 evts
            padded_pocaz   = np.zeros((nEvts,maxLen))-99.
            padded_pocax   = np.zeros((nEvts,maxLen))-99.
            padded_pocay   = np.zeros((nEvts,maxLen))-99.
            padded_pocaA  = np.zeros((nEvts,maxLen))-99.
            padded_pocaB  = np.zeros((nEvts,maxLen))-99.
            padded_pocaC  = np.zeros((nEvts,maxLen))-99.
            padded_pocaD  = np.zeros((nEvts,maxLen))-99.
            padded_pocaE  = np.zeros((nEvts,maxLen))-99.
            padded_pocaF  = np.zeros((nEvts,maxLen))-99.

##  add the following 210829 mds
            padded_zBin     = np.zeros((nEvts,maxLen))-99.
            padded_zOffset  = np.zeros((nEvts,maxLen))-99.

            for i, e in enumerate(pocaz):
                fillingLength = min(len(e),maxLen)
                padded_pocaz[i,:fillingLength] = pocaz[i][:fillingLength].astype(dtype_Y)
                padded_pocax[i,:fillingLength] = pocax[i][:fillingLength].astype(dtype_Y)
                padded_pocay[i,:fillingLength] = pocay[i][:fillingLength].astype(dtype_Y)
                padded_pocaA[i,:fillingLength] = A[i][:fillingLength].astype(dtype_Y)
                padded_pocaB[i,:fillingLength] = B[i][:fillingLength].astype(dtype_Y)
                padded_pocaC[i,:fillingLength] = C[i][:fillingLength].astype(dtype_Y)
                padded_pocaD[i,:fillingLength] = D[i][:fillingLength].astype(dtype_Y)
                padded_pocaE[i,:fillingLength] = E[i][:fillingLength].astype(dtype_Y)
                padded_pocaF[i,:fillingLength] = F[i][:fillingLength].astype(dtype_Y)
## add the following 210829  mds
                padded_zBin[i,:fillingLength] = zBin[i][:fillingLength].astype(dtype_Y)
                padded_zOffset[i,:fillingLength] = zOffset[i][:fillingLength].astype(dtype_Y)

            padded_pocaz   = padded_pocaz[:,np.newaxis,:]
            padded_pocax   = padded_pocax[:,np.newaxis,:]
            padded_pocay   = padded_pocay[:,np.newaxis,:]
            padded_pocaA   = padded_pocaA[:,np.newaxis,:]
            padded_pocaB   = padded_pocaB[:,np.newaxis,:]
            padded_pocaC   = padded_pocaC[:,np.newaxis,:]
            padded_pocaD   = padded_pocaD[:,np.newaxis,:]
            padded_pocaE   = padded_pocaE[:,np.newaxis,:]
            padded_pocaF   = padded_pocaF[:,np.newaxis,:]
            padded_zBin    = padded_zBin[:,np.newaxis,:]
            padded_zOffset = padded_zOffset[:,np.newaxis,:]

## (multiplied by 0.001 at line 129 to create a nicer range 
## for deep learning)
            print('1000.*padded_pocaz[0:2,0,0:3] = ',1000.*padded_pocaz[0:2,0,0:3])
            print('padded_zBin[0:2,0,0:3] = ',1000.*padded_zBin[0:2,0,0:3])
            print('padded_zOffset[0:2,0,0:3] = ',1000.*padded_zOffset[0:2,0,0:3])

##             X = ja.concatenate((padded_pocaz,padded_pocax,padded_pocay,padded_pocaA,padded_pocaB,padded_pocaC,padded_pocaD,padded_pocaE,padded_pocaF),axis=1).astype(dtype_X)

## 210829 mds  --   zBin and zOffset; remove .astype *hoping* that the
##                 integer type of zOffset will be preserved
            X = ja.concatenate((padded_zBin,padded_zOffset,padded_pocaz,padded_pocax,padded_pocay,padded_pocaA,padded_pocaB,padded_pocaC,padded_pocaD,padded_pocaE,padded_pocaF),axis=1).astype(dtype_X)

## mds             print("X.shape = ",X.shape)
## mds             print('X[0:4,0,0:99] = ',X[0:4,0,0:99])

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

####### -----------------


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

