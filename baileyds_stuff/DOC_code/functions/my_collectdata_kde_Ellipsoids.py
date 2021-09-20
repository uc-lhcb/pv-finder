import torch
from torch.utils.data import TensorDataset

import numpy as np
from pathlib import Path
from functools import partial
import warnings
from collections import namedtuple

import time

import matplotlib.pyplot as plt

from model.jagged import concatenate
from model.utilities import Timer

from model.ellipsoids import six_ellipsoid_parameters

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward

ja = awkward.JaggedArray

dtype_X = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU
dtype_Y = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU

VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n", "cat"))

#Gaussian noise transform I found
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def normalize(tensor):
    a = np.amin(tensor.cpu().numpy())
    b = np.amax(tensor.cpu().numpy())
    print(tensor)
    print('max:', b, '| min:', a)
    print('subtract:', (tensor-a))
    print('div by:', (b-a))
    print('to get:', (tensor-a)/(b-a))
    return (tensor-a)/(b-a)
    
    
def add_noise(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor + torch.randn_like(tensor)*std + mean
    
def mean_smoothing(tensor, window_size=10):
    device = tensor.device
    window_mid = int(window_size/2)

    print('initial tensor shape:', tensor.shape)
    
    nEvts = tensor.shape[0]
    nFeatures = 4000
    print("nEvts:", nEvts)
    
    y = tensor.view(nEvts,-1,nFeatures)

    y = y.transpose(1,2) 
    
    print('y shape:', y.shape)
    
    old_arr = y.cpu().numpy()
    
    
    smooth_arr = np.ones_like(old_arr)*old_arr[0,0]
    for i in range(smooth_arr.shape[0]):
        for j in range(window_mid, smooth_arr.shape[1]-window_mid):
            #print('middle:', old_arr[i,j,0], '| sum:', sum([old_arr[i,j+k,0] for k in range(-window_mid, window_mid)]))
            smooth_arr[i,j,0] = sum([old_arr[i,j+k,0] for k in range(-window_mid, window_mid)])/window_size
#         plt.figure(figsize=(12,8))
#         plt.plot(old_arr[i, :, 0])
#         plt.plot(smooth_arr[i, :, 0])
    
    smooth_arr = torch.from_numpy(smooth_arr).to(device)
    
    smooth_arr = smooth_arr.transpose(1,2)
    smooth_arr = smooth_arr.view(nEvts, -1)
    
    print('output tensor shape:', smooth_arr.shape)
    
    return smooth_arr
    
def smooth_matr(arr_size, w, device):
    M = torch.zeros((arr_size,arr_size), device=device)
    K = sum([(1+w-abs(i)) for i in range(-w, w+1)])
    for i in range(-w, w+1):
        M += (((1+w)-abs(i))/K)*torch.tril(torch.triu(torch.ones((arr_size,arr_size), device=device), diagonal=i), diagonal=i)
    return M
    
def mean_better_smoothing(tensor, window_width=10):
    
    print("\nSmoothing...")
    start_time = time.time()
    
    device = tensor.device
    
    nEvts = tensor.shape[0]
    nFeatures = 4000
    
    print('Reshaping tensor')
    y = tensor.view(nEvts,-1,nFeatures)
    y = y.transpose(1,2) 
    #old_arr = y.cpu().numpy()
    
    smooth_arr = torch.ones_like(y)
    
    print('Creating smoothing array')
    #M = torch.from_numpy(smooth_matr(nFeatures, window_width)).to(device)
    M = smooth_matr(nFeatures, window_width, device)
    
    print('Looping through events')
    for i in range(smooth_arr.shape[0]):
        #print(i, '/', smooth_arr.shape[0])
        
        for d in range(3):
            smooth_arr[i,:,d] = torch.matmul(M, y[i,:,d])
             
#         plt.figure(figsize=(12,8))
#         plt.plot(y[i, :, 0].cpu().numpy())
#         plt.plot(smooth_arr[i, :, 0].cpu().numpy())
    
    #smooth_arr = torch.from_numpy(smooth_arr).to(device)
    
    smooth_arr = smooth_arr.transpose(1,2)
    smooth_arr = smooth_arr.view(nEvts, -1)
    
    #print('output tensor shape:', smooth_arr.shape)
    
    print("Smoothing finished; took", time.time()-start_time, 's')
    
    return smooth_arr    
    
def collect_t2kde_data_withtransform(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    slice=None,
    norm=False,
    noise=False,
    smooth = 10,
    verbose=True,
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
            
            pocaz = np.asarray(0.001*afile["poca_z"].astype(dtype_Y))
            pocax = np.asarray(afile["poca_x"].astype(dtype_Y))
            pocay = np.asarray(afile["poca_y"].astype(dtype_Y))
            pocaMx = np.asarray(afile["major_axis_x"].astype(dtype_Y))
            if verbose == True:
                print("pocaMx.shape = ", pocaMx.shape)
            pocaMy = np.asarray(afile["major_axis_y"].astype(dtype_Y))
            pocaMz = np.asarray(afile["major_axis_z"].astype(dtype_Y))

            nEvts = len(pocaz)
            if verbose == True:    
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
            if verbose == True:
                print("majorAxis.shape = ",majorAxis.shape)


            poca_m1x = np.asarray(afile["minor_axis1_x"].astype(dtype_Y))
            poca_m1y = np.asarray(afile["minor_axis1_y"].astype(dtype_Y))
            poca_m1z = np.asarray(afile["minor_axis1_z"].astype(dtype_Y))

            mx = np.multiply(poca_m1x.reshape(nEvts,1),xhat)
            my = np.multiply(poca_m1y.reshape(nEvts,1),yhat)
            mz = np.multiply(poca_m1z.reshape(nEvts,1),zhat)
            minorAxis_1 = mx+my+mz
            if verbose == True:
                print("minorAxis_1.shape = ",minorAxis_1.shape)

            poca_m2x = np.asarray(afile["minor_axis2_x"].astype(dtype_Y))
            poca_m2y = np.asarray(afile["minor_axis2_y"].astype(dtype_Y))
            poca_m2z = np.asarray(afile["minor_axis2_z"].astype(dtype_Y))


            mx = np.multiply(poca_m2x.reshape(nEvts,1),xhat)
            my = np.multiply(poca_m2y.reshape(nEvts,1),yhat)
            mz = np.multiply(poca_m2z.reshape(nEvts,1),zhat)
            minorAxis_2 = mx+my+mz
            if verbose == True:
                print("minorAxis_2.shape = ",minorAxis_1.shape)


            A, B, C, D, E, F = six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2)
            if verbose == True:
                print("A.shape = ",A.shape)
                
            for iTrk in range(10):
                if verbose == True:
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
## mdsAA              print("A[iTrk][0] = ",A[iTrk][0])
## mdsAA              print("B[iTrk][0] = ",B[iTrk][0])
## mdsAA              print("C[iTrk][0] = ",C[iTrk][0])
## mdsAA              print("D[iTrk][0] = ",D[iTrk][0])
## mdsAA              print("E[iTrk][0] = ",E[iTrk][0])
## mdsAA              print("F[iTrk][0] = ",F[iTrk][0])
## mds              print("majorAxis[iTrk][0] = ", majorAxis[iTrk][0])
## mds              print("majorAxis[iTrk][1] = ", majorAxis[iTrk][1])
## mds              print("majorAxis[iTrk][2] = ", majorAxis[iTrk][2])


            

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

            padded_pocaz  = padded_pocaz[:,np.newaxis,:]
            padded_pocax  = padded_pocax[:,np.newaxis,:]
            padded_pocay  = padded_pocay[:,np.newaxis,:]
            padded_pocaA = padded_pocaA[:,np.newaxis,:]
            padded_pocaB = padded_pocaB[:,np.newaxis,:]
            padded_pocaC = padded_pocaC[:,np.newaxis,:]
            padded_pocaD = padded_pocaD[:,np.newaxis,:]
            padded_pocaE = padded_pocaE[:,np.newaxis,:]
            padded_pocaF = padded_pocaF[:,np.newaxis,:]

            X = ja.concatenate((padded_pocaz,padded_pocax,padded_pocay,padded_pocaA,padded_pocaB,padded_pocaC,padded_pocaD,padded_pocaE,padded_pocaF),axis=1).astype(dtype_X)

## mds            print("X = ",X)
            if verbose == True:
                print("len(X) = ",len(X))
            Xlist.append(X)
            Ylist.append(Y)
            if verbose == True:
                print("len(Xlist) = ",len(Xlist))
    X = np.concatenate(Xlist, axis=0)
    Y = np.concatenate(Ylist, axis=0)
    if verbose == True:
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
            
        #applying Gaussian noise with predefined std and mean
        if norm == True:
            y_t = normalize(y_t)
        
        if noise != False:
            mean = noise[0]
            std = noise[1]
            y_t = add_noise(y_t, mean, std)
            
        #print('before smoothing:', y_t.size())    
        if smooth != False:
            y_t = mean_better_smoothing(y_t, smooth)
        #print('before smoothing:', y_t.size())   
        
        dataset = TensorDataset(x_t, y_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)
    if verbose == True:
        print("x_t.shape = ",x_t.shape)
        print("x_t.shape[0] = ", x_t.shape[0])
        print("x_t.shape[1] = ", x_t.shape[1])
    nFeatures = 6
    x_t.view(x_t.shape[0],nFeatures,-1)
    if verbose == True:
        print("x_t.shape = ",x_t.shape)
    
    
    return loader