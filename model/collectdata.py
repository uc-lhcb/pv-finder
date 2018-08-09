
# Y_file = "Output_Y_75000_July_26_2.npy"
# X_file = "Input_X_75000_July_26_2.npy"

import torch
from torch.utils.data import TensorDataset
import numpy as np
import time

class Timer(object):
    def __init__(self, message, verbose=True):
        self.message = message
        self.verbose = verbose
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, *args):
        self.end_time = time.time()
        if self.verbose:
            print(self.message.format(time=self.end_time - self.start_time))

def collect_data(XY_file, training, validation, device = None, verbose=True):
    "Load a pair of files into three tensor datasets"

    # We devide the input X by 2500, so most of the values are between 0 and 1.
    # Also we want (N,1,4000) as our shape for X and (N,4000) for Y

    if verbose:
        print("Loading", XY_file)

    with Timer("Loaded files in {time:.4} s", verbose), np.load(XY_file) as XY:
        X=XY['X'].astype(np.float32)[:,np.newaxis,:] / 2500.
        Y=XY['Y'].astype(np.float32)

    if training <= 1:
        training = int(len(X) * training)

    if validation <= 1:
        validation = int(len(X) * validation)

    assert len(X) == len(Y), 'Lengths must match'
    assert len(X) >= training + validation, 'Must have two or three parts'

    train = slice(0, training)
    val = slice(training, training + validation)
    test = slice(training + validation, len(X))

    print("Samples in Training:", training, "Validation:", validation, "Test:", len(X) - training - validation)

    if device is None:
        with Timer("Constructed datasets on CPU in {time:.4} s"):
            train_ds = TensorDataset(torch.tensor(X[train]),
                                     torch.tensor(Y[train]))
            valid_ds = TensorDataset(torch.tensor(X[val]),
                                     torch.tensor(Y[val]))
            tests_ds = TensorDataset(torch.tensor(X[test]),
                                     torch.tensor(Y[test]))
    else:
        with Timer("Constructed datasets on device in {time:.4} s"):
            train_ds = TensorDataset(torch.tensor(X[train]).to(device),
                                     torch.tensor(Y[train]).to(device))
            valid_ds = TensorDataset(torch.tensor(X[val]).to(device),
                                     torch.tensor(Y[val]).to(device))
            tests_ds = TensorDataset(torch.tensor(X[test]).to(device),
                                     torch.tensor(Y[test]).to(device))

    return train_ds, valid_ds, tests_ds


