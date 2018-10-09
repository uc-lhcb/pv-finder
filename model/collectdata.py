
import torch
from torch.utils.data import TensorDataset
import numpy as np
from utilities import Timer
from pathlib import Path
from functools import partial

def collect_data(*files, batch_size=1, dtype=np.float32, device=None, masking=False, slice=None, **kargs):
    """
    This function collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data('a.npz', 'b.npz')
    
    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """
    
    
    Xlist = []
    Ylist = []
    
    for XY_file in files:
        if Path(XY_file).suffix != '.h5':
            load = np.load
        else:
            import h5py
            load = partial(h5py.File, mode='r')
    
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), load(XY_file) as XY:
            X = np.asarray(XY['kernel'])[:,np.newaxis,:].astype(dtype)
            Y = np.asarray(XY['pv']).astype(dtype)

            if masking:
                # Set the result to nan if the "other" array is above threshold
                # and the current array is below threshold
                Y[(np.asarray(XY['pv_other']) > 0.01) & (Y < 0.01)] = dtype(np.nan)
                
            Xlist.append(X)
            Ylist.append(Y)
            
    X = np.concatenate(Xlist)
    Y = np.concatenate(Ylist)
            
    if slice:
        X = X[slice, :, :]
        Y = Y[slice, :]
            
    if device is None:
        device = torch.device('cpu')

    with Timer(start=f"Constructing {X.shape[0]} event dataset on {device}"):
        dataset = TensorDataset(torch.tensor(X).to(device),
                                torch.tensor(Y).to(device))

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         **kargs)
    return loader
