
# Y_file = "Output_Y_75000_July_26_2.npy"
# X_file = "Input_X_75000_July_26_2.npy"

import torch
from torch.utils.data import TensorDataset
import numpy as np
from utilities import Timer

class DataCollector:
    """
    This class loads data, and produces dataloaders from it.
    """
    
    __slots__ = "X Y train val test dtype".split()
    
    def __init__(self, XY_file, training, validation, *, dtype=np.float32):
        """
        Give this class a file name, and the training/validation sizes (test computed automatically).
        """
        
        self.dtype = dtype
        
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), np.load(XY_file) as XY:
            self.X = XY['kernel'][:,np.newaxis,:] 
            self.Y = XY['pv']
            
        if training <= 1:
            training = int(len(self.X) * training)

        if validation <= 1:
            validation = int(len(self.X) * validation)

        assert len(self.X) == len(self.Y), 'Lengths of X and Y must match'
        assert len(self.X) >= training + validation, 'Must have two or three parts'
        
        self.train = slice(0, training)
        self.val = slice(training, training + validation)
        self.test = slice(training + validation, len(self.X))
        
        print("Samples in Training:", training, "Validation:", validation, "Test:", len(self.X) - training - validation)
        
    def _get_dataset(self, slice_ds, batch_size, events=None, device=None, **kargs):
        """
        Internal function that loads a dataset. Keyword arguments are passed on to DataLoader.
        """
        
        if events is not None:
            length = slice_ds.stop - slice_ds.start
            assert events <= length, f'You asked for too many events, max is {length}'
            slice_ds = slice(slice_ds.start, slice_ds.start + events)
        
        if device is None:
            device = torch.device('cpu')
            
        with Timer(start=f"Constructing dataset on {device}"):
            dataset = TensorDataset(torch.tensor(self.X[slice_ds].astype(self.dtype)).to(device),
                                    torch.tensor(self.Y[slice_ds].astype(self.dtype)).to(device))

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             **kargs)
        return loader
        
    def get_training(self, batch_size, events=None, device=None, **kargs):
        """
        batch_size: The number of events per batch
        events: The number of training events to load (all by default)
        device: The device to load onto (CPU by default)
        **kargs: Any other keyword arguments go to dataloader
        """
        return self._get_dataset(self.train, batch_size, events, device, **kargs)
        
    def get_validation(self, batch_size, events=None, device=None, **kargs):
        """
        batch_size: The number of events per batch
        events: The number of validation events to load (all by default)
        device: The device to load onto (CPU by default)
        **kargs: Any other keyword arguments go to dataloader
        """
        return self._get_dataset(self.val, batch_size, events, device, **kargs)
    
    def get_testing(self, batch_size, events=None, device=None, **kargs):
        """
        batch_size: The number of events per batch
        events: The number of testing events to load (all by default)
        device: The device to load onto (CPU by default)
        **kargs: Any other keyword arguments go to dataloader
        """
        return self._get_dataset(self.test, batch_size, events, device, **kargs)

