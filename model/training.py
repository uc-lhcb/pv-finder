import time

import torch
import torch.optim as optim
from collections import namedtuple
from contextlib import contextmanager
import sys

from utilities import tqdm_redirect

Results = namedtuple("Results", ['cost','val','time','epoch'])

def trainNet(model, dataset_train, dataset_val, loss, batch_size, n_epochs, *,
             learning_rate=1e-3, notebook=None):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """
    
    
    # Print all of the hyperparameters of the training iteration
    if not notebook:
        given_args = "learning_rate n_epochs batch_size loss model".split()
        print("{0:=^80}".format(" HYPERPARAMETERS "))
        for item in given_args:
            print(item, "=", locals()[item])
        print(f"dataset_train: {dataset_train.tensors[0].size()[0]} events")
        print(f"dataset_val: {dataset_val.tensors[0].size()[0]} events")
        print("="*80)
        
    # Set up notebook or regular progress bar (or none)
    if notebook is None:
        progress = None
    elif notebook:
        from tqdm import tqdm_notebook as progress
    elif sys.stdout.isatty():
        from tqdm import tqdm as progress
    else:
        # Don't display progress if this is not a
        # notebook and not connected to the terminal
        progress = None
        

    # Get the current device
    layer = model
    if isinstance(layer, torch.nn.DataParallel):
        layer = list(layer.children())[0]
    layer = list(layer.children())[0]
    if isinstance(layer, torch.nn.DataParallel):
        layer = list(layer.children())[0]
    device = layer.weight.device
    device_matches = device == dataset_train.tensors[0].device
    
    # Get training data (only pin if devices don't match)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=not device_matches)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=not device_matches)
    
    # Create our optimizer function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists for the cost of every epoch
    cost_epoch = []
    val_epoch  = []
    time_epoch = []
    
    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")
        
    if progress:
        epoch_iterator = progress(range(n_epochs), desc="Epochs",
                                  postfix='train=start, val=start', dynamic_ncols=True,
                                  position=0, file=sys.stdout)
    else:
        epoch_iterator = range(n_epochs)

    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(model, loss, train_loader, optimizer, device, progress=progress)
        cost_epoch.append(total_train_loss / len(train_loader))

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = validate(model, loss, val_loader, device)
        val_epoch.append(total_val_loss / len(val_loader))

        # Record total time
        time_epoch.append(time.time() - training_start_time)

        # Pretty print a description
        if progress:
            epoch_iterator.postfix = f'train={cost_epoch[-1]:.4}, val={val_epoch[-1]:.4}'
        
        # Redirect stdout if needed to avoid clash with progress bar
        with tqdm_redirect(progress):
            if not notebook:
                print(f'Epoch {epoch}: train={cost_epoch[-1]:.6}, val={val_epoch[-1]:.6}, took {time_epoch[-1]:.5} s')
            yield Results(cost_epoch, val_epoch, time_epoch, epoch)


def train(model, loss, loader, optimizer, device, progress=None):
    total_loss = 0.0
    
    # switch to train mode
    model.train()
    
    if progress:
        loader = progress(loader, postfix='train=start',
                          desc="Training", mininterval=1.0, dynamic_ncols=True,
                          position=1, leave=False, file=sys.stdout)
    
    for inputs, labels in loader:
        if device is not None and inputs.device != device:
            inputs, labels = inputs.to(device), labels.to(device)

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = model(inputs)
        loss_output = loss(outputs, labels)
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output.data.item()
        
        if progress:
            loader.postfix = f'train={loss_output.data.item():.4g}'

    return total_loss 


def validate(model, loss, loader, device):
    total_loss = 0
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            if device is not None and inputs.device != device:
                inputs, labels = inputs.to(device), labels.to(device)

            #Forward pass
            val_outputs = model(inputs)
            loss_output = loss(val_outputs, labels)
            
            total_loss += loss_output.data.item()
    return total_loss