import time
import torch
from collections import namedtuple
import sys
import os

import numpy as np

from model.utilities import tqdm_redirect, import_progress_bar, get_device_from_model
from model.efficiency import efficiency, ValueSet

from .new_optimizers import normed_dot

"""
Based on the model/training_kde.py file, and is made of the modified training 
functions to work with various algorithms.
"""

Results = namedtuple("Results", ["epoch", "cost", "val", "time"])

PARAM_EFF = {
    "difference": 5.0,
    "threshold": 1e-2,
    "integral_threshold": 0.2,
    "min_width": 3,
}

def select_gpu(selection=None):
    """
    Select a GPU if availale.

    selection can be set to get a specific GPU. If left unset, it will REQUIRE that a GPU be selected by environment variable. If -1, the CPU will be selected.
    """

    if str(selection) == "-1":
        return torch.device("cpu")

    # This must be done before any API calls to Torch that touch the GPU
    if selection is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selection)

    if not torch.cuda.is_available():
        print("Selecting CPU (CUDA not available)")
        return torch.device("CPU")
    elif selection is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is *required* when running with CUDA available"
        )

    print(torch.cuda.device_count(), "available GPUs (initially using device 0):")
    for i in range(torch.cuda.device_count()):
        print(" ", i, torch.cuda.get_device_name(i))

    return torch.device("cuda:0")


def train(model, loss, loader, optimizer, device, progress):
    total_loss = 0.0

    # switch to train mode
    model.train()

    loader = progress(
        loader,
        postfix="train=start",
        desc="Training",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )

    for inputs, labels in loader:
        if inputs.device != device:
            inputs, labels = inputs.to(device), labels.to(device)

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = model(inputs)
        loss_output = loss(outputs, labels)
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output.data.item()
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss_output.data.item():.4g}"
    return total_loss

# modified to use the modified Adam, so you can get the grad, momenta, 
# and step size
def instrum_train(model, loss, loader, optimizer, device, progress):
    total_loss = 0.0

    # switch to train mode
    model.train()

    loader = progress(
        loader,
        postfix="train=start",
        desc="Training",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )
    
    # to store the optimizer outputs
    opt_outs = []

    for inputs, labels in loader:
        if inputs.device != device:
            inputs, labels = inputs.to(device), labels.to(device)

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = model(inputs)
        loss_output = loss(outputs, labels)
        loss_output.backward()
        opt_outs.append(optimizer.step())

        total_loss += loss_output.data.item()
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss_output.data.item():.4g}"
            
    return total_loss, opt_outs

def validate(model, loss, loader, device):
    total_loss = 0
    eff = ValueSet(0, 0, 0, 0)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            if inputs.device != device:
                inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            val_outputs = model(inputs)
            loss_output = loss(val_outputs, labels)

            total_loss += loss_output.data.item()

    return total_loss

"""
modified trainNet to use different algorithms and read the outputs from Adam

new features:
 -> ep_optimizer: set equal to an epoch optimizer, and it will step every epoch 
                  (see new_optimizers.py)
 -> lr_scheduler: set equal to a pytorch learning rate scheduler (or any scheduler that 
                  has a step method, I guess)
 -> careful: if not False, should be a tuple in the form 
             (loss_increase_limit, lr_decrease_factor, more careful? (boolean) )
             e.g. (0.02. 0.8, False) is for careful Adam where 
                  if loss increases by more than 2%, the learning rate decreases by 20%
 -> adaptive: if not False, should be the loss_increase_factor
             e.g. 0.05 is for adaptive Adam where if the 2 previous epoch steps are collinear,
                  the learning rate increases by 5%
                  
Final note: this is not guaranteed functional if you use these functions
simultaneously, except for adaptive and careful, so be warned.
"""
def trainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    epoch_start=0,
    ep_optimizer=None,
    lr_scheduler=None,
    careful = False,
    adaptive = False
):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """

    # Print all of the hyperparameters of the training iteration
    if not notebook:
        print("{0:=^80}".format(" HYPERPARAMETERS "))
        print(
            f"""\
n_epochs: {n_epochs}
batch_size: {train_loader.batch_size} events
dataset_train: {train_loader.dataset.tensors[0].size()[0]} events
dataset_val: {val_loader.dataset.tensors[0].size()[0]} events
loss: {loss}
optimizer: {optimizer}
model: {model}"""
        )
        print("=" * 80)

    # Set up notebook or regular progress bar (or none)
    progress = import_progress_bar(notebook)

    # Get the current device
    device = get_device_from_model(model)

    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")

    epoch_iterator = progress(
        range(epoch_start, n_epochs),
        desc="Epochs",
        postfix="train=start, val=start",
        dynamic_ncols=True,
        position=0,
        file=sys.stderr,
    )

    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")
   
    #for careful Adam, if used
    if careful != False:
        (loss_increase_limit, lr_decrease_factor, more_careful) = careful
        prev_train_loss = None
        
        if more_careful == True:
            prev_model_state = None
            prev_opt_state = None
        
    if adaptive != False:
        lr_increase_factor = adaptive
        prev_step = None
        
    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        
        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        
        cost_epoch = total_train_loss / len(train_loader)
        
        ###########################################
        
        # main modified section
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            if s != None:
                print('step scalar:', s)
            
        if lr_scheduler != None:
            lr_scheduler.step()
          
        if careful != False:
            if prev_train_loss != None:
                if cost_epoch > prev_train_loss*(1+loss_increase_limit):
                    print('CAREFUL:\n     loss increase too large')
                    for group in optimizer.param_groups:
                        new_lr = lr_increase_factor*group['lr']
                        group['lr'] = new_lr
                    print("     lr changed to", new_lr)
                    
                    if more_careful == True:
                        model.load_state_dict(prev_model_state)
                        print("MORE CAREFUL:")
                        print('    model state reset')
                        optimizer.load_state_dict(prev_opt_state)
                        print('    optimizer state reset')
                        
            prev_train_loss = cost_epoch
            prev_model_state = model.state_dict()
            prev_opt_state = optimizer.state_dict()
            
        if adaptive != False:
            if prev_step != None:
                dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
                print("ADAPTIVE:")
                print("     dot: %.3f" %dot)
                print('     lr factor: %.3f' %(1+dot*lr_increase_factor))

                for group in optimizer.param_groups:
                    new_lr = (1+dot*lr_increase_factor)*group['lr']
                    group['lr'] = new_lr
                print("     lr changed to", new_lr)
                
            prev_step = []
            for t in optimizer.prev_step:
                prev_step.append(t.clone())
                
            
        ###########################################    
            
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = validate(model, loss, val_loader, device)
        val_epoch = total_val_loss / len(val_loader)

        # Record total time
        time_epoch = time.time() - training_start_time

        # Pretty print a description
        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        # Redirect stdout if needed to avoid clash with progress bar
        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, took {time_epoch:.5} s"
        )
##        write(f"  Validation {cur_val_eff}")

        yield Results(epoch, cost_epoch, val_epoch, time_epoch)

def instrumented_trainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    epoch_start=0,
    ep_optimizer=None,
    lr_scheduler=None,
    careful = False,
    adaptive = False
):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """

    # Print all of the hyperparameters of the training iteration
    if not notebook:
        print("{0:=^80}".format(" HYPERPARAMETERS "))
        print(
            f"""\
n_epochs: {n_epochs}
batch_size: {train_loader.batch_size} events
dataset_train: {train_loader.dataset.tensors[0].size()[0]} events
dataset_val: {val_loader.dataset.tensors[0].size()[0]} events
loss: {loss}
optimizer: {optimizer}
model: {model}"""
        )
        print("=" * 80)

    # Set up notebook or regular progress bar (or none)
    progress = import_progress_bar(notebook)

    # Get the current device
    device = get_device_from_model(model)

    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")

    epoch_iterator = progress(
        range(epoch_start, n_epochs),
        desc="Epochs",
        postfix="train=start, val=start",
        dynamic_ncols=True,
        position=0,
        file=sys.stderr,
    )

    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")

    # stores the outputs from modified Adam
    grads = []
    m1s = []
    m2s = []
    step_sizes = []
    
    #for careful Adam, if used
    if careful != False:
        (loss_increase_limit, lr_decrease_factor, more_careful) = careful
        prev_train_loss = None
        
        if more_careful == True:
            prev_model_state = None
            prev_opt_state = None
        
    if adaptive != False:
        lr_increase_factor = adaptive
        prev_step = None
        
    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        
        # Run the training step
        total_train_loss, opt_outs = instrum_train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        
        grads = grads + opt_outs[0]
        m1s = m1s + opt_outs[1]
        m2s = m2s + opt_outs[2]
        step_sizes.append(opt_outs[3])
        
        cost_epoch = total_train_loss / len(train_loader)
        
        ###########################################
        
        # main modified section
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            if s != None:
                print('step scalar:', s)
            
        if lr_scheduler != None:
            lr_scheduler.step()
          
        if careful != False:
            if prev_train_loss != None:
                if cost_epoch > prev_train_loss*(1+loss_increase_limit):
                    print('CAREFUL:\n     loss increase too large')
                    for group in optimizer.param_groups:
                        new_lr = lr_increase_factor*group['lr']
                        group['lr'] = new_lr
                    print("     lr changed to", new_lr)
                    
                    if more_careful == True:
                        model.load_state_dict(prev_model_state)
                        print("MORE CAREFUL:")
                        print('    model state reset')
                        optimizer.load_state_dict(prev_opt_state)
                        print('    optimizer state reset')
                        
            prev_train_loss = cost_epoch
            prev_model_state = model.state_dict()
            prev_opt_state = optimizer.state_dict()
            
        if adaptive != False:
            if prev_step != None:
                dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
                print("ADAPTIVE:")
                print("     dot: %.3f" %dot)
                print('     lr factor: %.3f' %(1+dot*lr_increase_factor))

                for group in optimizer.param_groups:
                    new_lr = (1+dot*lr_increase_factor)*group['lr']
                    group['lr'] = new_lr
                print("     lr changed to", new_lr)
                
            
        ###########################################    
            
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = validate(model, loss, val_loader, device)
        val_epoch = total_val_loss / len(val_loader)

        # Record total time
        time_epoch = time.time() - training_start_time

        # Pretty print a description
        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        # Redirect stdout if needed to avoid clash with progress bar
        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, took {time_epoch:.5} s"
        )
##        write(f"  Validation {cur_val_eff}")

        yield Results(epoch, cost_epoch, val_epoch, time_epoch), grads, m1s, m2s, step_sizes    
