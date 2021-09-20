import time
import torch
from collections import namedtuple
import sys
import os

import numpy as np

from model.utilities import tqdm_redirect, import_progress_bar, get_device_from_model
from model.efficiency import efficiency, ValueSet

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
    step_schedule=None,
    entropy = False
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


    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        if entropy != True:
            # Run the training step
            total_train_loss = train(
                model, loss, train_loader, optimizer, device, progress=progress
            )
        
        if entropy == True:
            total_train_loss = entropy_train(
                 model, loss, train_loader, optimizer, device, progress=progress
            )
        
        cost_epoch = total_train_loss / len(train_loader)
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None:
            step_schedule.step()

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

def gradnorm_trainNet(
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
    step_schedule=None
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


    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        # Run the training step
        total_train_loss, grad_norms = gradnorm_train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        
        cost_epoch = total_train_loss / len(train_loader)
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None:
            step_schedule.step()

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

        yield Results(epoch, cost_epoch, val_epoch, time_epoch), grad_norms    
    
def carefultrainNet(
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
    step_schedule=None, 
    loss_increase_limit = 0.05, 
    lr_factor = 0.7
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

    prev_train_loss = 1e6
    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large')
            #optimizer.backstep()
            for group in optimizer.param_groups:
                new_lr = lr_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
#         else:
#             prev_train_loss = cost_epoch
        prev_train_loss = cost_epoch
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def morecarefultrainNet(
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
    step_schedule=None, 
    loss_increase_limit = 0.05, 
    lr_factor = 0.7
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
    
    prev_opt_state = None
    prev_model_state = None
    prev_train_loss = 1e6
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large')
            #optimizer.backstep()
            model.load_state_dict(prev_model_state)
            print('model state reset')
            optimizer.load_state_dict(prev_opt_state)
            print('optimizer state reset')
            for group in optimizer.param_groups:
                new_lr = lr_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
#         else:
#             prev_train_loss = cost_epoch
        prev_train_loss = cost_epoch
        prev_model_state = model.state_dict()
        prev_opt_state = optimizer.state_dict()
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def fixedcarefultrainNet(
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
    step_schedule=None, 
    loss_increase_limit = 0.02, 
    lr_factor = 0.8
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

    prev_train_loss = 1e6
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        optimizer.clear_prev_step()
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large; model backstepping')
            optimizer.backstep()
            for group in optimizer.param_groups:
                new_lr = lr_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
        else:
            prev_train_loss = cost_epoch
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def step_norm(tens_list):
    #print(sum([torch.sum(torch.square(t)) for t in tens_list]), 'vs', sum([torch.sum(torch.mul(tens_list[i], tens_list[i])) for i in range(len(tens_list))]))
    return torch.sqrt(sum([torch.sum(torch.square(t)) for t in tens_list]))

def normed_dot(list_a, list_b):
    dot = sum([torch.sum(torch.mul(list_a[i], list_b[i])) for i in range(len(list_a))])
    print('num:', dot)
    print('denom:', (step_norm(list_a)*step_norm(list_b)))
    return dot/(step_norm(list_a)*step_norm(list_b))
   
def epochsteptrainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    device = None,
    epoch_start=0,
    ep_optimizer=None,
    step_schedule=None
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
    if device == None:
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

    prev_train_loss = 1e6
    prev_step = None
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        #clears accumulated steps from previous epoch
        #optimizer.clear_prev_step()
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        ###########################################################
        
        #checks angle of current step to prev_step, and adjusts lr
        if prev_step != None:
            #print([torch.sum(prev_step[i]-optimizer.prev_step[i]) for i in range(len(prev_step))])
            dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
            print("dot:", dot)
            
            if ep_optimizer != None:
                ep_optimizer.step(dot)
            
        prev_step = []
        for t in optimizer.prev_step:
            prev_step.append(t.clone())
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

        ###########################################################
            
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
    
def adaptivetrainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    device = None,
    epoch_start=0,
    ep_optimizer=None,
    step_schedule=None, 
    loss_increase_limit = 0.02, 
    lr_factor = 0.8, 
    lr_increase_factor = 0.05
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
    if device == None:
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

    prev_train_loss = 1e6
    prev_step = None
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        #clears accumulated steps from previous epoch
        #optimizer.clear_prev_step()
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
#         #checks for cost increase to backstep and lower lr if necessary
#         write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
#         if cost_epoch > prev_train_loss*(1+loss_increase_limit):
#             print('loss increase too large; model backstepping')
#             #optimizer.backstep()
#             for group in optimizer.param_groups:
#                 new_lr = lr_factor*group['lr']
#                 group['lr'] = new_lr
#             print("lr changed to", new_lr)
        
#         else:
#             prev_train_loss = cost_epoch
                
        #########################################################################
    
        #checks angle of current step to prev_step, and adjusts lr
        if prev_step != None:
            #print([torch.sum(prev_step[i]-optimizer.prev_step[i]) for i in range(len(prev_step))])
            dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
            print("dot:", dot, '| ratio:', (1+dot*lr_increase_factor))
            #print("dot:", dot, '| ratio:', (1+(2*lr_increase_factor)*(dot-0.5)))
            
            for group in optimizer.param_groups:
                new_lr = (1+dot*lr_increase_factor)*group['lr']
                #new_lr = (1+(2*lr_increase_factor)*(dot-0.5))*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
            
        prev_step = []
        for t in optimizer.prev_step:
            prev_step.append(t.clone())
            
        ###############################################################################
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def adaptivecarefultrainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    device = None,
    epoch_start=0,
    ep_optimizer=None,
    step_schedule=None, 
    loss_increase_limit = 0.02, 
    lr_factor = 0.8, 
    lr_increase_factor = 0.10
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
    if device == None:
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

    prev_train_loss = 1e6
    prev_step = None
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        #clears accumulated steps from previous epoch
        #optimizer.clear_prev_step()
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        #########################################################################
        
        #careful part
        
        #checks for cost increase to backstep and lower lr if necessary
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large; model backstepping')
            #optimizer.backstep()
            for group in optimizer.param_groups:
                new_lr = lr_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
        prev_train_loss = cost_epoch
                
        #########################################################################
    
        #adaptive part
        
        #checks angle of current step to prev_step, and adjusts lr
        if prev_step != None:
            #print([torch.sum(prev_step[i]-optimizer.prev_step[i]) for i in range(len(prev_step))])
            dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
            print("dot:", dot, '| ratio:', (1+dot*lr_increase_factor))
            #print("dot:", dot, '| ratio:', (1+(2*lr_increase_factor)*(dot-0.5)))
            
            for group in optimizer.param_groups:
                new_lr = (1+dot*lr_increase_factor)*group['lr']
                #new_lr = (1+(2*lr_increase_factor)*(dot-0.5))*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
            
        prev_step = []
        for t in optimizer.prev_step:
            prev_step.append(t.clone())
            
        ###############################################################################
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def adaptivemorecarefultrainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    device = None,
    epoch_start=0,
    ep_optimizer=None,
    step_schedule=None, 
    loss_increase_limit = 0.02, 
    lr_factor = 0.8, 
    lr_increase_factor = 0.10
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
    if device == None:
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

    prev_train_loss = 1e6
    prev_step = None
    prev_model_state = model.state_dict()
    prev_opt_state = optimizer.state_dict()
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        #clears accumulated steps from previous epoch
        #optimizer.clear_prev_step()
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        #########################################################################
        
        #more careful part
        
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large')
            #optimizer.backstep()
            model.load_state_dict(prev_model_state)
            print('model state reset')
            optimizer.load_state_dict(prev_opt_state)
            print('optimizer state reset')
            for group in optimizer.param_groups:
                new_lr = lr_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
#         else:
#             prev_train_loss = cost_epoch
        prev_train_loss = cost_epoch
        prev_model_state = model.state_dict()
        prev_opt_state = optimizer.state_dict()
                
        #########################################################################
    
        #adaptive part
        
        #checks angle of current step to prev_step, and adjusts lr
        if prev_step != None:
            #print([torch.sum(prev_step[i]-optimizer.prev_step[i]) for i in range(len(prev_step))])
            dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
            print("dot:", dot, '| ratio:', (1+dot*lr_increase_factor))
            #print("dot:", dot, '| ratio:', (1+(2*lr_increase_factor)*(dot-0.5)))
            
            for group in optimizer.param_groups:
                new_lr = (1+dot*lr_increase_factor)*group['lr']
                #new_lr = (1+(2*lr_increase_factor)*(dot-0.5))*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
            
        prev_step = []
        for t in optimizer.prev_step:
            prev_step.append(t.clone())
            
        ###############################################################################
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def adaptivetrainNet2(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    device = None,
    epoch_start=0,
    ep_optimizer=None,
    step_schedule=None, 
    loss_increase_limit = 0.02, 
    lr_factor = 0.8, 
    lr_increase_factor = 0.05
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
    if device == None:
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

    prev_train_loss = 1e6
    prev_step = None
    # Loop for n_epochs
    for epoch in epoch_iterator:
        
        #clears accumulated steps from previous epoch
        optimizer.clear_prev_step()
        
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        #checks for cost increase to backstep and lower lr if necessary
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large; model backstepping')
            optimizer.backstep()
            for group in optimizer.param_groups:
                new_lr = lr_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
        else:
            prev_train_loss = cost_epoch
            
        #checks angle of current step to prev_step, and adjusts lr
        if prev_step != None:
            #print([torch.sum(prev_step[i]-optimizer.prev_step[i]) for i in range(len(prev_step))])
            dot = normed_dot(prev_step, optimizer.prev_step).cpu().numpy()
            #print("dot:", dot, '| ratio:', (1+dot*lr_increase_factor))
            print("dot:", dot, '| ratio:', (1+(2*lr_increase_factor)*(dot-0.5)))
            
            for group in optimizer.param_groups:
                #new_lr = (1+dot*lr_increase_factor)*group['lr']
                new_lr = (1+(2*lr_increase_factor)*(dot-0.5))*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        prev_step = []
        for t in optimizer.prev_step:
            prev_step.append(t.clone())
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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
    
def lesscarefultrainNet(
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
    step_schedule=None, 
    loss_increase_limit = 0.05, 
    lr_decrease_factor = 0.7,
    lr_increase_factor = 0.05
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

    prev_train_loss = 1e6
    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)
        
        write = getattr(progress, "write", print)
        
        write('new cost:', cost_epoch, '| old cost:', prev_train_loss)
        if cost_epoch > prev_train_loss*(1+loss_increase_limit):
            print('loss increase too large; model backstepping')
            optimizer.backstep()
            for group in optimizer.param_groups:
                new_lr = lr_decrease_factor*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)
        
        else:
            for group in optimizer.param_groups:
                new_lr = (1+lr_increase_factor)*group['lr']
                group['lr'] = new_lr
            print("lr changed to", new_lr)    
            prev_train_loss = cost_epoch
        
        if ep_optimizer != None:
            s = ep_optimizer.step()
            print('step scalar:', s)
            
        if step_schedule != None and cost_epoch <= prev_train_loss*(1+loss_increase_limit):
            step_schedule.step()

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

def gradnorm_train(model, loss, loader, optimizer, device, progress):
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
    
    grad_norms = []

    for inputs, labels in loader:
        if inputs.device != device:
            inputs, labels = inputs.to(device), labels.to(device)

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = model(inputs)
        loss_output = loss(outputs, labels)
        loss_output.backward()
        outs = optimizer.step()
        this_grad_norms = outs[0]

        grad_norms.append(this_grad_norms)
        
        total_loss += loss_output.data.item()
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss_output.data.item():.4g}"
    return total_loss, grad_norms

def entropy_train(model, loss, dataloader, optimizer, device, progress):
    total_loss = 0.0

    # switch to train mode
    model.train()

    iter_loader =  iter(dataloader)
    
    loader = progress(
        dataloader,
        postfix="train=start",
        desc="Training",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )

#     for x,y in dataloader:
#         def helper():
#             def feval():
# #                 x,y = next(iter_loader)
#                 if x.device != device:
#                     x,y = x.to(device), y.to(device)

#                 optimizer.zero_grad()
#                 yh = model(x)
#                 f = loss.forward(yh, y)
#                 f.backward()
                
#                 return f.data.item()
#             return feval
            
    for i, (x,y) in enumerate(dataloader):
        if i%100 == 0:
            print(i, '/', len(dataloader))
        def helper(x, y):
            def feval(x=x, y=y):
                if x.device != device:
                    x,y = x.to(device), y.to(device)

                optimizer.zero_grad()
                yh = model(x)
                f = loss.forward(yh, y)
                f.backward()
                
                return f.data.item()
            return feval

        f = optimizer.step(helper(x, y), model, loss)
        
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={f:.4g}"
        
        total_loss += f
        
    return total_loss

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
