import torch
import torch.optim as optim
from collections import namedtuple
import time

Results = namedtuple("Results", ['cost','val','time','epoch'])

def trainNet(model, dataset_train, dataset_val, loss, batch_size, epoch_iterator, *,
             learning_rate=1e-3, verbose=True):
    
    # Print all of the hyperparameters of the training iteration
    if verbose:
        given_args = locals()
        print("{0:=^80}".format(" HYPERPARAMETERS "))
        for item in given_args:
            print(item, "=", given_args[item])
        print("="*80)

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
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, pin_memory=not device_matches)
    
    # Create our optimizer function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists for the cost of every epoch
    cost_epoch = []
    val_epoch  = []
    time_epoch = []
    
    if verbose:
        print("Number of batches:", len(train_loader))
        
    # Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()
        
        total_train_loss = train(model, loss, train_loader, optimizer, device)
        cost_epoch.append(total_train_loss / len(train_loader))
        
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = validate(model, loss, val_loader, device)
        val_epoch.append(total_val_loss / len(val_loader))
        
        time_epoch.append(time.time() - training_start_time)
        
        yield Results(cost_epoch, val_epoch, time_epoch, epoch)


def train(model, loss, loader, optimizer, device):
    total_loss = 0.0
    
    # switch to train mode
    model.train()
    
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