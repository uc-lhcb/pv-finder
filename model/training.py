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
    
    #Get training data
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    #Create our optimizer function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Time for printing
    
    ## array for the cost of every epoch
    cost_epoch = []
    val_epoch  = []
    time_epoch = []
    
    if verbose:
        print("Number of batches:", len(train_loader))
        
    #Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()
        total_train_loss = 0
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            # Get inputs
            inputs, labels = data
            
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # Forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            # Save statistics
            total_train_loss += loss_size.data.item()
                
        every_batch_loss = total_train_loss / len(train_loader)
        
        
        cost_epoch.append(every_batch_loss)
        
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            #Forward pass
            val_outputs = model(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
            
        val_epoch.append(total_val_loss / len(val_loader))
        time_epoch.append(time.time() - training_start_time)
        
        yield Results(cost_epoch, val_epoch, time_epoch, epoch)