import torch
import torch.optim as optim
from collections import namedtuple
import time
from tqdm import tqdm as progress_bar

Results = namedtuple("Results", ['cost','val','time'])

def trainNet(model, dataset_train, dataset_val, loss, batch_size, n_epochs, *,
             learning_rate=1e-3, name=None):
    
    given_args = locals()
    #Print all of the hyperparameters of the training iteration:
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
    training_start_time = time.time()
    ## array for the cost of every epoch
    cost_epoch=torch.zeros(n_epochs)
    val_epoch=torch.zeros(n_epochs)
    
    print("Number of batches:", len(train_loader))
        
    #Loop for n_epochs
    progress = progress_bar(range(n_epochs))
    for epoch in progress:
        
        total_train_loss = 0
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            #Get inputs
            inputs, labels = data
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            total_train_loss += loss_size.data.item()
                
        every_batch_loss = total_train_loss / len(train_loader)
        progress.set_description(f"Current loss {every_batch_loss:.10f}")
        
        cost_epoch[epoch] = every_batch_loss            
        #At the end of the epoch, do a pass on the validation set
        # validation 
        
        total_val_loss = 0
        for inputs, labels in val_loader:
            #Forward pass
            val_outputs = model(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
            
        val_epoch[epoch] =  total_val_loss / len(val_loader)
        # time so far
        
        if name:
            torch.save(model.state_dict(), f'{name}_{epoch}.pyt')
        
    return Results(cost_epoch.numpy(), val_epoch.numpy(), time.time() - training_start_time)
        
        
 
    