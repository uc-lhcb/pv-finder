import time
import torch
from collections import namedtuple
import sys
import os
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pandas as pd

from utils.utilities import tqdm_redirect, import_progress_bar, get_device_from_model
from utils.efficiency_utils import ValueSet, eval_efficiency
from utils.plotting_utils import *

Results_cost     = namedtuple("Results_cost",  ["epoch", "cost", "val", "time"])
Results_cost_eff = namedtuple("Results_cost_eff", ["epoch", "cost", "val", "time", "eff_val"])

## ================================================================================================        
def train_model(model, 
                loss, 
                opt, 
                train_data, 
                valid_data, 
                configs,
                outPutFolder_weights, 
                outPutFolder_plots):

    ## ================================================================================
    ## Start with some definitions and settings
    ## ================================================================================
    
    
    ## --------------------------------------------
    ## Define the results panda france and call the 
    ## plotting methods here
    if configs["model_class"]=="GNN-tracks" or configs["model_class"] == "tracks-to-KDE":
        ax, lines = cost_train_plot()
        results = pd.DataFrame([], columns=Results_cost._fields)
    else:
        ax, tax, lax, lines = dual_train_plots()
        results = pd.DataFrame([], columns=Results_cost_eff._fields)

            
    ## --------------------------------------------
    ## Set some plotting options
    fig = ax.figure
    plt.tight_layout()
    
    ## ================================================================================
    ## Loop over epochs for the training
    ## ================================================================================
    for result in trainNet(model, 
                           opt, 
                           loss,
                           train_data, 
                           valid_data,
                           configs, 
                           epoch_start=len(results),
                           notebook=True):

        ## --------------------------------------------
        ## Append the training results after each epoch
        if int(pd.__version__[0])>=2:
            results.loc[len(results)] = pd.Series(result._asdict())
        else:
            results = results.append(pd.Series(result._asdict()), ignore_index=True)
        xs = results.index

        ## --------------------------------------------
        # Update the plot above
        lines['train'].set_data(results.index,results.cost)
        lines['val'].set_data(results.index,results.val)
        if configs["model_class"] == "tracks-to-hist" or configs["model_class"] == "KDE-to-hist":
            replace_in_ax(lax, lines['eff'], xs, results['eff_val'].apply(lambda x: x.eff_rate))
            replace_in_ax(tax, lines['fp'],  xs, results['eff_val'].apply(lambda x: x.fp_rate))
            

        ## --------------------------------------------
        ## Filter first cost epoch (can be really large)
        if min(min(results.cost), min(results.val))>0:
            max_cost = max(max(results.cost if len(results.cost)<2 else results.cost[1:]), max(results.val))
            min_cost = min(min(results.cost), min(results.val))
        else:
            max_cost = max(max(results.cost), max(results.val))
            min_cost = min(min(results.cost), min(results.val))
            
        ## --------------------------------------------
        ## The plot limits need updating too
        if min(min(results.cost), min(results.val))>0:
            ax.set_ylim(min_cost*.9, max_cost*1.1)
        else:
            ax.set_ylim(min_cost*1.1, max_cost*1.1)
            
        ax.set_xlim(-.5, len(results.cost) - .5)

        ## --------------------------------------------
        ## Redraw the figure
        fig.canvas.draw()

        ## --------------------------------------------
        # Save each model state dictionary        
        torch.save(model.state_dict(), outPutFolder_weights / f'{configs["model_type"]}_weights_epoch{result.epoch}.pyt') 

    ## ================================================================================
    # Save final model state dictionary
    final_weights = outPutFolder_weights / f'train_epoch_final.pyt'
    print("")
    print("*"*100)
    print("Saving model weights from final iteration in:")
    print("")
    print(final_weights)
    torch.save(model.state_dict(), final_weights) 
    
    ## ================================================================================
    # Save training plots
    print("")
    print("*"*100)
    print("Saving training plots in:")
    print("")
    print(outPutFolder_plots)
    plt.savefig(str(outPutFolder_plots / f'train_val_cost.png'))
    plt.savefig(str(outPutFolder_plots / f'train_val_cost.pdf'))
    print("")
    print("*"*100)
    
        
## ================================================================================================        
def trainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    configs,    
    *,
    notebook=None,
    epoch_start=0,
):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """

    ## ----------------------------------------------------------------------------------------
    ## Get the number of epochs for the training from the config file
    n_epochs = configs["training_configs"]["n_epochs"]
    
    ## ----------------------------------------------------------------------------------------
    ## Print all of the hyperparameters of the training iteration
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

    ## ----------------------------------------------------------------------------------------
    ## Set up notebook or regular progress bar (or none)
    progress = import_progress_bar(notebook)
    ## ----------------------------------------------------------------------------------------
    ## Get the current device
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


    ## ----------------------------------------------------------------------------------------
    ## Loop for n_epochs
    for epoch in epoch_iterator:
        training_start_time = time.time()

        ## --------------------------------------------------------
        ## Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, configs, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)

        ## --------------------------------------------------------
        ## At the end of the epoch, do a pass on the validation set
        if configs["model_class"]=="GNN-tracks" or configs["model_class"]=="tracks-to-KDE":
            total_val_loss = validate(model, loss, val_loader, device, configs)
        else:
            total_val_loss, iter_val_eff = validate(model, loss, val_loader, device, configs)
            
        val_epoch = total_val_loss / len(val_loader)

        ## --------------------------------------------------------
        ## Record total time
        time_epoch = time.time() - training_start_time

        ## --------------------------------------------------------
        ## Pretty print a description
        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        ## --------------------------------------------------------
        ## Redirect stdout if needed to avoid clash with progress bar
        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, took {time_epoch:.5} s"
        )

        if configs["model_class"]=="GNN-tracks" or configs["model_class"]=="tracks-to-KDE":
            yield Results_cost(epoch, cost_epoch, val_epoch, time_epoch)
        else:
            write(f"  Validation eff: {iter_val_eff}")
            yield Results_cost_eff(epoch, cost_epoch, val_epoch, time_epoch, iter_val_eff)
            

## ================================================================================================        
def train(model, loss, loader, optimizer, device, configs, progress):
    total_loss = 0.0

    ## --------------------------------------------------------
    ## Switch to train mode
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

    if not configs["model_class"]=="GNN-tracks":
        ## --------------------------------------------------------
        ## Get the inputs and labels data
        for inputs, labels in loader:
            if inputs.device != device:
                inputs, labels = inputs.to(device), labels.to(device)

            ## --------------------------------------------------------
            ## Set the parameter gradients to zero
            optimizer.zero_grad()

            ## --------------------------------------------------------
            ## Forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_output = loss(outputs, labels)
            loss_output.backward()
            optimizer.step()

            total_loss += loss_output.data.item()
            if hasattr(loader, "postfix"):
                loader.postfix = f"train={loss_output.data.item():.4g}"
    else:
        if configs["model_type"]=="InteractionGNN_node_PV_position_norm" and configs["models_config"][configs["model_type"]]["loss_penalty"]:
        
            ## --------------------------------------------------------
            ## Get the inputs and labels data
            for data in loader:                        

                loss_batches=0
                nBatches = (data.batch[-1].item()+1)

                for iBatch in range(nBatches):
                    data_iBatch = data.get_example(iBatch)
                    x_iBatch = data_iBatch.x_norm
                    e_iBatch = data_iBatch.edge_index
                    y_iBatch = data_iBatch.y_pos_norm

                    if x_iBatch.device != device:
                        x_iBatch, e_iBatch, y_iBatch = x_iBatch.to(device), e_iBatch.to(device), y_iBatch.to(device)

                    pvs_z_iBatch = torch.tensor(data_iBatch.true_pvs_z)
                    pvs_z_iBatch_norm = (pvs_z_iBatch-data_iBatch.z_min)/(data_iBatch.z_max-data_iBatch.z_min)
                    pvs_z_iBatch_norm = pvs_z_iBatch_norm.to(device)

                    outputs_iBatch = model(x_iBatch,e_iBatch)

                    loss_iBatch = loss.compute_loss_penalty(x_iBatch, outputs_iBatch, y_iBatch, pvs_z_iBatch_norm, device)
                    loss_iBatch.backward()
                    optimizer.step()           

                    loss_batches+=loss_iBatch.data.item()

                loss_output = loss_batches/nBatches
                

                #total_loss += loss_output.data.item()
                total_loss += loss_output
                if hasattr(loader, "postfix"):
                    loader.postfix = f"train={loss_output:.4g}"

        else:
            
            ## --------------------------------------------------------
            ## Get the inputs and labels data
            for data in loader:                        

                x = data.x
                e = data.edge_index
                if configs["model_type"]=="GNN_Interaction":
                    y = data.y
                elif configs["model_type"]=="InteractionGNN_node_position":
                    y = data.y_dist
                elif configs["model_type"]=="InteractionGNN_node_PV_position":
                    y = data.y_pos
                elif configs["model_type"]=="InteractionGNN_node_PV_position_norm":
                    x = data.x_norm

                    if configs["global_configs"][configs["Experiment"]]["addSecondVertex"]:
                        y = data.y_pos_ov_added_norm
                    else:
                        y = data.y_pos_norm

                if x.device != device:
                    x, e, y = x.to(device), e.to(device), y.to(device)

                ## --------------------------------------------------------
                ## Set the parameter gradients to zero
                optimizer.zero_grad()

                ## --------------------------------------------------------
                ## Forward pass, backward pass, optimize
                if configs["model_type"]=="InteractionGNN_node_PV_position_norm":                
                    if configs["models_config"]["InteractionGNN_node_PV_position_norm"]["loss_dist"]:
                        outputs = model(x,e)
                        loss_output = loss.compute_loss_dist(outputs, y)
                        loss_output.backward()
                        optimizer.step()
                    elif configs["models_config"]["InteractionGNN_node_PV_position_norm"]["loss_dist_sigma_norm"]:
                        outputs = model(x,e)
                        loss_output = loss.compute_loss_dist_sigma_norm(x, outputs, y)
                        loss_output.backward()
                        optimizer.step()
                    elif configs["models_config"]["InteractionGNN_node_PV_position_norm"]["loss_dist_sigma_norm_ov_added"]:
                        if not y.shape[1]==6:
                            print("Target shape: %d should be 6!"%(y.shape[1]))
                        else:
                            outputs = model(x,e)
                            loss_output = loss.compute_loss_dist_sigma_norm_ov_added(x, outputs, y)
                            loss_output.backward()
                            optimizer.step()
                    else:
                        print("ERROR !!!!!!!!!!!!!!!!!!!")

                else:
                    outputs = model(x,e)
                    loss_output = loss.compute_loss(outputs, y)
                    loss_output.backward()
                    optimizer.step()

                total_loss += loss_output.data.item()
                if hasattr(loader, "postfix"):
                    loader.postfix = f"train={loss_output.data.item():.4g}"
        
    return total_loss


## ================================================================================================        
def validate(model, loss, val_loader, device, configs):

    ## -----------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## -----------------------------------------------------------------
    ## Get the global configuration 
    global_configs = configs['global_configs'][Exp]
    
    ## --------------------------------------------------------
    ## switch to evaluate mode
    model.eval()

    ## --------------------------------------------------------
    ## Reset total loss and efficiency values 
    total_loss = 0
    eff = ValueSet(0, 0, 0, 0)
    
    ## --------------------------------------------------------
    ## Get the efficiency parameters dictionnary from the config file
    eff_params = configs["efficiency_config"]
    
    if not configs["model_class"]=="GNN-tracks":
        ## --------------------------------------------------------
        ## Get the inputs and labels data
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.device != device:
                    inputs, labels = inputs.to(device), labels.to(device)

                ## --------------------------------------------------------
                ## Forward pass
                val_outputs = model(inputs)
                loss_output = loss(val_outputs, labels)

                total_loss += loss_output.data.item()

            ## --------------------------------------------------------
            ## Depending on the model class either compute efficiency 
            ## and return it together with the total loss...
            if configs["model_class"]=="tracks-to-hist" or configs["model_class"]=="KDE-to-hist":
                ## Switch the model from cuda to cpu
                if 'cuda' in str(device):
                    model.to('cpu')

                ## Get the full validation data (not by batch)            
                inputs  = val_loader.dataset.tensors[0]            
                outputs = model(inputs).cpu().numpy()
                labels  = val_loader.dataset.tensors[1].cpu().numpy()

                ## If doing intervals, reshape the data to be shaped as events
                # Start by computing the number of intervals per event
                nBinsKDE_perEvt           = global_configs["n_bins_poca_kde"]
                nBinsPerInterval_perEvt   = global_configs["nBinsPerInterval"]
                nIntervals_perEvt         = int(nBinsKDE_perEvt/nBinsPerInterval_perEvt)   
                nEvts                     = int(len(inputs)/nIntervals_perEvt)

                if configs["training_configs"]["doIntervals"]:
                    outputs = outputs.reshape((nEvts,nBinsKDE_perEvt))
                    labels  = labels.reshape((nEvts,nBinsKDE_perEvt))

                for iEvt in range(nEvts):                
                    eff += eval_efficiency(labels[iEvt], outputs[iEvt], **eff_params)

                ## Switch the model from cpu back to cuda
                if 'cuda' in str(device):
                    model.to('cuda')

        if configs["model_class"]=="tracks-to-hist" or configs["model_class"]=="KDE-to-hist":
            return total_loss, eff

        elif configs["model_class"]=="tracks-to-KDE":
            ## ...or simply the total loss for tracks-to-KDE model
            return total_loss

        else:
            print("ERROR:: Model class (%s) is not in the current list of available model"%(model_class))
            sys.exit()
    else:
        ## --------------------------------------------------------
        ## Get the inputs and labels data
        with torch.no_grad():
            
            if configs["model_type"]=="InteractionGNN_node_PV_position_norm" and configs["models_config"][configs["model_type"]]["loss_penalty"]:

                ## --------------------------------------------------------
                ## Get the inputs and labels data
                for data in val_loader:                        

                    loss_batches=0
                    nBatches = (data.batch[-1].item()+1)

                    for iBatch in range(nBatches):
                        data_iBatch = data.get_example(iBatch)
                        x_iBatch = data_iBatch.x_norm
                        e_iBatch = data_iBatch.edge_index
                        y_iBatch = data_iBatch.y_pos_norm

                        if x_iBatch.device != device:
                            x_iBatch, e_iBatch, y_iBatch = x_iBatch.to(device), e_iBatch.to(device), y_iBatch.to(device)

                        pvs_z_iBatch = torch.tensor(data_iBatch.true_pvs_z)
                        pvs_z_iBatch_norm = (pvs_z_iBatch-data_iBatch.z_min)/(data_iBatch.z_max-data_iBatch.z_min)
                        pvs_z_iBatch_norm = pvs_z_iBatch_norm.to(device)

                        outputs_iBatch = model(x_iBatch,e_iBatch)

                        loss_iBatch = loss.compute_loss_penalty(x_iBatch, outputs_iBatch, y_iBatch, pvs_z_iBatch_norm, device)
                        #loss_iBatch.backward()
                        #optimizer.step()           

                        loss_batches+=loss_iBatch.data.item()

                    loss_output = loss_batches/nBatches
                    #total_loss += loss_output.data.item()
                    total_loss += loss_output
            else:
                       
                for data in val_loader:
                    x = data.x
                    e = data.edge_index
                    if configs["model_type"]=="GNN_Interaction":
                        y = data.y
                    elif configs["model_type"]=="InteractionGNN_node_position":
                        y = data.y_dist
                    elif configs["model_type"]=="InteractionGNN_node_PV_position":
                        y = data.y_pos
                    elif configs["model_type"]=="InteractionGNN_node_PV_position_norm":
                        x = data.x_norm

                        if configs["global_configs"][configs["Experiment"]]["addSecondVertex"]:
                            y = data.y_pos_ov_added_norm
                        else:
                            y = data.y_pos_norm

                    if x.device != device:
                        x, e, y = x.to(device), e.to(device), y.to(device)

                    ## --------------------------------------------------------
                    ## Forward pass
                    if configs["model_type"]=="InteractionGNN_node_PV_position_norm":
                        if configs["models_config"]["InteractionGNN_node_PV_position"]["loss_dist"]:
                            outputs = model(x,e)
                            loss_output = loss.compute_loss_dist(outputs, y)
                            #loss_output.backward()
                            #optimizer.step()
                            total_loss += loss_output.data.item()
                        elif configs["models_config"]["InteractionGNN_node_PV_position"]["loss_dist_sigma_norm"]:
                            outputs = model(x,e)
                            loss_output = loss.compute_loss_dist_sigma_norm(x, outputs, y)
                            #loss_output.backward()
                            #optimizer.step()
                            total_loss += loss_output.data.item()
                        elif configs["models_config"]["InteractionGNN_node_PV_position"]["loss_dist_sigma_norm_ov_added"]:
                            outputs = model(x,e)
                            loss_output = loss.compute_loss_dist_sigma_norm_ov_added(x, outputs, y)
                            #loss_output.backward()
                            #optimizer.step()
                            total_loss += loss_output.data.item()
                        else:
                            print("ERROR !!!!!!!!!!!!!!!!!!!")

                    else:
                        outputs = model(x,e)
                        loss_output = loss.compute_loss(outputs, y)
                        #loss_output.backward()
                        #optimizer.step()
                        total_loss += loss_output.data.item()
                                        
        return total_loss
