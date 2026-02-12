#!/usr/bin/env python3
'''
Methods to prepare the model, including the loss function and the optimizer. 
Having all these steps merged here allows to make sure the ordering between these steps is done properly.
Indeed, a comment on the web at https://pytorch.org/docs/stable/optim.html says

```
If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it. 
Parameters of a model after .cuda() will be different objects with those before the call.

In general, you should make sure that optimized parameters live in consistent locations when 
optimizers are constructed and used.
```

'''

import torch
import yaml
import sys
from models.Loss import Loss, GNN_Loss
from models.models import *

## =================================================
def update_model_dict(model, update_dict_name, verbose=True):

    
    ## Load the current model dictionnary
    model_dict = model.state_dict()
    if verbose:
        print("Current model dictionnary:")
        index = 0
        for k,v in model_dict.items():
            print("index, k =  ",index,"  ",k)
            index = index+1
        print("")
     
    ## Load the input dictionnary
    if verbose:
        print("Updating model dictionnary from:")
        print("  ==> %s"%update_dict_name)
    if torch.cuda.is_available():
        update_dict = torch.load(update_dict_name)
    else:
        update_dict = torch.load(update_dict_name, map_location=torch.device('cpu'))
        
    if verbose:
        print("Updated dictionnary:")
        index = 0
        for k,v in update_dict.items():
            print("index, k =  ",index,"  ",k)
            index = index+1
        print("")
    
        
    ## 1. Filter out unnecessary keys
    update_dict = {k: v for k, v in update_dict.items() if k in model_dict}
    if verbose:
        print("update_dict iterated")
    ## 2. Overwrite entries in the existing state dict
    model_dict.update(update_dict) 
    ## 3. Load the new state dict
    #   need to use strict=False as the two models state model attributes do not agree exactly
    #   see https://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module.load_state_dict
    model.load_state_dict(update_dict,strict=False)
    
    
    return model

## =================================================
def get_optimizer(model_params, learning_rate, optimizer_type):
    
    ## *********************************************
    print("*"*100)
    print("Initializing the Optimizer\n")
    print("")
    print("with the following parameters:\n")
    print("   - Optimizer type  =",optimizer_type)
    print("   - Learning rate   =",learning_rate)
    print("")
    print("*"*100)
    ## *********************************************
    
    ## ------------------------------------------------------
    ## Get the optimizer (default is Adam) but can be changed
    if optimizer_type=="Adam":        
        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    elif optimizer_type=="SGD":
        optimizer = torch.optim.SGD(model_params, lr=learning_rate)    
    else:
        print("Requested optimizer type (%s) not implemented. Please select one from:"%(optimizer_type))
        print("[Adam, SGD]")
        sys.exit()    
    
    return optimizer


## =================================================
def get_loss(configs):
    
    ## From the model class initiate the corresponding model
    if (configs["model_class"]=="tracks-to-KDE" 
        or configs["model_class"]=="tracks-to-hist" 
        or configs["model_class"]=="KDE-to-hist"):
        loss = Loss(configs)

    elif configs["model_class"]=="GNN-tracks":
        loss = GNN_Loss(configs)
    
    return loss

## =================================================
def get_model(configs):
    
    ## Get the model class from the configuration file
    model_class = configs["model_class"]

    ## Get the model type (which architecture to be used)
    model_type = configs["model_type"]        
    
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## Get the global configuration 
    global_configs = configs['global_configs'][Exp]
        
    ## From the model class initiate the corresponding model
    if model_class=="tracks-to-KDE":
        ## Get the model configuration corresponding to the model type
        model_configs = configs["models_config"][model_type]
        
        ## Now define the model
        if model_type=="FCN6L":
            model = FCN6L(n_InputFeatures  = model_configs["n_InputFeatures"],
                          n_OutputFeatures = global_configs["nBinsPerInterval"],
                          l_HiddenNodes    = model_configs["l_HiddenNodes"],
                          LeakyReLU_param  = model_configs["LeakyReLU_param"],
                          predScaleFactor  = model_configs["predScaleFactor"],
                          maskVal          = global_configs["tracks_default_POCA"],
                         )
        elif model_type=="FCN6L_UNet":
            model = FCN6L_UNet(n_InputFeatures  = model_configs["n_InputFeatures"],
                               n_OutputFeatures = global_configs["nBinsPerInterval"],
                               l_HiddenNodes    = model_configs["l_HiddenNodes"],
                               LeakyReLU_param  = model_configs["LeakyReLU_param"],
                               predScaleFactor  = model_configs["predScaleFactor"],
                               maskVal          = global_configs["tracks_default_POCA"],
                              )            
        return model
    
    elif model_class=="tracks-to-hist":        
        ## Get the model configuration corresponding to the model type
        model_configs = configs["models_config"][model_type]
        
        model = FCN6L_UNet(n_InputFeatures  = model_configs["n_InputFeatures"],
                           n_OutputFeatures = global_configs["nBinsPerInterval"],
                           l_HiddenNodes    = model_configs["l_HiddenNodes"],
                           LeakyReLU_param  = model_configs["LeakyReLU_param"],
                           predScaleFactor  = model_configs["predScaleFactor"],
                           maskVal          = global_configs["tracks_default_POCA"],
                          )            
        return model
    
    elif model_class=="KDE-to-hist":
        model = KDE_to_hist_model
        return model

    elif model_class=="GNN-tracks" and model_type=="GNN_Interaction":
        model = InteractionGNN(configs["models_config"]["GNN_Interaction"])
        return model

    elif model_class=="GNN-tracks" and model_type=="InteractionGNN_node_position":
        model = InteractionGNN_node_position(configs["models_config"]["InteractionGNN_node_position"])
        return model
    elif model_class=="GNN-tracks" and model_type=="InteractionGNN_node_PV_position":
        model = InteractionGNN_node_PV_position(configs["models_config"]["InteractionGNN_node_PV_position"])
        return model
    elif model_class=="GNN-tracks" and model_type=="InteractionGNN_node_PV_position_norm":
        model = InteractionGNN_node_PV_position_norm(configs["models_config"]["InteractionGNN_node_PV_position_norm"])
        return model
    
    else:
        print("Requested model class (%s) not implemented. Please select one from:"%(model_class))
        print("[tracks-to-KDE, tracks-to-hist, KDE-to-hist]")
        sys.exit()    
        

## =================================================    
def prepare_model(configs,
                  device=None):
    
    ## ---------------------------------------------------------------------------
    ## Get the model corresponding to the considered configuration
    model = get_model(configs)
    if not device==None:
        model = model.to(device)
    ## ---------------------------------------------------------------------------
    ## Get the optimizer 
    optimizer = get_optimizer(model.parameters(), 
                              configs["training_configs"]["learning_rate"], 
                              configs["training_configs"]["optimizer"])    
        
    ## ---------------------------------------------------------------------------
    ## Get the Loss function (automatically build from the configuration file parameters)    
    loss = get_loss(configs)
        
    return model, loss, optimizer