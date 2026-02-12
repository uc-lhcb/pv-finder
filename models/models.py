#!/usr/bin/env python3
'''
File containing all the models definitions. 
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

from functools import partial

#from __future__ import annotations
import typing
from typing import Union

from torch_scatter import scatter_add, scatter_max
from torch.utils.checkpoint import checkpoint



## ==============================================================================    
## ==============================================================================    
## ==============================================================================    
class Conv_BN_Relu(nn.Sequential):
    """convolution => BN => ReLU
    This class simply combines a few layers into a "block", which will be very commonly used throughout multiple different models.
    You can specify the parameters of the conv layer when you initialize, and the rest will be automatically sorted out.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0):
        super(Conv_BN_Relu, self).__init__(
              nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
              nn.BatchNorm1d(out_channels),
              nn.ReLU(),
              nn.Dropout(dropout)
              # Swish_module(),
        )

## ==============================================================================    
## ==============================================================================    
## ==============================================================================    
class Up(nn.Sequential):
    """transpose convolution => convolution => BN => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            Conv_BN_Relu(out_channels, out_channels, kernel_size=kernel_size, dropout=dropout)
        )
    
## =====a=========================================================================    
## ==============================================================================    
## ==============================================================================    
def combine(x, y, mode='concat'):
    if mode == 'concat':
        return torch.cat([x, y], dim=1)
    elif mode == 'add':
        return x+y
    else:
        raise RuntimeError(f'''Invalid option {mode} from choices 'concat' or 'add' ''')

## ======================================================================
## These functions were defined so that you can specify components of a 
## U-Net network without importing new objects/classes. With these, you
## can just use strings to specify what you want. This would be useful
## if we ever do a grid search over different model architectures, or move 
## to AWS where we want to configure architecture without writing any new code.
## ======================================================================
## --------------------------------------------------------
downsample_options = {
    'Conv_BN_Relu':Conv_BN_Relu,
}
## --------------------------------------------------------
upsample_options = {
    'Up':Up,
}
## --------------------------------------------------------

## ==============================================================================    
## ==============================================================================    
## ==============================================================================    
class FCN6L(nn.Module):
    
    ## --------------------------------------------------------
    ## --------------------------------------------------------
    ## Activation function to be applied to the output layer
    softplus = torch.nn.Softplus()

    ## --------------------------------------------------------
    ## --------------------------------------------------------
    def __init__(self, 
                 n_InputFeatures  = 9, 
                 n_OutputFeatures = 100, 
                 l_HiddenNodes    = [25,25,25,25,25], 
                 LeakyReLU_param  = 0.01,
                 predScaleFactor  = 0.001,
                 maskVal          = -99.,
                 verbose          = False
                ):
        super(FCN6L,self).__init__()

        self.verbose = verbose
        
        ## *********************************************
        print("*"*100)
        print("Initializing the FCN6L model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =",n_InputFeatures)
        print("   - n_OutputFeatures =",n_OutputFeatures)
        print("   - l_HiddenNodes    =",l_HiddenNodes)
        print("   - LeakyReLU_param  =",LeakyReLU_param)
        print("   - maskVal          =",maskVal)
        print("   - predScaleFactor  =",predScaleFactor)
        print("")
        print("*"*100)
        ## *********************************************
        
        ## --------------------------------------------------------
        self.n_InputFeatures  = n_InputFeatures
        self.n_OutputFeatures = n_OutputFeatures
        self.l_HiddenNodes    = l_HiddenNodes
        self.LeakyReLU_param  = LeakyReLU_param
        self.maskVal          = maskVal
        self.predScaleFactor  = predScaleFactor
        
        ## --------------------------------------------------------
        self.Nodes_L1 = l_HiddenNodes[0]
        self.Nodes_L2 = l_HiddenNodes[1]
        self.Nodes_L3 = l_HiddenNodes[2]
        self.Nodes_L4 = l_HiddenNodes[3]
        self.Nodes_L5 = l_HiddenNodes[4]
       
        ## --------------------------------------------------------
        self.layer1 = nn.Linear(
                                in_features = n_InputFeatures,
                                out_features = self.Nodes_L1,
                                bias = True)
        self.layer2 = nn.Linear(
                                in_features = self.layer1.out_features,
                                out_features = self.Nodes_L2,
                                bias = True)
        self.layer3 = nn.Linear(
                                in_features = self.layer2.out_features,
                                out_features = self.Nodes_L3,
                                bias = True)
        self.layer4 = nn.Linear(
                                in_features = self.layer3.out_features,
                                out_features = self.Nodes_L4,
                                bias = True)
        self.layer5 = nn.Linear(
                                in_features = self.layer4.out_features,
                                out_features = self.Nodes_L5,
                                bias = True)

        ## --------------------------------------------------------
        # Output layer defined to have nBinsPerInterval output
        self.layer6 = nn.Linear(
                                in_features = self.layer5.out_features,
                                out_features = self.n_OutputFeatures,
                                bias = True)
                
    ## --------------------------------------------------------
    ## --------------------------------------------------------
    def forward(self, x):

        ## --------------------------------------------------------
        ## Activation function to be applied between each hidden layer
        leaky = nn.LeakyReLU(self.LeakyReLU_param)

        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]

        ## --------------------------------------------------------
        ## Construct masking from the input tracks data to allow 
        ## filtering only entries with tracks
        mask = x[:,0,:] > (self.maskVal+1)
        
        ## --------------------------------------------------------
        ## Construct filter
        filt = mask.float()
        
        ## --------------------------------------------------------
        f1 = filt.unsqueeze(2)
        
        ## --------------------------------------------------------
        f2 = f1.expand(-1,-1,self.n_OutputFeatures)
        # print("filt.shape = ",filt.shape)
        # print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
        # print("after transpose, x.shape = ", x.shape)
      
        ## --------------------------------------------------------
        ## Start forward pass here 
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        ## --------------------------------------------------------
        ## Output layer
        x = (self.layer6(x))        
        x = self.softplus(x)
        # print("after softplus, x.shape = ",x.shape)
       
        ## --------------------------------------------------------
        x.view(nEvts,-1,self.n_OutputFeatures)

        ## --------------------------------------------------------
        ## Apply masking
        x1 = torch.mul(f2,x)
        # print("x1.shape = ",x1.shape)

        ## --------------------------------------------------------
        x1.view(nEvts,-1,self.n_OutputFeatures)
        ## --------------------------------------------------------
        ## Sum contributions from all tracks to the output (KDE here)
        y_prime = torch.sum(x1,dim=1)


        ##        print("y_prime.shape = ",y_prime.shape)

        ##        print("y_pred[:,0:10] =  ",y_pred[:,0:10])
        ##        print("y_prime[:,0:2] =  ",y_prime[:,0:10])
        
        ## --------------------------------------------------------
        ## Return prediction after scaling by predScaleFactor, which
        ## is meant to scale back values in a "reasonnable range", 
        ## i.e. close to unity!
        y_pred = torch.mul(y_prime,self.predScaleFactor)
        
        return y_pred

## ==============================================================================    
## ==============================================================================    
## ==============================================================================    
class FCN6L_UNet(nn.Module):
    
    ## Activation function to be applied to the output layer
    softplus = torch.nn.Softplus()

    def __init__(self, 
                 n_InputFeatures  = 9, 
                 n_OutputFeatures = 100, 
                 l_HiddenNodes    = [20,20,20,20,20], 
                 n_LatentChannels = 8,
                 n_UNetChannels   = 64,
                 sc_mode          = "concat",
                 dropout          = 0.25,
                 LeakyReLU_param  = 0.01,
                 predScaleFactor  = 0.001,
                 maskVal          = -99.,
                 d_selection      = 'Conv_BN_Relu',
                 u_selection      = 'Up',
                 verbose          = False
                ):
        super(FCN6L_UNet,self).__init__()
              
        ## *********************************************
        print("*"*100)
        print("Initializing the FCN6L_UNet model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =",n_InputFeatures)
        print("   - n_OutputFeatures =",n_OutputFeatures)
        print("   - l_HiddenNodes    =",l_HiddenNodes)
        print("   - n_LatentChannels =",n_LatentChannels)
        print("   - n_UNetChannels   =",n_UNetChannels)
        print("   - d_selection      =",d_selection)
        print("   - u_selection      =",u_selection)
        print("   - sc_mode          =",sc_mode)
        print("   - dropout          =",dropout)
        print("   - LeakyReLU_param  =",LeakyReLU_param)
        print("   - maskVal          =",maskVal)
        print("   - predScaleFactor  =",predScaleFactor)
        print("")
        print("*"*100)
        ## *********************************************
        
        ## --------------------------------------------------------
        self.n_InputFeatures  = n_InputFeatures
        self.n_OutputFeatures = n_OutputFeatures
        self.n_LatentChannels = n_LatentChannels
        self.n_UNetChannels   = n_UNetChannels
        self.mode             = sc_mode
        self.dropout          = dropout
        self.LeakyReLU_param  = LeakyReLU_param
        self.maskVal          = maskVal
        self.predScaleFactor  = predScaleFactor
        
        ## --------------------------------------------------------
        self.Nodes_L1 = l_HiddenNodes[0]
        self.Nodes_L2 = l_HiddenNodes[1]
        self.Nodes_L3 = l_HiddenNodes[2]
        self.Nodes_L4 = l_HiddenNodes[3]
        self.Nodes_L5 = l_HiddenNodes[4]
        
        ## --------------------------------------------------------
        self.verbose = verbose
        
        # ========================================================================
        # Fully Connected part of the network
        # ========================================================================
                 
       
        ## --------------------------------------------------------
        self.layer1 = nn.Linear(
                                in_features = n_InputFeatures,
                                out_features = self.Nodes_L1,
                                bias = True)
        self.layer2 = nn.Linear(
                                in_features = self.layer1.out_features,
                                out_features = self.Nodes_L2,
                                bias = True)
        self.layer3 = nn.Linear(
                                in_features = self.layer2.out_features,
                                out_features = self.Nodes_L3,
                                bias = True)
        self.layer4 = nn.Linear(
                                in_features = self.layer3.out_features,
                                out_features = self.Nodes_L4,
                                bias = True)
        self.layer5 = nn.Linear(
                                in_features = self.layer4.out_features,
                                out_features = self.Nodes_L5,
                                bias = True)
        ## --------------------------------------------------------
        # Output layer defined to have nBinsPerInterval output
        self.layer6A = nn.Linear(
                                 in_features = self.layer5.out_features,
                                 out_features = self.n_LatentChannels*self.n_OutputFeatures,
                                 bias = True)

        
        ## ========================================================================
        ## UNet part of the network
        ## ========================================================================

        ## -----------------------------------------------------------------------
        ## General definitions        
        self.relu = nn.ReLU()

        if self.mode == 'concat':
            self.factor = 2
        else:
            self.factor = 1
            
        ## --------------------------------------------------------------------------------
        ## Make sure that if we configure the architecture using the strings, that the string is a valid choice
        assert d_selection in downsample_options.keys(), f'Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}'
        assert u_selection in upsample_options.keys(), f'Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}'

        ## --------------------------------------------------------------------------------
        ## Selection of the main component that will be use in the decoder/encoder
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]                                      
            
        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Down Block 0 -> 1 (receiving input of shape [n_LatentChannels*n_OutputFeatures])
        self.rcbn1 = d_block(self.n_LatentChannels, self.n_UNetChannels, kernel_size = 25, dropout=self.dropout)
        ## --------------------------------------------------------------------------------
        ## Down Block 1 -> 2
        self.rcbn2 = d_block(self.n_UNetChannels, self.n_UNetChannels, kernel_size = 7,  dropout=self.dropout)
        ## --------------------------------------------------------------------------------
        ## Down Block 2 -> 3
        self.rcbn3 = d_block(self.n_UNetChannels, self.n_UNetChannels, kernel_size = 5,  dropout=self.dropout)

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 3 -> 2'
        self.up1 = u_block(self.n_UNetChannels, self.n_UNetChannels, kernel_size = 5, dropout=self.dropout)
        ## --------------------------------------------------------------------------------
        ## Up Block 2' -> 1'
        self.up2 = u_block(self.n_UNetChannels*self.factor, self.n_UNetChannels, kernel_size = 5, dropout=self.dropout)

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 1' -> 0'
        self.out_intermediate = nn.Conv1d(self.n_UNetChannels*self.factor, self.n_UNetChannels, 5, padding=2)
        ## --------------------------------------------------------------------------------
        ## Up Block 0' -> output
        ##
        ## We need to project the n-dimensional output channels down to one, 
        ## so we can call ".squeeze()" to remove it        
        self.outc = nn.Conv1d(self.n_UNetChannels, 1, 5, padding=2) 

        ## --------------------------------------------------------------------------------
        self.maxPool1d = nn.MaxPool1d(2)            

    ## --------------------------------------------------------
    ## --------------------------------------------------------
    def forward(self, x):
        
        ## ====================================================
        ##  Forward pass of the Fully connected layers
        ## ====================================================
        
        ## --------------------------------------------------------
        ## Activation function to be applied between each hidden layer
        leakyRL = nn.LeakyReLU(self.LeakyReLU_param)
        
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]

        ## --------------------------------------------------------
        ## Construct masking from the input tracks data to allow 
        ## filtering only entries with tracks
        mask = x[:,0,:] > (self.maskVal+1)
        
        ## --------------------------------------------------------
        ## Construct filter
        filt = mask.float()
        
        ## --------------------------------------------------------
        f1 = filt.unsqueeze(2)
        
        ## --------------------------------------------------------
        f2 = f1.expand(-1,-1,self.n_OutputFeatures)
        # print("filt.shape = ",filt.shape)
        # print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
        # print("after transpose, x.shape = ", x.shape)
            
        ## --------------------------------------------------------
        ## make a copy of the initial features so they can be passed along using a skip connection 
        x0 = x 
        x = leakyRL(self.layer1(x))
        x = leakyRL(self.layer2(x))
        x = leakyRL(self.layer3(x))
        x = leakyRL(self.layer4(x))
        x = leakyRL(self.layer5(x))
        ## --------------------------------------------------------
        ## produces n_LatentChannels x nBins bins for intervals
        x = leakyRL(self.layer6A(x))
        
        ## --------------------------------------------------------
        x = x.view(nEvts,nTrks,self.n_LatentChannels,self.n_OutputFeatures)

        ## --------------------------------------------------------
        ## here we are summing over all the tracks, creating "y"
        ## which has a sum of all tracks' contributions in each of
        ## n_LatentChannels for each event and each bin of the (eventual)
        ## KDE histogram
        ## print("before unsqueezing, f2.shape = ",f2.shape)
        f2 = torch.unsqueeze(f2,2)
        # print("x.shape = ",x.shape)
        # print("after unsqueezing,  f2 = torch.unsqueeze(f2,2), f2,shape = ",f2.shape)
        x = torch.mul(f2,x)
        outputFCN = torch.sum(x,dim=1)
        # print(' after summation: y0.shape = ',y0.shape)

        ## ====================================================
        ##  Forward pass of the UNet layers
        ## ====================================================

        ## --------------------------------------------------------
        xd1 = self.rcbn1(outputFCN)  # n_OutputFeatures
        ## --------------------------------------------------------
        xd2 = self.rcbn2(xd1)        # n_OutputFeatures
        xd2 = self.maxPool1d(xd2)    # n_OutputFeatures / 2
        ## --------------------------------------------------------
        xd3 = self.rcbn3(xd2)        # n_OutputFeatures / 2
        xd3 = self.maxPool1d(xd3)    # n_OutputFeatures / 4

        ## --------------------------------------------------------
        xu1 = self.up1(xd3)          # n_OutputFeatures / 2
        # Add a skip connection using "combine"
        xu1_skip = combine(xu1, xd2, 
                           mode=self.mode)
        ## --------------------------------------------------------
        xu2 = self.up2(xu1_skip)     # n_OutputFeatures
        # Add a skip connection using "combine"
        xu2_skip = combine(xu2, xd1, 
                           mode=self.mode)

        ## --------------------------------------------------------
        # Make an intermediate Conv layer
        x   = self.out_intermediate(xu2_skip) # n_OutputFeatures
        ## --------------------------------------------------------
        # Make final Conv layer
        logits_x = self.outc(x)

        ## --------------------------------------------------------
        # squeeze removes empty dimensions.. (n_UNetChannels, 1, n_OutputFeatures) -> (n_UNetChannels, n_OutputFeatures)
        outputs = F.softplus(logits_x).squeeze() 
        
        ## --------------------------------------------------------
        ## Return prediction after scaling by predScaleFactor, which
        ## is meant to scale back values in a "reasonnable range", 
        ## i.e. close to unity!
        y_pred = torch.mul(outputs,self.predScaleFactor)
        return y_pred
    

## ==============================================================================    
## ==============================================================================    
def make_mlp(
    input_size: int,
    sizes: typing.List[int],
    hidden_activation: str = "ReLU",
    output_activation: Union[str, None] = "ReLU",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Construct an MLP with specified fully-connected layers."""
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(getattr(nn, hidden_activation)())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(getattr(nn, output_activation)())
    return nn.Sequential(*layers)


## ==============================================================================    
## ==============================================================================    
class InteractionGNN(nn.Module):
    
    """An interaction network class"""
    
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def __init__(self, hparams):
        """
        Initialise the Lightning Module that can scan over different GNN training
        regimes
        """
        super(InteractionGNN,self).__init__()
        
        # ============================================================    
        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )
        if self.hparams.get("use_scatter_mean", False):
            concatenation_factor += 1
        if not self.hparams["bidir"]:
            concatenation_factor = (concatenation_factor - 1) * 2 + 1

        n_input_features: int = hparams["n_InputFeatures"]
            
        n_edge_layers: int = hparams["n_edge_layers"]
        n_node_layers: int = hparams["n_node_layers"]
        n_hidden:      int = hparams["n_hidden"]
            
            
        ## *********************************************
        print("*"*100)
        print("Initializing the InteractionGNN model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =",n_input_features)
        print("")
        print("   - #layers in edge encoder MLP   : int -> ",n_edge_layers)
        print("   - #layers in node encoder MLP   : int -> ",n_node_layers)
        print("   - #layers in edge output MLP    : int -> ",n_node_layers)
        print("   - dimension of encoding space   : int -> ",n_hidden)
        print("   - #iteration for graph encoding : int -> ",self.hparams["n_graph_iters"])
        print("   - activation of MLP output      : func ->",hparams["hidden_activation"])        
        print("")
        print("   - bidir                         : bool ->",self.hparams["bidir"])        
        print("   - concatenation_factor          : int -> ",concatenation_factor)
        print("")
        print("*"*100)
        ## *********************************************
            
        # ============================================================    
        # Setup input network
        # ============================================================    
        self.node_encoder = make_mlp(
            n_input_features,  # 9 (x, y, z, 6 ellipsoids params)
            [n_hidden] * hparams.get("n_node_encoder_layers", n_node_layers),  # 3
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],  # SiLU
            layer_norm=hparams["layernorm"],  # True
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_encoder = make_mlp(
            2 * (n_hidden),  # input
            [n_hidden] * hparams.get("n_edge_encoder_layers", n_edge_layers),  # 3
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_network = make_mlp(
            3 * n_hidden,
            [n_hidden] * n_edge_layers,  # 6
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # The node network computes new node features
        # ============================================================    
        self.node_network = make_mlp(
            concatenation_factor * n_hidden,
            [n_hidden] * n_node_layers,
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # Final edge output classification network
        # ============================================================    
        self.output_edge_classifier = make_mlp(
            3 * n_hidden,
            [n_hidden] * hparams.get("n_edge_classifier_layers", n_edge_layers)  # 3
            + [1],  # -> output layer has dim 1
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def message_step(self, x, start, end, e):
        # Compute new node features

        edge_messages = torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                scatter_max(e, start, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, start, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs) + x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)

        e_out = self.edge_network(edge_inputs)

        e_out += e

        return x_out, e_out

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)

        return self.output_edge_classifier(classifier_inputs).squeeze(-1)

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def forward(self, x, edge_index):
        start, end = edge_index
        # start = edge_index[0]
        # end = edge_index[1]

        # ENCODING ------
        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x, use_reentrant=True)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1), use_reentrant=True)

        # MESSAGE PASSING ------
        for _ in range(self.hparams["n_graph_iters"]):  # 6
            # x: (n_hits, 256)
            # e: (n_edges, 256)
            x, e = checkpoint(self.message_step, x, start, end, e, use_reentrant=True)

        # OUTPUT -----
        # Compute final edge scores; use original edge directions only
        output = checkpoint(self.output_step, x, start, end, e, use_reentrant=True)

        return output


    def get_score(self, x, edge_index):
        return torch.sigmoid(self.forward(x, edge_index))

    
## ==============================================================================    
## ==============================================================================    
class InteractionGNN_node_position(nn.Module):
    
    """An interaction network class"""
    
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def __init__(self, hparams):
        """
        Initialise the Lightning Module that can scan over different GNN training
        regimes
        """
        super(InteractionGNN_node_position,self).__init__()
        
        # ============================================================    
        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )
        if self.hparams.get("use_scatter_mean", False):
            concatenation_factor += 1
        if not self.hparams["bidir"]:
            concatenation_factor = (concatenation_factor - 1) * 2 + 1

        n_input_features: int = hparams["n_InputFeatures"]
            
        n_edge_layers: int = hparams["n_edge_layers"]
        n_node_layers: int = hparams["n_node_layers"]
        n_hidden:      int = hparams["n_hidden"]            
            
        ## *********************************************
        print("*"*100)
        print("Initializing the InteractionGNN model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =",n_input_features)
        print("")
        print("   - #layers in edge encoder MLP   : int -> ",n_edge_layers)
        print("   - #layers in node encoder MLP   : int -> ",n_node_layers)
        print("   - #layers in edge output MLP    : int -> ",n_node_layers)
        print("   - dimension of encoding space   : int -> ",n_hidden)
        print("   - #iteration for graph encoding : int -> ",self.hparams["n_graph_iters"])
        print("   - activation of MLP output      : func ->",hparams["hidden_activation"])        
        print("")
        print("   - bidir                         : bool ->",self.hparams["bidir"])        
        print("   - concatenation_factor          : int -> ",concatenation_factor)
        print("")
        print("*"*100)
        ## *********************************************
            
        # ============================================================    
        # Setup input network
        # ============================================================    
        self.node_encoder = make_mlp(
            n_input_features,  # 9 (x, y, z, 6 ellipsoids params)
            [n_hidden] * hparams.get("n_node_encoder_layers", n_node_layers),  # 3
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],  # SiLU
            layer_norm=hparams["layernorm"],  # True
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_encoder = make_mlp(
            2 * (n_hidden),  # input
            [n_hidden] * hparams.get("n_edge_encoder_layers", n_edge_layers),  # 3
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_network = make_mlp(
            3 * n_hidden,
            [n_hidden] * n_edge_layers,  # 6
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # The node network computes new node features
        # ============================================================    
        self.node_network = make_mlp(
            concatenation_factor * n_hidden,
            [n_hidden] * n_node_layers,
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # Final node network
        # ============================================================    
        self.output_node_predictor = make_mlp(
            1 * n_hidden,
            [n_hidden] * hparams.get("n_node_classifier_layers", n_node_layers)  # 3
            + [1],  # -> output layer has dim 1
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def message_step(self, x, start, end, e):
        # Compute new node features

        edge_messages = torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                scatter_max(e, start, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, start, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs) + x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)

        e_out = self.edge_network(edge_inputs)

        e_out += e

        return x_out, e_out

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def output_step(self, x):
        classifier_inputs = x

        return self.output_node_predictor(classifier_inputs).squeeze(-1)

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def forward(self, x, edge_index):
        start, end = edge_index
        # start = edge_index[0]
        # end = edge_index[1]

        # ENCODING ------
        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x, use_reentrant=True)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1), use_reentrant=True)

        # MESSAGE PASSING ------
        for _ in range(self.hparams["n_graph_iters"]):  # 6
            # x: (n_hits, 256)
            # e: (n_edges, 256)
            x, e = checkpoint(self.message_step, x, start, end, e, use_reentrant=True)

        # OUTPUT -----
        # Compute final node position predictor; use original edge directions only
        output = checkpoint(self.output_step, x, use_reentrant=True)

        return output



## ==============================================================================    
## ==============================================================================    
class InteractionGNN_node_PV_position(nn.Module):
    
    """An interaction network class"""
    
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def __init__(self, hparams):
        """
        Initialise the Lightning Module that can scan over different GNN training
        regimes
        """
        super(InteractionGNN_node_PV_position,self).__init__()
        
        # ============================================================    
        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )
        if self.hparams.get("use_scatter_mean", False):
            concatenation_factor += 1
        if not self.hparams["bidir"]:
            concatenation_factor = (concatenation_factor - 1) * 2 + 1

        n_input_features: int = hparams["n_InputFeatures"]
            
        n_edge_layers: int = hparams["n_edge_layers"]
        n_node_layers: int = hparams["n_node_layers"]
        n_hidden:      int = hparams["n_hidden"]            
        n_output_features: int = hparams["n_output_features"]            
            
        ## *********************************************
        print("*"*100)
        print("Initializing the InteractionGNN model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =",n_input_features)
        print("")
        print("   - #layers in edge encoder MLP   : int -> ",n_edge_layers)
        print("   - #layers in node encoder MLP   : int -> ",n_node_layers)
        print("   - #layers in edge output MLP    : int -> ",n_node_layers)
        print("   - dimension of encoding space   : int -> ",n_hidden)
        print("   - #iteration for graph encoding : int -> ",self.hparams["n_graph_iters"])
        print("   - activation of MLP output      : func ->",hparams["hidden_activation"])        
        print("")
        print("   - bidir                         : bool ->",self.hparams["bidir"])        
        print("   - concatenation_factor          : int -> ",concatenation_factor)
        print("")
        print("   - N output features             : int ->",n_output_features)        
        print("")        
        print("*"*100)
        ## *********************************************
            
        # ============================================================    
        # Setup input network
        # ============================================================    
        self.node_encoder = make_mlp(
            n_input_features,  # 11 (x, y, z, sx, sy, sz, rho_xy, rho_xz, rho_yz, tx, ty)
            [n_hidden] * hparams.get("n_node_encoder_layers", n_node_layers),  # 3
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],  # SiLU
            layer_norm=hparams["layernorm"],  # True
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_encoder = make_mlp(
            2 * (n_hidden),  # input
            [n_hidden] * hparams.get("n_edge_encoder_layers", n_edge_layers),  # 3
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_network = make_mlp(
            3 * n_hidden,
            [n_hidden] * n_edge_layers,  # 6
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # The node network computes new node features
        # ============================================================    
        self.node_network = make_mlp(
            concatenation_factor * n_hidden,
            [n_hidden] * n_node_layers,
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # Final node network
        # ============================================================    
        self.output_node_predictor = make_mlp(
            1 * n_hidden,
            [n_hidden] * hparams.get("n_node_classifier_layers", n_node_layers)  # 3
            + [n_output_features],  # -> output layer has dim 1
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def message_step(self, x, start, end, e):
        # Compute new node features

        edge_messages = torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                scatter_max(e, start, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, start, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs) + x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)

        e_out = self.edge_network(edge_inputs)

        e_out += e

        return x_out, e_out

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def output_step(self, x):
        classifier_inputs = x

        return self.output_node_predictor(classifier_inputs).squeeze(-1)

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def forward(self, x, edge_index):
        start, end = edge_index
        # start = edge_index[0]
        # end = edge_index[1]

        # ENCODING ------
        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x, use_reentrant=True)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1), use_reentrant=True)

        # MESSAGE PASSING ------
        for _ in range(self.hparams["n_graph_iters"]):  # 6
            # x: (n_hits, 256)
            # e: (n_edges, 256)
            x, e = checkpoint(self.message_step, x, start, end, e, use_reentrant=True)

        # OUTPUT -----
        # Compute final node position predictor; use original edge directions only
        output = checkpoint(self.output_step, x, use_reentrant=True)

        return output

## ==============================================================================    
## ==============================================================================    
class InteractionGNN_node_PV_position_norm(nn.Module):
    
    """An interaction network class"""
    
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def __init__(self, hparams):
        """
        Initialise the Lightning Module that can scan over different GNN training
        regimes
        """
        super(InteractionGNN_node_PV_position_norm,self).__init__()
        
        # ============================================================    
        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )
        if self.hparams.get("use_scatter_mean", False):
            concatenation_factor += 1
        if not self.hparams["bidir"]:
            concatenation_factor = (concatenation_factor - 1) * 2 + 1

        n_input_features: int = hparams["n_InputFeatures"]
            
        n_edge_layers: int = hparams["n_edge_layers"]
        n_node_layers: int = hparams["n_node_layers"]
        n_hidden:      int = hparams["n_hidden"]            
        n_output_features: int = hparams["n_output_features"]            
            
        ## *********************************************
        print("*"*100)
        print("Initializing the InteractionGNN model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =",n_input_features)
        print("")
        print("   - #layers in edge encoder MLP   : int -> ",n_edge_layers)
        print("   - #layers in node encoder MLP   : int -> ",n_node_layers)
        print("   - #layers in edge output MLP    : int -> ",n_node_layers)
        print("   - dimension of encoding space   : int -> ",n_hidden)
        print("   - #iteration for graph encoding : int -> ",self.hparams["n_graph_iters"])
        print("   - activation of MLP output      : func ->",hparams["hidden_activation"])        
        print("")
        print("   - bidir                         : bool ->",self.hparams["bidir"])        
        print("   - concatenation_factor          : int -> ",concatenation_factor)
        print("")
        print("   - N output features             : int ->",n_output_features)        
        print("")        
        print("*"*100)
        ## *********************************************
            
        # ============================================================    
        # Setup input network
        # ============================================================    
        self.node_encoder = make_mlp(
            n_input_features,  # 11 (x, y, z, sx, sy, sz, rho_xy, rho_xz, rho_yz, tx, ty)
            [n_hidden] * hparams.get("n_node_encoder_layers", n_node_layers),  # 3
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],  # SiLU
            layer_norm=hparams["layernorm"],  # True
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_encoder = make_mlp(
            2 * (n_hidden),  # input
            [n_hidden] * hparams.get("n_edge_encoder_layers", n_edge_layers),  # 3
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )  # -> 256

        # ============================================================    
        # The edge network computes new edge features from connected nodes
        # ============================================================    
        self.edge_network = make_mlp(
            3 * n_hidden,
            [n_hidden] * n_edge_layers,  # 6
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # The node network computes new node features
        # ============================================================    
        self.node_network = make_mlp(
            concatenation_factor * n_hidden,
            [n_hidden] * n_node_layers,
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # ============================================================    
        # Final node network
        # ============================================================    
        self.output_node_predictor = make_mlp(
            1 * n_hidden,
            [n_hidden] * hparams.get("n_node_classifier_layers", n_node_layers)  # 3
            + [n_output_features],  # -> output layer has dim 1
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def message_step(self, x, start, end, e):
        # Compute new node features

        edge_messages = torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                scatter_max(e, start, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, start, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs) + x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)

        e_out = self.edge_network(edge_inputs)

        e_out += e

        return x_out, e_out

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def output_step(self, x):
        classifier_inputs = x

        return self.output_node_predictor(classifier_inputs).squeeze(-1)

    # ================================================================    
    # ----------------------------------------------------------------    
    # ================================================================    
    def forward(self, x, edge_index):
        start, end = edge_index
        # start = edge_index[0]
        # end = edge_index[1]

        # ENCODING ------
        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x, use_reentrant=True)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1), use_reentrant=True)

        # MESSAGE PASSING ------
        for _ in range(self.hparams["n_graph_iters"]):  # 6
            # x: (n_hits, 256)
            # e: (n_edges, 256)
            x, e = checkpoint(self.message_step, x, start, end, e, use_reentrant=True)

        # OUTPUT -----
        # Compute final node position predictor; use original edge directions only
        output = checkpoint(self.output_step, x, use_reentrant=True)

        return output

    