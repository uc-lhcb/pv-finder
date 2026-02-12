#!/usr/bin/env python3
'''
Definitions of the different loss functions used for the different models.
'''

import torch
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

# ===========================================================================================
# ===========================================================================================
class GNN_Loss(torch.nn.Module):
    # ---------------------------------------------------------------------------------------
    def __init__(self, configs=""):

        super().__init__()
        
        self.model_type = configs['model_type']
        self.alpha      = configs['models_config'][self.model_type]["alpha"]

        if self.model_type=="InteractionGNN_node_PV_position_norm":
            self.beta       = configs['models_config'][self.model_type]["loss_penalty_beta"]

        if self.model_type=="InteractionGNN":        
            self.hparams   = configs['models_config']["InteractionGNN"]    
                                   
            
    # ---------------------------------------------------------------------------------------
    def compute_loss_dist(self, x, y):
        
        xT = torch.transpose(x,0,1)
        yT = torch.transpose(y,0,1)

        distance = torch.sqrt(torch.pow(xT[0]-yT[0],2)+
                              torch.pow(xT[1]-yT[1],2)+
                              torch.pow(xT[2]-yT[2],2)
                             )
        
        distance_alpha = torch.pow(distance,self.alpha)
        
        loss = distance_alpha.sum()/len(y)                

        return loss        

    # ---------------------------------------------------------------------------------------
    def compute_loss_dist_sigma_norm(self, inputs, x, y):
        
        ## Input features
        iT = torch.transpose(inputs,0,1)
        
        inv_sigma_XYZ = 1./(iT[3]*iT[4]*iT[5]) ## 1./(sigma_X * sigma_Y * sigma_Z)
        
        sum_w = inv_sigma_XYZ.sum()
        
        ## Model predictions
        xT = torch.transpose(x,0,1)
        ## Model targets
        yT = torch.transpose(y,0,1)

        distance = torch.sqrt(torch.pow((xT[0]-yT[0]),2)+
                              torch.pow((xT[1]-yT[1]),2)+
                              torch.pow((xT[2]-yT[2]),2)
                             )
        
        distance_alpha = torch.pow(distance,self.alpha)
        
        distance_alpha_w = distance_alpha * inv_sigma_XYZ
                
        loss = distance_alpha_w.sum()/sum_w         

        return loss        
    
    # ---------------------------------------------------------------------------------------
    def compute_loss_dist_sigma_norm_ov_added(self, inputs, x, y):
        
        ## Input features
        iT = torch.transpose(inputs,0,1)
        
        inv_sigma_XYZ = 1./(iT[3]*iT[4]*iT[5]) ## 1./(sigma_X * sigma_Y * sigma_Z)
        sum_w = inv_sigma_XYZ.sum()
        
        ## Model predictions
        xT = torch.transpose(x,0,1)
        ## Model targets
        yT = torch.transpose(y,0,1)

        
        distance_pv = torch.sqrt(torch.pow((xT[0]-yT[0]),2)+
                                 torch.pow((xT[1]-yT[1]),2)+
                                 torch.pow((xT[2]-yT[2]),2)
                                )        
        distance_pv_alpha = torch.pow(distance_pv,self.alpha)        
        distance_pv_alpha_w = distance_pv_alpha * inv_sigma_XYZ        
        distance_pv_alpha_w_norm = distance_pv_alpha_w.sum()/sum_w

        distance_ov = torch.sqrt(torch.pow((xT[3]-yT[3]),2)+
                                 torch.pow((xT[4]-yT[4]),2)+
                                 torch.pow((xT[5]-yT[5]),2)
                                )
        distance_ov_alpha = torch.pow(distance_ov,self.alpha)        
        distance_ov_alpha_w = distance_ov_alpha * inv_sigma_XYZ        
        distance_ov_alpha_w_norm = distance_ov_alpha_w.sum()/sum_w
        
        loss = (2.*distance_pv_alpha_w_norm + distance_ov_alpha_w_norm)/3.

        return loss        
    
    # ---------------------------------------------------------------------------------------
    def compute_loss_penalty(self, inputs, x, y, true_z_pvs, device):


        iT = torch.transpose(inputs,0,1)

        inv_sigma_XYZ = 1./(iT[3]*iT[4]*iT[5]) ## 1./(sigma_X * sigma_Y * sigma_Z)

        sum_w = inv_sigma_XYZ.sum()

        xT = torch.transpose(x,0,1)
        yT = torch.transpose(y,0,1)

        if len(true_z_pvs)>0:

            y_cat = torch.zeros(len(yT[2])).to(device)

            for iPV in range(len(true_z_pvs)):
                loc = torch.where(yT[2]==true_z_pvs[iPV])
                y_cat[loc] = iPV
            pred_cat = torch.zeros(len(yT[2])).to(device)
            for iTrack in range(len(xT[2])):
                z = xT[2][iTrack]
                cat = torch.argmin(abs(true_z_pvs-z))
                pred_cat[iTrack] = cat
            compare = torch.where(pred_cat==y_cat, 1, 0)
            
            distance = torch.sqrt(torch.pow(xT[0]-yT[0],2)+
                                  torch.pow(xT[1]-yT[1],2)+
                                  torch.pow(xT[2]-yT[2],2)
                                 )
            
            d_unmatch_w_sum = (distance[compare==0]*inv_sigma_XYZ[compare==0]).sum()
            d_match_w_sum   = (distance[compare==1]*inv_sigma_XYZ[compare==1]).sum()

            loss = ( (self.beta * d_unmatch_w_sum) + d_match_w_sum)/sum_w

        else:
            distance = torch.sqrt(torch.pow(xT[0]-yT[0],2)+
                                  torch.pow(xT[1]-yT[1],2)+
                                  torch.pow(xT[2]-yT[2],2)
                                 )
            distance_w_sum = (distance*inv_sigma_XYZ).sum()
            loss = distance_w_sum/sum_w
            
        return loss
    


    # ---------------------------------------------------------------------------------------
    def compute_loss(self, x, y):
        
        if self.model_type=="InteractionGNN_node_position":

            sigma     = 0.01
            diff      = torch.sub(x,y)
            diff      = diff/sigma
            chisq     = torch.pow(diff,2)
            chisqNdof = chisq.sum() / len(y)
            #ave_chisqNdof = chisqNdof / nBatch

            return chisqNdof
        
        elif (self.model_type=="InteractionGNN_node_PV_position" or self.model_type=="InteractionGNN_node_PV_position_norm"):
            
            xT = torch.transpose(x,0,1)
            yT = torch.transpose(y,0,1)

            distance = torch.sqrt(torch.pow(xT[0]-yT[0],2)+
                                  torch.pow(xT[1]-yT[1],2)+
                                  torch.pow(xT[2]-yT[2],2)
                                 )
            distance_norm = distance.sum()/len(y)                

            return distance_norm
            
        else:
            # Compute weights on positive samples
            if self.hparams["focal_loss"]:
                if "weight" in self.hparams:
                    weight = torch.tensor(self.hparams["weight"])
                else:
                    weight = (~y).sum() / y.shape[0]
            else:
                if "weight" in self.hparams:
                    weight = torch.tensor(self.hparams["weight"])
                else:
                    weight = (~y).sum() / y.sum()

            manual_weights = None

            # Compute weighted loss
            ## Bool to choose between two possible loss function: 
            if self.hparams["focal_loss"]:                        
                ## - sigmoid_focal_loss: 
                ##    https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html
                ##
                loss = sigmoid_focal_loss(
                    inputs    = x,
                    targets   = y.float(),
                    alpha     = weight,
                    gamma     = self.hparams["gamma"],
                    reduction = "mean",
                )
            else:
                ## - binary_cross_entropy_with_logits:
                ##    https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
                ##  
                loss = F.binary_cross_entropy_with_logits(
                    x,
                    y.float(),
                    weight     = manual_weights,
                    pos_weight = weight,
                )

        return loss

# ===========================================================================================
# ===========================================================================================
class Loss(torch.nn.Module):
    # ===========================================================================================
    def __init__(self, configs=""):

        super().__init__()

        ## Get the considered experiement: LHCb / ATLAS / CMS...
        Exp = configs['Experiment']
        self.model_class = configs["model_class"]

        ## epsilon is a parameter that can be adjusted.
        self.epsilon     = configs["training_configs"]["epsilon_loss"]
        ## asym_coeff adjust asymmetry; 1.0 <==> symmetric
        self.asym_coeff  = configs["training_configs"]["asym_coeff"]
        
        ## nBins is used for normalisation of the loss and depends if using full KDE or intervals
        if self.model_class=="tracks-to-KDE" or self.model_class=="tracks-to-hist":
            self.nBins = configs['global_configs'][Exp]["nBinsPerInterval"]
            
        elif self.model_class=="KDE-to-hist":
            self.nBins = configs['global_configs'][Exp]["n_bins_poca_kde"]
            
           
        self.verbose = False

        tag_func = {"tracks-to-KDE": "built in Chi^2",
                    "KDE-to-hist":   "built in -log(2r/(r^2 + 1)) with asym parameter",
                    "tracks-to-hist":"built in -log(2r/(r^2 + 1)) with asym parameter"}
                        
        ## *********************************************
        print("*"*100)
        print("Initializing the Loss using %s function"%(tag_func[self.model_class]))
        print("")
        print("with the following parameters:\n")
        if self.model_class=="KDE-to-hist" or self.model_class=="tracks-to-hist" or self.model_class=="tracks-to-KDE":
            print("   - Epsilon    =",self.epsilon)
            if self.model_class=="KDE-to-hist" or self.model_class=="tracks-to-hist":
                print("   - Asym coeff =",self.asym_coeff)
            print("   - nBins (for normalisation)    =",self.nBins)
        print("")
        print("*"*100)
        ## *********************************************
        
    # ===========================================================================================
    def forward(self, x, y):

        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        nBatch = y.shape[0]
        if self.verbose:
            print("nBatch = ", nBatch)
            print("y.shape = ",y.shape)
            print("x.shape = ",x.shape)

        '''
        # *******************************************
        NOTE THAT THIS NOTE TESTED AND ONE SHOULD BE 
        VERY CAUTIOUS WHEN USING KDE-to-hist models!!
        '''
        if self.model_class=="KDE-to-hist":
            y = y.transpose(1,2) 
            y = y[:,:,0] ## Get only the KDE, not the X_max, Y_max
        '''
        # *******************************************
        '''

        # --------------------------------------------------------------------------------
        # In the current implementation, not nan shoudl appear as only create target hist for valid PVs 
        # Nevertheless, we keep this implementation as at worst it should do nothing...
        #
        # Make a boolean mask of non-nan values of the target histogram.
        # This will be used to select items from y:
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html#boolean-or-mask-index-arrays
        #
        # Note that if masking was not requested when loading the data, there
        # will be no NaNs and this will be all Trues and will do nothing special.
        valid = ~torch.isnan(y)

        # --------------------------------------------------------------------------------
        # Using a loss function which is built to optimise the ratio between 
        # predicted and target histograms with an asymmetry parameter acting
        # like a golbal shift on the predicted shape (shifting up or down).
        # This will result in 
        # - increasing efficiency AND false FP (shifting   up predicted hist)
        # - decreasing efficiency AND false FP (shifting down predicted hist)        
        # --------------------------------------------------------------------------------
        if self.model_class=="KDE-to-hist" or self.model_class=="tracks-to-hist":
            # --------------------------------------------------------------------------------
            # Compute r, only including non-nan values. r will probably be shorter than x and y.        
            r = torch.abs((x[valid] + self.epsilon) / (y[valid] + self.epsilon))

            # --------------------------------------------------------------------------------
            # Compute -log(2r/(r² + 1))
            alpha = -torch.log(2*r / (r**2 + 1))
            alpha = alpha * (1.0 + self.asym_coeff * torch.exp(-r))

            # --------------------------------------------------------------------------------
            # Sum up the alpha values, and divide by the length of x and y. Note this is not quite
            # a .mean(), since alpha can be a bit shorter than x and y due to masking.
            beta      = alpha.sum() / self.nBins
            ave_beta  = beta / nBatch

            return ave_beta

        elif self.model_class=="tracks-to-KDE":
            # --------------------------------------------------------------------------------
            # Compute a standard MSE like loss function when training tracks-to-KDE models as
            # this seems better suited for this problem.

            sigma     = 0.01
            diff      = torch.sub(x[valid],y[valid])
            diff      = diff/sigma
            chisq     = torch.pow(diff,2)
            chisqNdof = chisq.sum() / self.nBins
            ave_chisqNdof = chisqNdof / nBatch

            return ave_chisqNdof


                