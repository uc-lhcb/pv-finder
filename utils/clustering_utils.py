'''
File containing most of the methods usefull accross many scripts
'''

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
import sys
import time
from datetime import datetime
from pathlib import Path, PosixPath
import awkward as ak
import h5py as h5py
import json
import numpy as np
import pandas as pd
from collections import namedtuple
import torch

from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader

from tools.ToolBox import get_res

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN
import matplotlib.cm as mcm
import awkward as ak

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import mplhep as hep
plt.style.use(hep.style.ROOT)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

## ========================================================================================================================
## METHOD TO REMOVE TRUE PVs WITH : N_tracks < validPV_minTracks
## ========================================================================================================================
def skim_data_with_low_mutl_true_pv(data, model, validPV_minTracks):
        
    ## ------------------------------------------------------------------------------------
    ## Get the tracks parameters (x,y,z,sx,sy,sz) 
    tracks_x = data.x_norm.cpu()        
    tracks_x = np.array(tracks_x[:,:6])
    
    ## ------------------------------------------------------------------------------------
    ## Get the tracks category 
    tracks_cat = data.tracks_cat.cpu()
    tracks_cat = np.array([tracks_cat]).T
        
    ## ------------------------------------------------------------------------------------
    ## "y_pos" is already in the [zmin,zmax] range
    y = data.y_pos.cpu()
        
    ## ------------------------------------------------------------------------------------
    ## Get the predicted track PV locations in z    
    pred = model(data.x_norm,data.edge_index).cpu()
    
    ## ------------------------------------------------------------------------------------
    ## Get the MC truth PV z and n tracks 
    true_pvs_z = data.true_pvs_z
    true_pvs_n = data.true_pvs_n

    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
    ## Filter the truth level MC info  
    filt_truth_valid   = np.where((true_pvs_n>=validPV_minTracks) & (true_pvs_z>-100) & (true_pvs_z<300))
    filt_truth_invalid = np.where(true_pvs_n<validPV_minTracks)
    
    true_pvs_z_valid   = true_pvs_z[filt_truth_valid]
    true_pvs_n_valid   = true_pvs_n[filt_truth_valid]
    
    true_pvs_z_invalid = true_pvs_z[filt_truth_invalid]
    
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
    ## Filter the tracks info based on the 
    ## invalid PVs from "true_pvs_z_invalid"
    ## Get the number of tracks (before filtering)
    nTracks = len(tracks_x)
    ## Set the filter to be true for all tracks
    tracks_filt = np.ones(nTracks, dtype=bool)
    
    ## ------------------------------------------------------------------------------------
    ## Get the tracks true PV z location
    tracks_true_z = y.T[2]
    
    ## ------------------------------------------------------------------------------------
    ## Loop over the tracks and for each track check 
    ## its true PV location against the invalid PV location
    ## If identical set the tracks filter to False
    for iTrack in range(nTracks):
        track_true_z = tracks_true_z[iTrack]
        for pv_z in true_pvs_z_invalid:
            if track_true_z == pv_z:
                tracks_filt[iTrack] = False
            
    ## Finally apply the tracks filter to all tracks containers
    tracks_x_filt   = tracks_x[tracks_filt]
    tracks_cat_filt = tracks_cat[tracks_filt]
    y_filt          = y[tracks_filt]
    pred_filt       = pred[tracks_filt]
    
    return tracks_x_filt, tracks_cat_filt, y_filt, pred_filt, true_pvs_z_valid, true_pvs_n_valid
## ========================================================================================================================


## ========================================================================================================================
def shift_labels(labels):
    
    ## ------------------------------------------------------------------------------------------
    ## Shift back the labels to continuous integers 
    ## This step is needed each time we modify the labels content
    ## ------------------------------------------------------------------------------------------
    unique_labels = np.unique(labels[labels>=0])        
    #print("unique_labels",unique_labels)
    for idx, unique_label in enumerate(unique_labels):
        labels[labels==unique_label] = idx    
        
    return labels
## ========================================================================================================================





## ========================================================================================================================
def dist_2D(p,q):
    return np.sqrt(np.power(p[0]-q[0],2)+np.power(p[1]-q[1],2))
## ========================================================================================================================

## ========================================================================================================================
def get_dists_2D(clusters_pred,clusters_reco,z_pred,z_reco):
    dists = np.zeros(len(clusters_pred))
    for iCluster in range(len(clusters_pred)):
        dists[iCluster] = dist_2D(
            [clusters_pred[iCluster],clusters_reco[iCluster]],
            [z_pred,z_reco])
        
    return dists
## ========================================================================================================================

## ========================================================================================================================
def matched_unlabelled_tracks(labels, X, method="pred", min_dist=1, verbose=False):
        
        
    ## ------------------------------------------------------------------------------------------
    ## Function to expand the clusters by adding unmatched tracks to either seeds or proto-clusters
    ## 
    ## Three possible methods are implemented. All methods will need as input the seed of proto-cluster
    ## mean position based on the z_pred or z_reco values of the tracks forming the seed of proto-cluster. 
    ## 
    ## The three "aggregation" methods are then:
    ##
    ## A_  method=="pred": add all tracks satisfying | z_pred_track - cluster_z_pred_mean | < min_dist
    ## 
    ## B_  method=="reco": add all tracks satisfying | z_reco_track - cluster_z_reco_mean | < min_dist
    ## 
    ## C_  method=="2D": add all tracks satisfying 2D_dist < min_dist, 
    ##                   where 2D_dist is the distance between the track and seed in the [z_pred, z_reco] plane
    ## 
    ## The aggregation method will actually first assign any unmatched track to the closest 
    ## seed or proto-cluster before checking if the distance satisfies the conditions A, B or C depending 
    ## on the argument "method"!
    ##
    ## ------------------------------------------------------------------------------------------
        
    if verbose:
        print("labels",labels)
    ## ------------------------------------------------------------------------------------------
    ## Start by getting the cluster mean in the z_pred dimension:
    ## ------------------------------------------------------------------------------------------
    clusters_ID = np.unique(labels[labels>=0], return_counts=False)
    clusters_z_pred_mean = np.zeros(len(clusters_ID))
    clusters_z_reco_mean = np.zeros(len(clusters_ID))
    for iCluster in range(len(clusters_ID)):            
        filt = labels==int(clusters_ID[iCluster])
        clusters_z_pred_mean[iCluster] = np.mean(X[filt][:,8])
        clusters_z_reco_mean[iCluster] = np.mean(X[filt][:,2])
        if verbose:
            print("clusters_ID[%s]"%iCluster,clusters_ID[iCluster])
            print("clusters_z_pred_mean = %.3f"%clusters_z_pred_mean[iCluster])
            print("clusters_z_reco_mean = %.3f"%clusters_z_reco_mean[iCluster])

    ## ------------------------------------------------------------------------------------------
    ## Now loop over the tracks, and check the distance between all predicted cluster 
    ## to assign the track to the closest one. 
    if verbose:
        print("labels",labels)
    ## ------------------------------------------------------------------------------------------
    ## First check if there is at least one predicted cluster:
    if len(clusters_ID)>0:
        for iTrack in range(len(X)):
            ## ------------------------------------------------------------------------------------------
            ## Check only unmatched tracks
            if labels[iTrack]>=0: continue
                
            if method=="pred":
                dists = abs(clusters_z_pred_mean - X[iTrack,8])                
            elif method=="reco":
                dists = abs(clusters_z_reco_mean - X[iTrack,2])
            elif method=="2D":
                dists = get_dists_2D(clusters_z_pred_mean,clusters_z_reco_mean,X[iTrack,2],X[iTrack,8])

            if verbose:
                print("At track[%s]"%iTrack)
                print("     old label    = %d"%labels[iTrack])
                print("     method  =",method)
                print("     min_dist :",min_dist)
                print("     dists   :",dists)
                print("     closest_cluster_ID :",np.argmin(dists))
                print("     dists[closest_cluster_ID] :",dists[np.argmin(dists)])
            closest_cluster_ID = np.argmin(dists)
            ## ------------------------------------------------------------------------------------------
            ## Check if track is not too far away
            if dists[closest_cluster_ID] < min_dist:
                labels[iTrack] = closest_cluster_ID
    else:
        if verbose:
            print("No seed or proto-cluster here...")
            print("Will not perform any matching, and move to the next group!")
        
    return labels
## ========================================================================================================================


## ========================================================================================================================
def tracks_weighted_pos(tracks_x, tracks_y, tracks_z, 
                        tracks_x_sigma, tracks_y_sigma, tracks_z_sigma,
                        min_tracks_z_sigma = 0.1,
                        verbose=False
                       ):

    tracks_z_sigma = np.where(tracks_z_sigma<min_tracks_z_sigma,min_tracks_z_sigma,tracks_z_sigma)
    if verbose:
        print("tracks_z_sigma",tracks_z_sigma)
    w_x = np.power(tracks_x_sigma,-2)
    w_y = np.power(tracks_y_sigma,-2)
    w_z = np.power(tracks_z_sigma,-2)
    
    sum_w_x = np.sum(w_x)
    sum_w_y = np.sum(w_y)
    sum_w_z = np.sum(w_z)
    
    if verbose:
        print("sum_w_x",sum_w_x)
        print("sum_w_y",sum_w_y)
        print("sum_w_z",sum_w_z)
        #for iTrack in range(len(w_x)):        
        #    print("Track[%s] at z = %.3f +/- %.3f with weight norm z: %.1f"%(iTrack, tracks_z[iTrack], tracks_z_sigma[iTrack], (w_z[iTrack]/sum_w_z)*100.))
        
    mu_x = np.sum(w_x * tracks_x)/sum_w_x
    mu_y = np.sum(w_y * tracks_y)/sum_w_y
    mu_z = np.sum(w_z * tracks_z)/sum_w_z

    v_x = np.sum(w_x * np.power(tracks_x-mu_x,2))/sum_w_x
    v_y = np.sum(w_y * np.power(tracks_y-mu_y,2))/sum_w_y
    v_z = np.sum(w_z * np.power(tracks_z-mu_z,2))/sum_w_z
    
    s_x = 1./np.sqrt(sum_w_x)
    s_y = 1./np.sqrt(sum_w_y)
    s_z = 1./np.sqrt(sum_w_z)
        
    return [mu_x, mu_y, mu_z, s_x, s_y, s_z, v_x, v_y, v_z]
## ========================================================================================================================


## ========================================================================================================================
def get_pred_pv_info(X, nsigma = 1, min_sigma_z = 0.15, verbose=False):
    
    ## ---------------------------------------------------------------
    ## Get the tracks coordinates and uncertainties
    tracks_x_pv  = X.T[0]
    tracks_y_pv  = X.T[1]
    tracks_z_pv  = X.T[2]
    tracks_sx_pv = X.T[3]
    tracks_sy_pv = X.T[4]
    tracks_sz_pv = X.T[5]
        
    ## ---------------------------------------------------------------
    ## Get the corresponding group of tracks weighted mean, sigma and variance in cartesian coordinates
    ## Tracks with too small uncertainties (carying too much weight) are set to have sigma_z = min_sigma_z
    if verbose:
        print("-"*100)
        print("Getting the cluster weighted position and variance")
        print("")
    info_pv = tracks_weighted_pos(tracks_x_pv, tracks_y_pv, tracks_z_pv,
                                  tracks_sx_pv,tracks_sy_pv,tracks_sz_pv,
                                  min_sigma_z,
                                  verbose=verbose)

    ## ---------------------------------------------------------------
    ## Get the difference between each track position in Z and the weighted mean in Z
    dz_pv = tracks_z_pv-info_pv[2]
    #dzmin = zmin-info_pv[2]
    #dzmax = zmax-info_pv[2]

    ## ---------------------------------------------------------------
    ## Now normalize by the variance
    if verbose:
        print("-"*100)
        print("Normalizing the z range by the cluster variance: var =",np.sqrt(info_pv[8]))
        print("")
    dz_over_var_pv    = dz_pv/np.sqrt(info_pv[8])
    #dz_over_var_min   = dzmin/np.sqrt(info_pv[8])
    #dz_over_var_max   = dzmax/np.sqrt(info_pv[8])

    ## ---------------------------------------------------------------
    ## Filter tracks that are within a factor of nsigma*variance from the mean
    filt_good = np.where(abs(dz_over_var_pv)<nsigma)
    filt_bad  = np.where(abs(dz_over_var_pv)>=nsigma)
    dz_over_var_pv_filt = dz_over_var_pv[filt_good]

    if verbose:
        print("-"*100)
        print("Filtering tracks outside the %s sigma region of the variance"%nsigma)
        print("")
        print("valid filtered tracks:")
        print(filt_good)
    
    ## ---------------------------------------------------------------
    ## Recompute the mean and uncertainties for tracks that are within the window: nsigma*variance from the mean
    ## Tracks with too small uncertainties (carying too much weight) are set to have sigma_z = min_sigma_z
    if verbose:
        print("-"*100)
        print("Recomputing cluster position or those valid tracks only")
        print("")
    info_pv_filt_good = tracks_weighted_pos(tracks_x_pv[filt_good], tracks_y_pv[filt_good], tracks_z_pv[filt_good],
                                            tracks_sx_pv[filt_good],tracks_sy_pv[filt_good],tracks_sz_pv[filt_good],
                                            min_sigma_z,
                                            verbose=verbose)

    return info_pv_filt_good
## ========================================================================================================================


## ========================================================================================================================
def tracks_weighted_pred_pos(tracks_pred_z, 
                             tracks_z_sigma,
                             min_tracks_z_sigma = 0.1,
                             verbose=False
                            ):

    if verbose:
        print("In tracks_weighted_pred_pos")
        print("tracks_z_sigma",tracks_z_sigma)
        
    tracks_z_sigma = np.where(tracks_z_sigma<min_tracks_z_sigma,min_tracks_z_sigma,tracks_z_sigma)

    if verbose:
        print("tracks_z_sigma",tracks_z_sigma)
        
    w_z = np.power(tracks_z_sigma,-2)    
    sum_w_z = np.sum(w_z)
    
    if verbose:
        print("sum_w_z",sum_w_z)
        for iTrack in range(len(w_z)):        
            print("Track[%s] with pred z = %.3f with weight norm z: %.1f"%(iTrack, tracks_pred_z[iTrack], (w_z[iTrack]/sum_w_z)*100.))
        
    mu_z = np.sum(w_z * tracks_pred_z)/sum_w_z
    s_z = 1./np.sqrt(sum_w_z)
        
    return [mu_z, s_z]
## ========================================================================================================================


## ========================================================================================================================
def skim_low_mult_clusters(labels, min_tracks=2, verbose=False):

    ## ---------------------------------------------------------------------------------------------
    ## Set back labels of tracks in clusters with n_tracks < min_tracks to label == -1, i.e. unmatched!
    ## ---------------------------------------------------------------------------------------------
    clusters_ID, clusters_nTracks = np.unique(labels, return_counts=True)
    for iCluster in range(len(clusters_ID)):
        if verbose:
            print("clusters_ID[%s]"%iCluster,clusters_ID[iCluster])
            print("clusters_N[%s] "%iCluster,clusters_nTracks[iCluster])
        if clusters_nTracks[iCluster] < min_tracks:
            labels[labels==int(clusters_ID[iCluster])] = -1
    
    ## ---------------------------------------------------------------------------------------------
    ## Shift back the labels from 0 to N with continuous integers (no gaps)
    ## ---------------------------------------------------------------------------------------------
    labels = shift_labels(labels)
    
    return labels
## ========================================================================================================================


## ========================================================================================================================
def skim_low_precison_tracks(labels, X, sigma_z_max=4, verbose=False):

    ## ---------------------------------------------------------------------------------------------
    ## Set back the labels of tracks with sigma_z > sigma_z_max to label == -1, i.e. unmatched!
    ## ---------------------------------------------------------------------------------------------    
    for iTrack in range(len(X)):
        if X[iTrack,5]>sigma_z_max:
            labels[iTrack]=-1
            
    ## ---------------------------------------------------------------------------------------------
    ## Shift back the labels from 0 to N with continuous integers (no gaps)
    ## ---------------------------------------------------------------------------------------------
    labels = shift_labels(labels)
    
    return labels
## ========================================================================================================================



## ========================================================================================================================
def skim_neighbor_cluster(labels, X, deltaZ=1, nsigma=1, min_sigma_z=0.15, verbose=False):

    ## ---------------------------------------------------------------------------------------------
    if verbose:
        print("="*100)
        print("In skim_neighbor_cluster:")
    clusters_ID, clusters_nTracks = np.unique(labels[labels>=0], return_counts=True)
    if verbose:
        print("clusters_ID",clusters_ID)
        print("clusters_nTracks",clusters_nTracks)
    for iCluster in range(len(clusters_ID)-1):
        if verbose:
            print("clusters_ID[%s]"%iCluster,clusters_ID[iCluster])
            print("clusters_N[%s] "%iCluster,clusters_nTracks[iCluster])
            print("clusters_ID[%d]"%int(iCluster+1),clusters_ID[iCluster+1])
            print("clusters_N[%d] "%int(iCluster+1),clusters_nTracks[iCluster+1])
        X_cluster_a = X[labels==int(clusters_ID[iCluster])]
        X_cluster_b = X[labels==int(clusters_ID[iCluster+1])]
        pv_info_a = get_pred_pv_info(X_cluster_a, 
                                       nsigma = nsigma, 
                                       min_sigma_z=min_sigma_z, 
                                       verbose=verbose)
        pv_info_b = get_pred_pv_info(X_cluster_b, 
                                       nsigma = nsigma, 
                                       min_sigma_z=min_sigma_z, 
                                       verbose=verbose)
                 
        ## Check the difference in z between the two predicted PVs
        delta_z = abs(pv_info_b[2]-pv_info_a[2])
        if delta_z < deltaZ:
            ## Set the labels of tracks from the cluster with the least number of tracks as un-matched!
            if clusters_nTracks[iCluster]<clusters_nTracks[iCluster+1]:
                #labels[labels==int(clusters_ID[iCluster])] = -1
                labels[labels==int(clusters_ID[iCluster])] = int(clusters_ID[iCluster+1])
            else:
                #labels[labels==int(clusters_ID[iCluster+1])] = -1
                labels[labels==int(clusters_ID[iCluster+1])] = int(clusters_ID[iCluster])
                
    ## ---------------------------------------------------------------------------------------------
    ## Shift back the labels from 0 to N with continuous integers (no gaps)
    labels = shift_labels(labels)
    
    return labels
## ========================================================================================================================


## ========================================================================================================================
def skim_low_mult_neighbor_cluster(labels, X, deltaZ_max=1, deltaZ_min=1, min_multiplicity=10, min_sigma_z=0.15, verbose=False):

    if verbose:
        print("labels at start of skim_low_mult_neighbor_cluster",labels)
    
    ## ---------------------------------------------------------------------------------------------
    clusters_ID, clusters_nTracks = np.unique(labels[labels>=0], return_counts=True)
    cluster_pass = -1
    
    if verbose:
        print("clusters_ID",clusters_ID)
        print("clusters_nTracks",clusters_nTracks)
    
    for iCluster in range(len(clusters_ID)-2):
        #print("iCluster",iCluster)
        if iCluster==cluster_pass: continue
            
        if verbose:
            print("clusters_ID[%s]"%iCluster,clusters_ID[iCluster])
            print("clusters_N[%s] "%iCluster,clusters_nTracks[iCluster])
            print("clusters_ID[%d]"%int(iCluster+1),clusters_ID[iCluster+1])
            print("clusters_N[%d] "%int(iCluster+1),clusters_nTracks[iCluster+1])
            print("clusters_ID[%d]"%int(iCluster+2),clusters_ID[iCluster+2])
            print("clusters_N[%d] "%int(iCluster+2),clusters_nTracks[iCluster+2])
            
        n_tracks_cluster_b = clusters_nTracks[iCluster+1]
        
        ## ------------------------------------------------------------------------------------------------
        ## If middle cluster has more than "min_multiplicity" tracks, then do not do anything...
        if n_tracks_cluster_b>min_multiplicity: continue
            
        ## ------------------------------------------------------------------------------------------------
        ## Compute the mean of the three clusters
        ## ------------------------------------------------------------------------------------------------
        pred_z_a = X[labels==int(clusters_ID[iCluster])][:,8]
        tracks_sz_a = X[labels==int(clusters_ID[iCluster])][:,5]
        mean_pred_a = tracks_weighted_pred_pos(pred_z_a,tracks_sz_a,min_tracks_z_sigma = min_sigma_z,verbose=verbose)
        
        pred_z_b = X[labels==int(clusters_ID[iCluster+1])][:,8]
        tracks_sz_b = X[labels==int(clusters_ID[iCluster+1])][:,5]
        mean_pred_b = tracks_weighted_pred_pos(pred_z_b,tracks_sz_b,min_tracks_z_sigma = min_sigma_z,verbose=verbose)
        
        pred_z_c = X[labels==int(clusters_ID[iCluster+2])][:,8]
        tracks_sz_c = X[labels==int(clusters_ID[iCluster+2])][:,5]
        mean_pred_c = tracks_weighted_pred_pos(pred_z_c,tracks_sz_c,min_tracks_z_sigma = min_sigma_z,verbose=verbose)
        
        if verbose:
            print("mean_pred_a",mean_pred_a[0])
            print("mean_pred_b",mean_pred_b[0])
            print("mean_pred_c",mean_pred_c[0])
        
        ## ------------------------------------------------------------------------------------------------
        ## Check the difference in z between the middle and the two outer predicted PVs
        ## ------------------------------------------------------------------------------------------------
        delta_z_ab = abs(mean_pred_b[0]-mean_pred_a[0])
        delta_z_cb = abs(mean_pred_b[0]-mean_pred_c[0])
        if verbose:
            print("delta_z_ab",delta_z_ab)
            print("delta_z_cb",delta_z_cb)        
        if delta_z_ab < deltaZ_max and delta_z_cb < deltaZ_max:
            if verbose:
                print("condition AND",deltaZ_max)
            if delta_z_ab < deltaZ_min or delta_z_cb < deltaZ_min:
                if verbose:
                    print("condition OR",deltaZ_min)
                ## Set the labels of tracks from the central cluster (i.e. b) as unmatched!
                labels[labels==int(clusters_ID[iCluster+1])] = -1
                cluster_pass = iCluster+1
                
            
    ## ---------------------------------------------------------------------------------------------
    ## Shift back the labels from 0 to N with continuous integers (no gaps)
    labels = shift_labels(labels)
    
    if verbose:
        print("labels at end of skim_low_mult_neighbor_cluster",labels)
        
    return labels
## ========================================================================================================================


## ========================================================================================================================
def split_left_right_cluster(labels, X, 
                             method_sep         = "reco", 
                             nsigma_sep         =  2, 
                             min_tracks_cluster = 20, 
                             min_tracks_split   =  5,
                             min_sigma_z        = 0.15,
                             verbose=False):
    
    ## ---------------------------------------------------------------------------------------------
    ## This method will check if any given cluster has an "S shape", i.e. two blobs in the Z predicted 
    ## dimension. First, a check on the cluster multiplicity is performed, and if the cluster has less 
    ## than min_tracks_cluster tracks, nothing will be done. If not (n_tracks >= min_tracks_cluster) and 
    ## an S shape is detected, then it will further check of that the number of tracks corresponding to each blob
    ## is above a given threshold (min_tracks_split). If all conditions are met, then the cluster will be split 
    ## in two new clusters.
    
    ## ---------------------------------------------------------------------------------------------
    clusters_ID, clusters_nTracks = np.unique(labels[labels>=0], return_counts=True)

    
    d_cluster_labels = {}
    d_cluster_labels[-1] = np.where(labels==-1)[0]
    shift=0
    
    for iCluster in range(len(clusters_ID)):
        
        if verbose:
            print("*"*50)
            print("iCluster",iCluster)
            
        ## Get the original cluster tracks idx    
        cluster_tracks_idx = np.where(labels==int(clusters_ID[iCluster]))[0]
        
        ## Get the original cluster has less than min_tracks_cluster then do not go further
        ## and assign the cluster_tracks_idx to the dictionnary before moving to the 
        ## next cluster
        if clusters_nTracks[iCluster]<=min_tracks_cluster:
            if verbose:
                print("Number of tracks in cluster (%d) below minimum (%d)"%(clusters_nTracks[iCluster],min_tracks_cluster))
            d_cluster_labels[iCluster+shift] = cluster_tracks_idx
            continue
        
        pred_z = X[labels==int(clusters_ID[iCluster])][:,8]
        reco_z = X[labels==int(clusters_ID[iCluster])][:,2]
        tracks_sz = X[labels==int(clusters_ID[iCluster])][:,5]

        cluster_mean_pred = tracks_weighted_pred_pos(pred_z,tracks_sz,
                                                     min_tracks_z_sigma = min_sigma_z,
                                                     verbose=False)
        cluster_mean_reco = tracks_weighted_pred_pos(reco_z,tracks_sz,
                                                     min_tracks_z_sigma = min_sigma_z,
                                                     verbose=False)
        
        ## -------------------------------------------------
        low_track_pred_z    = pred_z[pred_z<=cluster_mean_pred[0]]
        low_tracks_pred_sz  = tracks_sz[pred_z<=cluster_mean_pred[0]]
        low_mean_pred       = tracks_weighted_pred_pos(low_track_pred_z,low_tracks_pred_sz,
                                                       min_tracks_z_sigma = min_sigma_z,
                                                       verbose=False)
        
        high_track_pred_z   = pred_z[pred_z>cluster_mean_pred[0]]
        high_tracks_pred_sz = tracks_sz[pred_z>cluster_mean_pred[0]]
        high_mean_pred      = tracks_weighted_pred_pos(high_track_pred_z,high_tracks_pred_sz,
                                                       min_tracks_z_sigma = min_sigma_z,
                                                       verbose=False)

        ## Check the number of tracks remaining for the split set
        ## of tracks as it cannot be smaller than min_tracks_split
        if len(low_track_pred_z)<min_tracks_split or len(high_track_pred_z)<min_tracks_split: 
            if verbose:
                print("Number of tracks in the low (%d) or high (%d) split cluster below minimum (%d)"%(len(low_track_pred_z),len(high_track_pred_z),min_tracks_split))
            d_cluster_labels[iCluster+shift] = cluster_tracks_idx
            continue
            
        ## -------------------------------------------------
        ## IMPORTANT NOTE:
        ## For the low and high based on the reco mean, 
        ## we still want to pick the predicted z position mean
        low_track_reco_z    = pred_z[reco_z<=cluster_mean_reco[0]]
        low_tracks_reco_sz  = tracks_sz[reco_z<=cluster_mean_reco[0]]
        low_mean_reco       = tracks_weighted_pred_pos(low_track_reco_z,low_tracks_reco_sz,
                                                       min_tracks_z_sigma = min_sigma_z,verbose=False)

        high_track_reco_z   = pred_z[reco_z>cluster_mean_reco[0]]
        high_tracks_reco_sz = tracks_sz[reco_z>cluster_mean_reco[0]]
        high_mean_reco      = tracks_weighted_pred_pos(high_track_reco_z,high_tracks_reco_sz,
                                                       min_tracks_z_sigma = min_sigma_z,verbose=False)
        
        
        if verbose:
            print("---------------------------")
            print("mean_pred",cluster_mean_pred[0])
            print("mean_pred_s",cluster_mean_pred[1])
            print("low_mean_pred",low_mean_pred[0])        
            print("low_mean_pred_s",low_mean_pred[1])        
            print("high_mean_pred",high_mean_pred[0])        
            print("high_mean_pred_s",high_mean_pred[1])        
            print("---------------------------")
            print("mean_reco",cluster_mean_reco[0])
            print("mean_reco_s",cluster_mean_reco[1])
            print("low_mean_reco",low_mean_reco[0])        
            print("low_mean_reco_s",low_mean_reco[1])        
            print("high_mean_reco",high_mean_reco[0])        
            print("high_mean_reco_s",high_mean_reco[1]) 
            
        delta_z_pred = abs(low_mean_pred[0]-high_mean_pred[0])
        s_pred = np.sqrt(low_mean_pred[1]*low_mean_pred[1]+high_mean_pred[1]*high_mean_pred[1])
        if verbose:
            print("---------------------------")
            print("delta_z_pred = %.3f"%delta_z_pred)
            print("s_pred     = %.3f"%s_pred)
        delta_z_reco = abs(low_mean_reco[0]-high_mean_reco[0])
        s_reco = np.sqrt(low_mean_reco[1]*low_mean_reco[1]+high_mean_reco[1]*high_mean_reco[1])
        if verbose:
            print("---------------------------")
            print("delta_z_reco = %.3f"%delta_z_reco)
            print("s_reco     = %.3f"%s_reco)
        
        ## If the mean of the low and high part of the cluster are separated by more than nsigma_sep times 
        ## the quadratic sum of uncertainties on the pred z position, then we split the cluster in two
        if method_sep=="pred":
            if delta_z_pred>nsigma_sep*s_pred:

                ## First, we need to shift the next clusters label by one
                #for iCluster_shift in range(iCluster+1,len(clusters_ID)):
                #    labels[labels==int(iCluster_shift)] = iCluster_shift+1

                ## Now, we can to shift the high tracks label by one without oevrlapping
                ## as the label for iCluster+1 is now free.
                filt_high = np.where(pred_z>cluster_mean_pred[0])[0]
                filt_low  = np.where(pred_z<=cluster_mean_pred[0])[0]
                high_cluster_tracks_idx = cluster_tracks_idx[filt_high]
                low_cluster_tracks_idx  = cluster_tracks_idx[filt_low]

                d_cluster_labels[iCluster+shift]   = low_cluster_tracks_idx
                d_cluster_labels[iCluster+shift+1] = high_cluster_tracks_idx

                ## Assign the label of the high part of the cluster a label = iCluster+1
                #labels[high_cluster_tracks_idx] = iCluster+1
                shift+=1
            else:   
                d_cluster_labels[iCluster+shift] = cluster_tracks_idx
                
        elif method_sep=="reco":
            if delta_z_reco>nsigma_sep*s_reco:

                ## First, we need to shift the next clusters label by one
                #for iCluster_shift in range(iCluster+1,len(clusters_ID)):
                #    labels[labels==int(iCluster_shift)] = iCluster_shift+1

                ## Now, we can shift the high tracks label by one without overlapping
                ## as the label for iCluster+1 is now free.
                filt_high = np.where(reco_z>cluster_mean_reco[0])[0]
                filt_low  = np.where(reco_z<=cluster_mean_reco[0])[0]
                high_cluster_tracks_idx = cluster_tracks_idx[filt_high]
                low_cluster_tracks_idx  = cluster_tracks_idx[filt_low]

                d_cluster_labels[iCluster+shift]   = low_cluster_tracks_idx
                d_cluster_labels[iCluster+shift+1] = high_cluster_tracks_idx

                ## Assign the label of the high part of the cluster a label = iCluster+1
                #labels[high_cluster_tracks_idx] = iCluster+1
                shift+=1
            else:   
                d_cluster_labels[iCluster+shift] = cluster_tracks_idx
                
    for key in d_cluster_labels.keys():
        if verbose:
            print("%s:"%key,d_cluster_labels[key])
        labels[d_cluster_labels[key]] = key
        
    return labels 
## ========================================================================================================================


## ========================================================================================================================
def merge_triplet_neighbor_cluster(labels, X, deltaZ_max=1, deltaZ_min=1, min_sigma_z=0.15, verbose=False):

    if verbose:
        print("labels",labels)
    
    ## ---------------------------------------------------------------------------------------------
    clusters_ID, clusters_nTracks = np.unique(labels[labels>=0], return_counts=True)
    cluster_pass = -1
    
    if verbose:
        print("clusters_ID",clusters_ID)
        print("clusters_nTracks",clusters_nTracks)
    
    for iCluster in range(len(clusters_ID)-2):
        #print("iCluster",iCluster)
        if iCluster==cluster_pass: continue
            
        if verbose:
            print("="*50)
            print("clusters_ID[%s]"%iCluster,clusters_ID[iCluster])
            print("clusters_N[%s] "%iCluster,clusters_nTracks[iCluster])
            print("clusters_ID[%d]"%int(iCluster+1),clusters_ID[iCluster+1])
            print("clusters_N[%d] "%int(iCluster+1),clusters_nTracks[iCluster+1])
            print("clusters_ID[%d]"%int(iCluster+2),clusters_ID[iCluster+2])
            print("clusters_N[%d] "%int(iCluster+2),clusters_nTracks[iCluster+2])
            
        n_tracks_cluster_b = clusters_nTracks[iCluster+1]
        if verbose:
            print("n_tracks_cluster_a",clusters_nTracks[iCluster])
            print("n_tracks_cluster_b",n_tracks_cluster_b)
            print("n_tracks_cluster_c",clusters_nTracks[iCluster+2])
        
            
        pred_z_a = X[labels==int(clusters_ID[iCluster])][:,8]
        tracks_sz_a = X[labels==int(clusters_ID[iCluster])][:,5]
        mean_pred_a = tracks_weighted_pred_pos(pred_z_a,tracks_sz_a,min_tracks_z_sigma = min_sigma_z,verbose=False)
        
        pred_z_b = X[labels==int(clusters_ID[iCluster+1])][:,8]
        tracks_sz_b = X[labels==int(clusters_ID[iCluster+1])][:,5]
        mean_pred_b = tracks_weighted_pred_pos(pred_z_b,tracks_sz_b,min_tracks_z_sigma = min_sigma_z,verbose=False)
        
        pred_z_c = X[labels==int(clusters_ID[iCluster+2])][:,8]
        tracks_sz_c = X[labels==int(clusters_ID[iCluster+2])][:,5]
        mean_pred_c = tracks_weighted_pred_pos(pred_z_c,tracks_sz_c,min_tracks_z_sigma = min_sigma_z,verbose=False)
        
        if verbose:
            print("mean_pred_a",mean_pred_a[0])
            print("mean_pred_b",mean_pred_b[0])
            print("mean_pred_c",mean_pred_c[0])
        
        ## Check the difference in z between the two predicted PVs
        delta_z_ab = abs(mean_pred_b[0]-mean_pred_a[0])
        delta_z_cb = abs(mean_pred_b[0]-mean_pred_c[0])
        if verbose:
            print("delta_z_ab",delta_z_ab)
            print("delta_z_cb",delta_z_cb)        
        if delta_z_ab < deltaZ_max and delta_z_cb < deltaZ_max:
            if verbose:
                print("condition AND",deltaZ_max)
            if delta_z_ab < deltaZ_min or delta_z_cb < deltaZ_min:
                if verbose:
                    print("condition OR",deltaZ_min)
                ## Get the tracks index of the central cluster
                cluster_tracks_idx = np.where(labels==int(clusters_ID[iCluster+1]))[0]
                if verbose:
                    print("cluster_tracks_idx",cluster_tracks_idx)

                ## Set the tracks index lists to be matched to either 
                ## the previous or next cluster 
                idx_match_a = []
                idx_match_c = []
                ## Fill the list of tracks index lists based on the distance in pred z
                for iTrack in range(len(pred_z_b)):
                    if abs(pred_z_b[iTrack]-mean_pred_a[0])<abs(pred_z_b[iTrack]-mean_pred_c[0]):
                        #print("idx_match_a ==> cluster_tracks_idx[%d]"%iTrack,cluster_tracks_idx[iTrack])
                        idx_match_a.append(cluster_tracks_idx[iTrack])
                    else:
                        #print("idx_match_c ==> cluster_tracks_idx[%d]"%iTrack,cluster_tracks_idx[iTrack])
                        idx_match_c.append(cluster_tracks_idx[iTrack])
                        
                if verbose:
                    print("idx_match_a",idx_match_a)
                    print("idx_match_c",idx_match_c)
                ## Set the labels of tracks from the central cluster (i.e. b) to either 
                ## the previous or next cluster 
                labels[idx_match_a] = clusters_ID[iCluster]
                labels[idx_match_c] = clusters_ID[iCluster+2]
                
                cluster_pass = iCluster+1
                
            
    ## ---------------------------------------------------------------------------------------------
    ## Shift back the labels from 0 to N with continuous integers (no gaps)
    labels = shift_labels(labels)
    if verbose:
        print("labels",labels)
    return labels
## ========================================================================================================================


## ========================================================================================================================
def merge_close_cluster_in_reco(labels, X, 
                                delta_z_reco_sig_max_abs=9, 
                                delta_z_reco_sig_max=5,
                                delta_z_pred_sig_max=5,
                                r_tracks_max=0.4, 
                                nsigma=1, 
                                min_sigma_z=0.15, 
                                verbose=False):

    ## --------------------------------------------------
    ## List of clusters to be merged
    ## --------------------------------------------------
    merging_clusters = []
    if verbose:
        print("merging_clusters",merging_clusters)            
    
    if verbose:
        print("="*100)
        print("In merge_close_cluster_in_reco:")
    clusters_ID, clusters_nTracks = np.unique(labels[labels>=0], return_counts=True)
    if verbose:
        print("clusters_ID",clusters_ID)
        print("clusters_nTracks",clusters_nTracks)
    ## --------------------------------------------------
    ## Loop over clusters
    ## --------------------------------------------------
    for iCluster in range(len(clusters_ID)-1):
        if verbose:
            print("clusters_ID[%s]"%iCluster,clusters_ID[iCluster])
            print("clusters_N[%s] "%iCluster,clusters_nTracks[iCluster])
            print("clusters_ID[%d]"%int(iCluster+1),clusters_ID[iCluster+1])
            print("clusters_N[%d] "%int(iCluster+1),clusters_nTracks[iCluster+1])
        X_cluster_a = X[labels==int(clusters_ID[iCluster])]
        X_cluster_b = X[labels==int(clusters_ID[iCluster+1])]
        pv_info_a = get_pred_pv_info(X_cluster_a, 
                                     nsigma = nsigma, 
                                     min_sigma_z=min_sigma_z, 
                                     verbose=verbose)
        pv_info_b = get_pred_pv_info(X_cluster_b, 
                                     nsigma = nsigma, 
                                     min_sigma_z=min_sigma_z, 
                                     verbose=verbose)
                 
                
        ## ----------------------------------------------------------------------------------------------------
        ## Check the difference in z reco between the two clusters
        ## ----------------------------------------------------------------------------------------------------
        delta_z_reco   = abs(pv_info_b[2]-pv_info_a[2])
        delta_z_reco_s = np.sqrt(pv_info_b[5]*pv_info_b[5]+pv_info_a[5]*pv_info_a[5])
        delta_z_reco_sig = delta_z_reco/delta_z_reco_s
        
        if delta_z_reco_sig < delta_z_reco_sig_max_abs:
            
            z_pred_a = tracks_weighted_pred_pos(X[labels==int(clusters_ID[iCluster])][:,8],
                                                X[labels==int(clusters_ID[iCluster])][:,5],                                            
                                                min_tracks_z_sigma = min_sigma_z,
                                                verbose=False)
            z_pred_b = tracks_weighted_pred_pos(X[labels==int(clusters_ID[iCluster+1])][:,8],
                                                X[labels==int(clusters_ID[iCluster+1])][:,5],                                            
                                                min_tracks_z_sigma = min_sigma_z,
                                                verbose=False)
            
            ## ----------------------------------------------------------------------------------------------------
            ## Check the difference in z pred between the two clusters
            ## ----------------------------------------------------------------------------------------------------
            delta_z_pred   = abs(z_pred_b[0]-z_pred_a[0])
            delta_z_pred_s = np.sqrt(z_pred_b[1]*z_pred_b[1]+z_pred_a[1]*z_pred_a[1])
            delta_z_pred_sig = delta_z_pred/delta_z_pred_s
        
            ## ----------------------------------------------------------------------------------------------------
            ## Check the ratio of tracks between the two clusters
            ## ----------------------------------------------------------------------------------------------------
            n_tracks_a = clusters_nTracks[iCluster]
            n_tracks_b = clusters_nTracks[iCluster+1]
            
            r_tracks = 1.
            if n_tracks_a<n_tracks_b:
                r_tracks = n_tracks_a/float(n_tracks_b)
            else:
                r_tracks = n_tracks_b/float(n_tracks_a)
                                
            if verbose:
                print("-"*30)
                print("z_reco_a       = %.3f +/- %.3f"%(pv_info_a[2],pv_info_a[5]))
                print("z_reco_b       = %.3f +/- %.3f"%(pv_info_b[2],pv_info_b[5]))
                print("delta_z_reco   = %.3f +/- %.3f"%(delta_z_reco,delta_z_reco_s))
                print("d/ds reco      = %.3f"%(delta_z_reco_sig))
                print("")
                print("z_pred_a       = %.3f +/- %.3f"%(z_pred_a[0],z_pred_a[1]))
                print("z_pred_b       = %.3f +/- %.3f"%(z_pred_b[0],z_pred_b[1]))
                print("delta_z_pred   = %.3f +/- %.3f"%(delta_z_pred,delta_z_pred_s))
                print("d/ds pred      = %.3f"%(delta_z_pred_sig))
                print("")
                print("N_a = %d"%n_tracks_a)
                print("N_b = %d"%n_tracks_b)
                print("r_tracks = %.3f"%r_tracks)
                print("-"*30)

            if delta_z_reco_sig < delta_z_reco_sig_max or delta_z_pred_sig < delta_z_pred_sig_max or r_tracks<r_tracks_max:
                if verbose:
                    print("MERGING ON !!!!")
                merging_clusters.append([iCluster, iCluster+1, delta_z_reco_sig, n_tracks_a, n_tracks_b])
    
    
    final_merging_clusters = []
    if verbose:
        print("merging_clusters",merging_clusters)            
    
    ## ----------------------------------------------------------------------------------------------------
    ## Loop over clusters to be merged
    ## ----------------------------------------------------------------------------------------------------
    last_idx = len(merging_clusters)-1
    checked_last = False
    for Idx in range(len(merging_clusters)-1):
        
        merging_cluster_a = merging_clusters[Idx]
        merging_cluster_b = merging_clusters[Idx+1]
        
        if merging_cluster_a[1]==merging_cluster_b[0]:
            
            if (Idx+1)==last_idx:
                checked_last = True
                
            if merging_cluster_a[2]<merging_cluster_b[2]:
                if merging_cluster_a[3]<merging_cluster_a[4]:
                    final_merging_clusters.append([merging_cluster_a[0],merging_cluster_a[1]])
                else:
                    final_merging_clusters.append([merging_cluster_a[1],merging_cluster_a[0]])
            else:
                if merging_cluster_b[3]<merging_cluster_b[4]:
                    final_merging_clusters.append([merging_cluster_b[0],merging_cluster_b[1]])
                else:
                    final_merging_clusters.append([merging_cluster_b[1],merging_cluster_b[0]])
        else:
            
            if merging_cluster_a[3]<merging_cluster_a[4]:
                final_merging_clusters.append([merging_cluster_a[0],merging_cluster_a[1]])
            else:
                final_merging_clusters.append([merging_cluster_a[1],merging_cluster_a[0]])
                    
    if merging_clusters and not checked_last:        
        merging_cluster_a = merging_clusters[last_idx]
        
        if merging_cluster_a[3]<merging_cluster_a[4]:
            final_merging_clusters.append([merging_cluster_a[0],merging_cluster_a[1]])
        else:
            final_merging_clusters.append([merging_cluster_a[1],merging_cluster_a[0]])
        
        
    if verbose:
        print("labels",labels)
    for Idx in range(len(final_merging_clusters)): 
        final_merging_cluster = final_merging_clusters[Idx]
        
        labels[labels==int(final_merging_cluster[0])] = int(final_merging_cluster[1])
        
    if verbose:
        print("labels",labels)
        
    ## ---------------------------------------------------------------------------------------------
    ## Shift back the labels from 0 to N with continuous integers (no gaps)
    labels = shift_labels(labels)
    
    return labels
## ========================================================================================================================


## ========================================================================================================================
## ========================================================================================================================
## ========================================================================================================================
## MAIN METHOD TO PERFORM CLUSTERING. ALMOST ALL DEDICATED METHODS CALLED WITHIN THIS FUNCTION 
## ========================================================================================================================
## ========================================================================================================================
## ========================================================================================================================
def my_clustering_diagonal(X, clusterin_config,
                           cat="pred", 
                           verbose=False):
    
    '''
    NOTE: THE CLUSTERING METHOD IS A SEQUENCE CONSISTING OF THE FOLLOWING STEPS:
    
    ## =========================
    ## PROTO-CLUSTERS FORMATION
    ## =========================
    
    ## The following 5 steps are used to form the proto-clusters from basic logic based on the 
       predicted PV Z position AND reconstructed Z position of the tracks.  
    
    1_ Build seeds from tracks that are within a diagonal band in [z_pred ; z_reco] plane
    
    2_ Discard tracks { skim_low_precison } with large uncertainties as these tracks are 
       typically away from diagonal band and should not be very significant
       
    3_ Discard seeds { skim_low_mult_seed } with a number of tracks less than a minimal number
       "min_tracks_seed". This allows to avoid creating seeds from a few "random tracks" that happens 
       to be close-by. Will reduce the FP rate AND the efficiency for low-multiplicity true PVs.
       
    4_ Form clusters from the selected seeds using the { matched_unlabelled_tracks } method. 
       This method is also used in some of the following steps, and has different clustering 
       methods that is selected from the input parameters parsed to the method. 
       Here (in step 4), the clusters are build by adding tracks with within a dist of [ +/- buildCluster_dist] 
       from the meen seed position in different and configurable dimensions (see header of the method for more details)  
    
    5_ Discard clusters { skim_low_mult_seed } with a number of tracks less than a minimal number
       "min_tracks_cluster". This allows to avoid creating clusters from a few "random tracks" that happens 
       to be close-by. Will reduce the FP rate AND the efficiency for low-multiplicity true PVs.
       
       
    ## =======================================
    ## FROM PROTO-CLUSTERS TO FINAL CLUSTERS
    ## =======================================
       
    ## Now that the proto-clusters are formed, it appears that we can improve performances 
       by treating some pathological cases: 
       
       - one predicted cluster is formed from tracks from two or more real PVS (this happens when two real PVs are close in Z)
         >>> This will decrease the efficiency
         
       - two predicted clusters that are close in Z are formed from tracks originating from one unique real PVs
         >>> This will increase the FP rate
         
       The following steps are meant to deal with these pathological cases.
       
    6_ We start by searching for pairs of close predicted PV in Z { skim_neighbor_cluster } with mean position in Z less 
       than a configurable value and set to unmatch the tracks from the cluster with the lowest multiplicity. 
       
    7_ Now look for triplets { skim_low_mult_neighbor_cluster } of close predicted clusters which most often correspond to 
       the case were two nearby PV are reconstrcuted as three clusters. If such a configuration is found, then check if the 
       "middle" cluster has a "low" multiplicity, and if so, set the tracks from this middle cluster as unmatched.
    
    8_ Search for special cases where two real PVs with large mutiplicities are close by and for which the GNN output 
       (i.e. predicted z PV of tracks) will generate a "S shape" distribution with tracks in the "middle" joining the 
       two more straigh lines in the z reco dimension. Given that tracks will be present between the two "lines", the 
       clustering will consider this as one PV. However, by splitting all predicted clusters in two regions (left and right)
       in the z reco dimension and comparing the mean in Z pred for the two set, one can check the compatibility of the
       mean predicted Z values. If the compatibility is above a threshold, then we would want to split the cluster in two 
       as it is more likelly to correspond to two real PVs
       
    9_ Once all "False" clusters formed from tracks originating from multiple true PVs, we can try merging together 
       multiple clusters that actually correspond to a unique PV. 
       This is done using the method { merge_triplet_neighbor_cluster }, that will scan the predicted clusters and study 
       the spearation in Z of all triplets, a check will be performed on the distance between the central cluster and the 
       two outer clusters.
       If the central cluster is relatively close to both the two outer clusters and that it is very close to one cluster 
       in particular, the tracks from the central cluster will be assigned to either one of the two outer clusters based on
       the distance between the tracks reconstructed Z position and the clusters mean position in Z.
       
    '''
    
    ## ----------------------------------------------------------------------
    ## Retrieve the configurable parameters:
    ## ----------------------------------------------------------------------
    d_diag_max = clusterin_config["d_diag_max"]
    d_pred_max = clusterin_config["d_pred_max"]
    
    skim_low_precison   = clusterin_config["skim_low_precison"]
    sigma_z_max         = clusterin_config["sigma_z_max"]
    
    skim_low_mult_seed  = clusterin_config["skim_low_mult_seed"]
    min_tracks_seed     = clusterin_config["min_tracks_seed"]
    
    buildCluster        = clusterin_config["buildCluster"]
    buildCluster_method = clusterin_config["buildCluster_method"]
    buildCluster_dist   = clusterin_config["buildCluster_dist"]
    
    skim_low_mult_cluster = clusterin_config["skim_low_mult_cluster"]
    min_tracks_cluster    = clusterin_config["min_tracks_cluster"]
    
    skim_neighbors        = clusterin_config["skim_neighbors"]
    deltaZ_cluster_min    = clusterin_config["deltaZ_cluster_min"]
    nsigma_skim_neighbors = clusterin_config["nsigma_skim_neighbors"]
    min_sigma_z           = clusterin_config["min_sigma_z"]

    skim_low_mult_neighbors = clusterin_config["skim_low_mult_neighbors"]
    deltaZ_low_mult_cluster_max = clusterin_config["deltaZ_low_mult_cluster_max"]
    deltaZ_low_mult_cluster_min = clusterin_config["deltaZ_low_mult_cluster_min"]
    min_multiplicity_neighbor_cluster = clusterin_config["min_multiplicity_neighbor_cluster"]
    
    
    split_LR_clusters                    = clusterin_config["split_LR_clusters"]
    split_LR_clusters_method             = clusterin_config["split_LR_clusters_method"]
    split_LR_clusters_nsigma             = clusterin_config["split_LR_clusters_nsigma"]
    split_LR_clusters_min_tracks_cluster = clusterin_config["split_LR_clusters_min_tracks_cluster"]
    split_LR_clusters_min_tracks_split   = clusterin_config["split_LR_clusters_min_tracks_split"]        
    
    merge_triplet_neighbors      = clusterin_config["merge_triplet_neighbors"]
    deltaZ_triplet_cluster_max   = clusterin_config["deltaZ_triplet_cluster_max"]
    deltaZ_triplet_cluster_min   = clusterin_config["deltaZ_triplet_cluster_min"]

    merge_close_cluster_in_rec     = clusterin_config["merge_close_cluster_in_rec"]
    merge_delta_z_reco_sig_max_abs = clusterin_config["merge_delta_z_reco_sig_max_abs"]
    merge_delta_z_reco_sig_max     = clusterin_config["merge_delta_z_reco_sig_max"]
    merge_delta_z_pred_sig_max     = clusterin_config["merge_delta_z_pred_sig_max"]
    merge_r_tracks_max             = clusterin_config["merge_r_tracks_max"]
    merge_nsigma                   = clusterin_config["merge_nsigma"]

    
    
    
    finalClustering        = clusterin_config["finalClustering"]
    finalClustering_method = clusterin_config["finalClustering_method"]
    finalClustering_dist   = clusterin_config["finalClustering_dist"]
    
    ## ---------------------------------------------------------------------------------------------
    ## NOTE: the tracks input "X" is ordered based on the predicted Z pv position from the GNN output
    ## ---------------------------------------------------------------------------------------------
    
    ## ---------------------------------------------------------------------------------------------
    ## Define the tracks label container from a np.array set with -1 (i.e. unmatched)   
    ## The 'labels' container is used to keep track of which tracks are clustered together
    ## i.e. with the same integer value. 
    ## Unmatched tracks at any step of the clustering methods will have label == -1
    labels = np.zeros(len(X))-1
    
    
    ## ------------------------------------------------------------------------------------------------------------
    ## The real clustering sequence (cat=="pred") is defined below when clustering tracks using the output of the GNN.
    ## Another and different clustering sequence (cat=="true") is defined in the case were we use the true tracks PV  
    ## position, usefull for plotting and performance evaluation.
    ## ------------------------------------------------------------------------------------------------------------
    if cat=="pred":
        
        '''
        ################################
        ##
        ## >>>>>>   STEP 1_   <<<<<<<
        ##
        ################################
        '''
        ## ---------------------------------------------------------------------------------------------
        ## Now start looping over tracks and search for seeds
        ## These are tracks that are the closest to the reco_z=pred_z diagonal line
        ## ------------------------------------------------------------------------------------------
        iCluster = 0
        first_cluster = True
        pred_z_start  = -1000.

        for iTrack in range(len(X)):
            reco_z = X[iTrack,2]
            pred_z = X[iTrack,8]
            d_diag = abs(reco_z-pred_z)/np.sqrt(2.)
            if d_diag<d_diag_max:

                if first_cluster:
                    ## First cluster => 
                    ## - store first track pred_z as pred_z_start
                    ## - set back first_cluster to False
                    pred_z_start = pred_z                
                    first_cluster = False

                ## --------------------------------------------------------------
                ## Compute difference between current track and track with lowest pred_z (i.e. pred_z_start)
                delta_z = (pred_z - pred_z_start)

                if delta_z<=d_pred_max:
                    ## This track belongs to the current cluster. 
                    labels[iTrack] = iCluster

                else:
                    ## This track belongs to another cluster. 
                    pred_z_start = pred_z                
                    iCluster += 1
                    labels[iTrack] = iCluster


                #labels[iTrack] = iCluster

                if verbose:
                    print("At track[%s]"%iTrack)
                    print("with %s_z = %.3f"%(cat,pred_z))
                    print("     reco_z = %.3f"%reco_z)
                    print("     d_diag = %.3f"%d_diag)
                    print("     pred_z_start = %.3f"%pred_z_start)
                    print("     delta_z_pred = %.3f"%(pred_z - pred_z_start))
                    print("     iCluster = %s"%iCluster)
                    print("     label  = %d"%labels[iTrack])

        '''
        ################################
        ##
        ## >>>>>>   STEP 2_   <<<<<<<
        ##
        ################################
        '''
        ## ---------------------------------------------------------------------------------------------
        ## Look for tracks with large uncertainties as we want to use only tracks with good precision 
        ## when building the clusters.
        ## If sigma_z of tracks large than "sigma_z_max" set them to unmtached (label==-1)
        ## ------------------------------------------------------------------------------------------
        if skim_low_precison:
            labels = skim_low_precison_tracks(labels, X, 
                                              sigma_z_max = sigma_z_max, 
                                              verbose     = verbose)
        #print("labels after skim_low_precison",labels)
        
        '''
        ################################
        ##
        ## >>>>>>   STEP 3_   <<<<<<<
        ##
        ################################        
        '''
        ## ---------------------------------------------------------------------------------------------
        ## Look for predicted cluster seeds that have less then "min_tracks_seed" tracks.
        ## If so, set all the corresponding clusters tracks to unmtached (label==-1)
        ## ------------------------------------------------------------------------------------------
        if skim_low_mult_seed:
            labels = skim_low_mult_clusters(labels, 
                                           min_tracks = min_tracks_seed, 
                                           verbose    = verbose)
            
        #print("labels after skim_low_mult_seed",labels)

        '''
        ################################
        ##
        ## >>>>>>   STEP 4_   <<<<<<<
        ##
        ################################        
        '''
        ## ---------------------------------------------------------------------------------------------
        ## For each seed expand the search in a band in the predicted z axis of width "buildCluster_dist" 
        ## which is the set from the mean value of pred_z value.
        ## ------------------------------------------------------------------------------------------
        if buildCluster:
            labels = matched_unlabelled_tracks(labels, X, 
                                               method   = buildCluster_method, 
                                               min_dist = buildCluster_dist, 
                                               verbose  = False)
            
        #print("labels after buildCluster",labels)
        
        '''
        ################################
        ##
        ## >>>>>>   STEP 5_   <<<<<<<
        ##
        ################################        
        '''
        ## ------------------------------------------------------------------------------------------
        ## Check for clusters of less than "min_tracks_cluster" tracks and set them unmtached (label==-1) 
        ## ------------------------------------------------------------------------------------------
        if skim_low_mult_cluster:
            labels = skim_low_mult_clusters(labels, 
                                           min_tracks = min_tracks_cluster, 
                                           verbose    = verbose)

        #print("labels after skim_low_mult_cluster",labels)

        
        '''
        ################################
        ##
        ## >>>>>>   STEP 6_   <<<<<<<
        ##
        ################################        
        '''        
        ## ------------------------------------------------------------------------------------------
        ## Check for nearby clusters (less than "deltaZ" [in mm] in pred z). 
        ## For two close clusters, set to unmtached (label==-1) the tracks from the cluster 
        ## with the lowest mutiplicity.  
        ## ------------------------------------------------------------------------------------------
        if skim_neighbors:
            labels = skim_neighbor_cluster(labels, X, 
                                           deltaZ      = deltaZ_cluster_min, 
                                           nsigma      = nsigma_skim_neighbors, 
                                           min_sigma_z = min_sigma_z, 
                                           verbose     = verbose)            
            
        #print("labels after skim_neighbors",labels)
            
        '''
        ################################
        ##
        ## >>>>>>   STEP 7_   <<<<<<<
        ##
        ################################        
        '''        
        ## ------------------------------------------------------------------------------------------
        ## Check for nearby clusters (less than "deltaZ" [in mm] in pred z). 
        ## For three close clusters, set to unmtached (label==-1) the tracks from the middle cluster 
        ## only if mutiplicity is low eanough. 
        ## ------------------------------------------------------------------------------------------
        if skim_low_mult_neighbors:
            labels = skim_low_mult_neighbor_cluster(labels, X, 
                                                    deltaZ_max  = deltaZ_low_mult_cluster_max, 
                                                    deltaZ_min  = deltaZ_low_mult_cluster_min, 
                                                    min_multiplicity = min_multiplicity_neighbor_cluster,
                                                    min_sigma_z = min_sigma_z, 
                                                    verbose = verbose)
            ## Repeat the method for multiple close predicted clusters occurences 
            labels = skim_low_mult_neighbor_cluster(labels, X, 
                                                    deltaZ_max  = deltaZ_low_mult_cluster_max, 
                                                    deltaZ_min  = deltaZ_low_mult_cluster_min, 
                                                    min_multiplicity = min_multiplicity_neighbor_cluster,
                                                    min_sigma_z = min_sigma_z, 
                                                    verbose = verbose)            
 

        '''
        ################################
        ##
        ## >>>>>>   STEP 8_   <<<<<<<
        ##
        ################################        
        '''        
        ## ------------------------------------------------------------------------------------------
        ## Check for asymmetric clusters (S shape), that would indicate two real PVs 
        ## ------------------------------------------------------------------------------------------
        if split_LR_clusters:
            labels = split_left_right_cluster(labels, X, 
                                              method_sep         = split_LR_clusters_method,
                                              nsigma_sep         = split_LR_clusters_nsigma,
                                              min_tracks_cluster = split_LR_clusters_min_tracks_cluster, 
                                              min_tracks_split   = split_LR_clusters_min_tracks_split,
                                              min_sigma_z        = min_sigma_z, 
                                              verbose=verbose)
        #print("labels after split_left_right_cluster",labels)
        
        
        
        '''
        ################################
        ##
        ## >>>>>>   STEP 9_   <<<<<<<
        ##
        ################################        
        '''        
        ## ------------------------------------------------------------------------------------------
        ## Check for triplet neighor clusters 
        ## ------------------------------------------------------------------------------------------
        if merge_triplet_neighbors:
            labels = merge_triplet_neighbor_cluster(labels, X, 
                                                    deltaZ_max  = deltaZ_triplet_cluster_max, 
                                                    deltaZ_min  = deltaZ_triplet_cluster_min, 
                                                    min_sigma_z = min_sigma_z, 
                                                    verbose = verbose) 
        
        
        '''
        ################################
        ##
        ## >>>>>>   STEP 10_   <<<<<<<
        ##
        ################################        
        '''        
        ## ------------------------------------------------------------------------------------------
        ## Check for close neighor clusters in the z reco dimension 
        ## ------------------------------------------------------------------------------------------
        if merge_close_cluster_in_rec:
            labels = merge_close_cluster_in_reco(labels, X, 
                                                 delta_z_reco_sig_max_abs = merge_delta_z_reco_sig_max_abs, 
                                                 delta_z_reco_sig_max     = merge_delta_z_reco_sig_max,
                                                 delta_z_pred_sig_max     = merge_delta_z_pred_sig_max,
                                                 r_tracks_max             = merge_r_tracks_max,
                                                 nsigma                   = merge_nsigma,
                                                 min_sigma_z              = min_sigma_z, 
                                                 verbose                  = verbose)
        
        ## ------------------------------------------------------------------------------------------
        ## Now assign all unmatched tracks to the closest cluster in the z_pred dimension:
        ## ------------------------------------------------------------------------------------------
        if finalClustering:
            labels = matched_unlabelled_tracks(labels, X, 
                                               method   = finalClustering_method, 
                                               min_dist = finalClustering_dist, 
                                               verbose  = verbose)
        
                
    elif cat=="true":
        iCluster = 0
        first_cluster = True
        for iTrack in range(len(X)):
            pred_z = X[iTrack,8]
            if first_cluster:
                ## First cluster => 
                ## - store first track pred_z as pred_z_start
                ## - set back first_cluster to False
                pred_z_start = pred_z                
                first_cluster = False            
            ## Compute difference between current track and track with lowest pred_z (i.e. pred_z_start)
            delta_z = (pred_z - pred_z_start)

            ## Here delta_z_max is simply an arbitrary low value since all delta_z for tracks from the 
            ## same PV should be equal to zero (using MC truth information!). 
            delta_z_max = 0.01
            if delta_z<=delta_z_max:
                ## This track belongs to the current cluster. 
                labels[iTrack] = iCluster
            else:
                ## This track belongs to another cluster. 
                pred_z_start = pred_z                
                iCluster += 1
                labels[iTrack] = iCluster
            if verbose:
                print("At track[%s]"%iTrack)
                print("with %s_z = %.3f"%(cat,pred_z))
                print("     pred_z_start = %.3f"%pred_z_start)
                print("     delta_z_pred = %.3f"%(pred_z - pred_z_start))
                print("     iCluster = %s"%iCluster)
                print("     label  = %d"%labels[iTrack])                

    ## ---------------------------------------------------------------------------------
    ## Transpose the labels so that we can add them to the tracks container X
    ## ---------------------------------------------------------------------------------
    labels_T = np.array(labels)[:,None]
    X_label  = np.concatenate([X,labels_T],axis=1)     
        
    return X_label
## ========================================================================================================================

## ========================================================================================================================
def get_PVs_info(X_labels, nsigma=1, min_sigma_z=0.15, verbose=False):
    
    ## Looping over all clusters (using the labels added to X in the clustering method),
    ## we can now retrieve the clusters parameters (nine params):
    ## weighted X,Y,Z ; weighted SX, SY, SZ and weighted variace in X, Y and Z    
    if verbose:
        print("-"*100)
        print("In get_PVs_info:")
    pvs_info = []

    labels = X_labels[:,9].T
    clusters_ID, clusters_nTracks = np.unique(labels[labels>=0], return_counts=True)
    for iCluster in range(len(clusters_ID)):
        pv_info = get_pred_pv_info(X_labels[np.where(X_labels[:,9]==int(clusters_ID[iCluster]))], 
                                   nsigma = nsigma, 
                                   min_sigma_z=min_sigma_z, 
                                   verbose=verbose)
        pv_info.append(int(clusters_nTracks[iCluster]))
        pvs_info.append(pv_info)
        
    return pvs_info
## ==================================================================================================


## ==================================================================================================
## ==================================================================================================
## ==================================================================================================
## METHOD TO RETRIEVE THE PREDICTED PVs INFO and to PLOT THEM -- 
## MAIN CLUSTERING IS CALLED IN THIS FUNCTION
## ==================================================================================================
## ==================================================================================================
## ==================================================================================================
def get_info_and_plot_cluster(X_pred, X_true, 
                              zmin, zmax, 
                              iEvt, iGroup, 
                              configs,
                              doPlot  = False,                             
                              verbose = False):


    ## ---------------------------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------
    ## Get the predicted labels, added to the tracks info X_pred container, from the clustering method
    X_pred_labels = my_clustering_diagonal(X_pred, 
                                           configs["GNN_clusterin_config"],
                                           cat="pred",                                           
                                           verbose=verbose)
    
    
    if verbose:
        print("-"*100)
        print("Getting the PVs predicted info:")
    pred_pvs_info = get_PVs_info(X_pred_labels,
                                 nsigma=configs["GNN_clusterin_config"]["nSigma_pv_info"], 
                                 min_sigma_z=configs["GNN_clusterin_config"]["min_sigma_z"],
                                 verbose=verbose)

        
    ## ---------------------------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------
    ## Get the true labels, added to the tracks info X_true container, from the clustering method
    X_true_labels = my_clustering_diagonal(X_true,
                                           configs["GNN_clusterin_config"],
                                           cat="true",
                                           verbose=verbose)
    
     
    ## Get the true labels outside of the plotting section, because we might want the
    ## n_tracks_true_vis info corresponding to the number of "visible" tracks available 
    ## to the GNN, and not the MC true number of tracks originating from the PV
    true_labels = X_true_labels[:,9].T
    _, n_tracks_true_vis = np.unique(true_labels, return_counts=True)
        
    if verbose:
        print("-"*100)
        print("Getting the PVs true info:")
    true_pvs_info = get_PVs_info(X_true_labels,
                                 nsigma=configs["GNN_clusterin_config"]["nSigma_pv_info"], 
                                 min_sigma_z=configs["GNN_clusterin_config"]["min_sigma_z"],
                                 verbose=verbose)    
     
    
    ## ---------------------------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------
    ## Now let's make plots... only if requested!
    if doPlot:  
        
        ## ---------------------------------------------------------------------------------------------
        ## ---------------------------------------------------------------------------------------------
        ## Define colors to be used for plotting
        colors = np.concatenate((mcm.Set1(range(8)),mcm.Set2(range(8))))
        
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## Get the PREDICTED data to be ploted
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------


        ## -------------------------------------------------------------------
        ## Get the predicted labels from the tracks info X_pred container
        pred_labels = X_pred_labels[:,9].T
        if verbose:
            print("pred_labels",pred_labels)
        ## -------------------------------------------------------------------
        ## Get the number of clusters: i.e. number of tracks with identical label that is >= 0
        n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        ## Get the number of un-associated tracks
        n_noise_ = list(pred_labels).count(-1)

        ## Set an array with idx values for the tracks
        pred_idx = np.arange(len(pred_labels))
        #print("pred_idx",pred_idx)

        ## Set an array with idx values for the tracks
        pred_idx = pred_idx[np.where(pred_labels>=0)]
        #print("pred_idx",pred_idx)

        ## Get the clusters label values
        pred_unique_labels = set(pred_labels)
        #print("pred_unique_labels",pred_unique_labels)

        pred_core_samples_mask = np.zeros_like(pred_labels, dtype=bool)
        #print("pred_core_samples_mask",pred_core_samples_mask)

        pred_core_samples_mask[pred_idx] = True
        #print("pred_core_samples_mask[pred_idx]",pred_core_samples_mask[pred_idx])

        ## Get the colors for each predicted cluster from the colors array 
        pred_colors = colors[:len(pred_unique_labels)]
        
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## Get the TRUE data to be ploted
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------

        ## Set an array with idx values for the tracks
        true_idx = np.arange(len(true_labels))

        ## Set an array with idx values for the tracks
        true_idx = true_idx[np.where(true_labels>=0)]

        ## Get the clusters label values
        true_unique_labels = set(true_labels)

        true_core_samples_mask = np.zeros_like(true_labels, dtype=bool)

        true_core_samples_mask[true_idx] = True

        ## Get the colors for each predicted cluster from the colors array 
        true_colors = colors[:len(true_unique_labels)]
        
        
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## DEFINE THE PLOT STRUCTURE AND STYLES
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        d_marker_style = {
            -1:"o",
            0:"o",
            1:"v",
            2:"s",
            3:"^",
            4:"*",
            5:"<",
            6:"P",
            7:">",
            8:"X",
            9:"D",
            10:"o",
            11:"v",
            12:"s",
            13:"^",
            14:"*",
            15:"<",
            16:"P",
            17:">",
            18:"X",
            19:"D",
        }      
        
        fsize_ax = 16
        ## ============================================================================================================
        ## Plotting
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=False, tight_layout=True)
        ax1, ax2 = axs

        ax1.set_xlabel("Z tracks reco [mm]", fontsize=fsize_ax)
        ax1.set_ylabel("Z PV predicted [mm]", fontsize=fsize_ax)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fsize_ax)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fsize_ax)

        ax2.set_xlabel("Z tracks reco [mm]", fontsize=fsize_ax)
        ax2.set_ylabel("Z PV true [mm]", fontsize=fsize_ax)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fsize_ax)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fsize_ax)

        ## Slightly enlarge the plot boundaries
        dz = (zmax-zmin)*0.1
        ax1.set_xlim(zmin-dz,zmax+dz)
        ax1.set_ylim(zmin-dz,zmax+dz)
        ax2.set_xlim(zmin-dz,zmax+dz)
        ax2.set_ylim(zmin-dz,zmax+dz)

        ## Draw diagonal lines 
        ax1.plot(ax1.get_xlim(), ax1.get_ylim(), ls="--", c=".3")
        ax2.plot(ax2.get_xlim(), ax2.get_ylim(), ls="--", c=".3")

        ## ------------------------------------------------------------        
        ## DRAW THE PREDICTED DATA HERE
        ## ------------------------------------------------------------        
        for k, col in zip(pred_unique_labels, pred_colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = pred_labels == k

            xy = X_pred[class_member_mask & pred_core_samples_mask]
            #xy = X_pred_label[class_member_mask & pred_core_samples_mask]
            #ax1.plot(
            ax1.errorbar(
                xy[:, 2],
                xy[:, 8],
                xerr=xy[:, 5],
                marker=d_marker_style[k],
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markeredgewidth=0.5,
                markersize=5,
                ecolor=tuple(col),
                capsize=3,
                elinewidth=1,
                ls='none'
            )
            ## -----------------------------------------------------
            ## Draw lines and bands for the predicted PVs position
            if pred_pvs_info and k>=0:
                X_plt     = pred_pvs_info[int(k)][2]
                X_plt_err = 3*pred_pvs_info[int(k)][5]

                #print("X_plt =",X_plt)
                #print("X_plt-X_plt_err =",X_plt-X_plt_err)
                #print("X_plt+X_plt_err =",X_plt+X_plt_err)
                if verbose:
                    print("pred_unique_labels",pred_unique_labels)
                    print("k",k)
                info_pred = tracks_weighted_pred_pos(xy[:, 8],
                                                     xy[:, 5],
                                                     min_sigma_z,
                                                     verbose=verbose)

                Y_plt = info_pred[0]
                if verbose:
                    print("Weighted pred z PV (tracks) =",X_plt)
                    print("Weighted pred z PV (gnn)    =",Y_plt)

                ax1.axvspan(X_plt-X_plt_err, X_plt+X_plt_err, 
                            #ymin=ax1.get_ylim()[0], 
                            #ymax=Y_plt, 
                            alpha=0.5, 
                            facecolor=tuple(col))                

                ax1.vlines(x     = [X_plt], 
                           ymin  = ax1.get_ylim()[0], 
                           ymax  = Y_plt, 
                           colors=tuple(col), 
                           ls='--', 
                           lw=1)


                #ax1.hlines(y    = [Y_plt], 
                #           xmin = ax1.get_xlim()[0], 
                #           xmax = X_plt,
                #           colors=tuple(col), ls='--', lw=2) 
            
            xy = X_pred[class_member_mask & ~pred_core_samples_mask]
            ax1.plot(
                xy[:, 2],
                xy[:, 8],
                d_marker_style[k],
                markerfacecolor="white",
                #markerfacecolor=tuple(col),
                markeredgecolor=tuple(col),
                markersize=4,
            )
        ## ------------------------------------------------------------        
        ## DRAW THE TRUE DATA HERE
        ## ------------------------------------------------------------        
        for k, col in zip(true_unique_labels, true_colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = true_labels == k

            xy = X_true[class_member_mask & true_core_samples_mask]
            #xy = X_true_label[class_member_mask & true_core_samples_mask]
            #ax2.plot(
            ax2.errorbar(
                xy[:, 2],
                xy[:, 8],
                xerr=xy[:, 5],
                marker=d_marker_style[k],
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markeredgewidth=0.5,
                markersize=5,
                ecolor=tuple(col),
                capsize=3,
                elinewidth=1,
                ls='none'
            )

            ## -----------------------------------------------------
            ## Draw lines and bands for the true PVs position on the ax1 subplot!
            if true_pvs_info and k>=0:
                X_plt_true     = true_pvs_info[int(k)][2]
                X_plt_true_err = 1*true_pvs_info[int(k)][5]

                #print("X_plt =",X_plt)
                #print("X_plt-X_plt_err =",X_plt-X_plt_err)
                #print("X_plt+X_plt_err =",X_plt+X_plt_err)
                info_true = tracks_weighted_pred_pos(xy[:, 8],
                                                     xy[:, 5],
                                                     min_sigma_z,
                                                     verbose=verbose)

                Y_plt_true = info_true[0]
                if verbose:
                    print("Weighted true z PV (tracks) =",X_plt_true)
                    print("Weighted true z PV (gnn)    =",Y_plt_true)

                ## True from MC PV info
                ax1.vlines(x     = [Y_plt_true], 
                           ymin  = ax1.get_ylim()[0], 
                           ymax  = Y_plt_true, 
                           colors="black", 
                           ls='--', 
                           lw=1)
                
                ## True from actual weighted tracks info using truth level association
                ax1.vlines(x     = [X_plt_true], 
                           ymin  = ax1.get_ylim()[0], 
                           ymax  = Y_plt_true, 
                           colors="black", 
                           #ls='--', 
                           lw=1)
                
                
            xy = X_true[class_member_mask & ~true_core_samples_mask]
            #xy = X_true_label[class_member_mask & ~true_core_samples_mask]
            ax2.plot(
                xy[:, 2],
                xy[:, 8],
                d_marker_style[k],
                markerfacecolor=tuple(col),
                markeredgecolor=tuple(col),
                markersize=4,
            )

        fig.suptitle(f"Event #%d ; Group #%d with z in [%2.f, %.2f]"%(iEvt,iGroup,zmin,zmax), fontsize=16)
        
        plt_name = "/data/home/sakar/ML_dir/GNN/Clustering/special_cases/Model_iter0_11features_"
        plt_name += "Evt%d_Group%d"%(iEvt, iGroup)
        plt_name += "_epoch50"
                
        plt.savefig("%s.pdf"%plt_name)
        plt.savefig("%s.png"%plt_name)
        
        plt.show()
        ## -------------------------------------------------------
        ## END OF THE PLOTTING SECTION
        ## -------------------------------------------------------
        
    return pred_pvs_info, n_tracks_true_vis
