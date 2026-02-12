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
from torch.utils.data import TensorDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

import scipy
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import cosine as sp_cos

from sklearn.preprocessing  import normalize
from torch_geometric.utils  import dense_to_sparse, to_edge_index, to_torch_coo_tensor, to_dense_adj
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader

from tools.ToolBox import norm_cdf, get_res

from utils.utilities import checkFileExists, DummyTqdmFile, tqdm_redirect, Timer
from utils.utilities import get_split_slices, Get_ak_from_hdf5

## ========================================================================================
def get_closest_round_z_val(x,template):
    val=x-np.floor(x)    
    return template[(np.abs(template - val)).argmin()]+np.floor(x)

## ========================================================================================
def compute_variable_range_target_hists(z_min, z_max, l_pvs_z, l_pvs_n, global_configs, verbose=False):
    
    template = [0.0,0.25,0.5,0.75,1.0]

    ## Get the resolutions for all vertexes
    pv_res   = get_res(l_pvs_n, global_configs)

    res_min = 0.025 ## [mm]
    #res_min = 1.5 ## [mm]
    ## The vertexes true resolution as a function of nTracks will be increased 
    ## if res < 150 microns such that the target peaks will get more narrow and 
    ## the areas larger as the number of tracks increases.
    pv_res_scale = np.where((res_min / pv_res) > 1, (res_min / pv_res), 1)
    #print("pv_res_scale",pv_res_scale)

    z_min = np.floor(z_min)
    z_max = np.floor(z_max)
    #print("z_min",z_min)
    #print("z_max",z_max)
    
    starting_z_hist = get_closest_round_z_val(z_min, template)
    ending_z_hist   = get_closest_round_z_val(z_max, template)

    #print("starting_z_hist",starting_z_hist)
    #print("ending_z_hist",ending_z_hist)
    
    z_range = ending_z_hist - starting_z_hist    
    #print("z_range",z_range)
    
    bin_width = 0.02 ## [mm]
    #bin_width = 1.00 ## [mm]

    n_bins_target_hist = int(z_range / bin_width)
    #print("n_bins_target_hist: ",n_bins_target_hist)
    
    ## Define the z_range over n_bins_target_hist
    zRange_over_nBins_ratio = int(n_bins_target_hist / z_range)
    #print("zRange_over_nBins_ratio",zRange_over_nBins_ratio)
    
    ## Bins width definition
    #bin_width = z_range/n_bins_target_hist
    half_bin_width = bin_width/2.
    #print("half_bin_width",half_bin_width)
    
    ## Edges of the bins definition
    edges = np.array([-half_bin_width, half_bin_width])
    #print("edges",edges)
    ## Range of z values of the target hists
    zvals_range = (starting_z_hist + half_bin_width, ending_z_hist - half_bin_width)
    #print("zvals_range",zvals_range)

    ## Get the bin number corresponding to the PV position in Z: 
    pv_bin = ak.values_astype(np.floor((l_pvs_z - zvals_range[0]) * zRange_over_nBins_ratio), "int16")
    #print("pv_bin",pv_bin)

    ## This is a placeholder to hold the values of the vertexes peak structure
    ## +/- 5 bins around peak maximum seems a good compromise, as for more bins the 
    ## hist values become quickly tiny. 
    nBins_around_vtx_peak = 40
    vtx_bins = np.arange(-nBins_around_vtx_peak, nBins_around_vtx_peak+1)
    #print("vtx_bins",vtx_bins)
    
    ## Define the bins basis for which the normal cumulative distribution will be evaluated
    ## For each vertex, the vxt position (in z basis) will be added
    ## The output is a set of two list values corresponding to the equivalent of bins edges shifted by one bin
    vtx_z_bins_ranges = bin_width * vtx_bins[np.newaxis, :] + edges[:, np.newaxis] + zvals_range[0]
    #print("vtx_z_bins_ranges",vtx_z_bins_ranges)

    ## Define the target histogram
    targetHists     = np.zeros([n_bins_target_hist], dtype=np.float16)
    targetHists_idx = np.arange(n_bins_target_hist, dtype=np.float16)
    
    ## Looping over the vertexes
    for iVtx in range(len(l_pvs_z)):
        
        vtx_z         = l_pvs_z[iVtx]
        vtx_bin       = pv_bin[iVtx]
        vtx_res       = pv_res[iVtx]
        vtx_res_scale = pv_res_scale[iVtx]
        if verbose:
            print("vtx_z",vtx_z)
            print("vtx_bin",vtx_bin)
            print("vtx_res",vtx_res)
            print("vtx_res_scale",vtx_res_scale)
        
        ## Get the bins like values of z to evaluate the normal cumulative distribution
        vtx_z_bins    = (vtx_bin / zRange_over_nBins_ratio + vtx_z_bins_ranges)
        if verbose:
            print("vtx_z_bins",vtx_z_bins)

        ## Get the values of the normal cumulative ditribution for the two set of vtx_z_bins
        norm_cdf_vals = norm_cdf(vtx_z, vtx_res, vtx_z_bins)
        if verbose:
            print("norm_cdf_vals",norm_cdf_vals)
        
        ## Take the difference of norm_cdf_Vals to get the normal PDF values as vtx_peak_Vals
        ## The vtx_peak_vals are further scaled based on the vtx_res_scale values defined above

        vtx_peak_vals = (norm_cdf_vals[1] - norm_cdf_vals[0])*vtx_res_scale
        
        vtx_filling_bins = vtx_bins + vtx_bin
        vtx_filling_bins = vtx_filling_bins[np.where((vtx_filling_bins<=targetHists_idx[-1]) & (vtx_filling_bins>=targetHists_idx[0]))]
        if verbose:
            print("vtx_peak_vals",vtx_peak_vals)
            print("vtx_bins + vtx_bin",vtx_bins + vtx_bin)
            print("vtx_filling_bins",vtx_filling_bins)
            
        targetHists[vtx_filling_bins] += vtx_peak_vals
            
    return targetHists

## ========================================================================================
def graph_intersection(pred_graph,
                       truth_graph,
                       device,
                       using_weights=False,
                       weights_bidir=None):
    '''
    if device is None:
        device = default_device
    elif str(device) == "cuda:0":
        device = "cuda"
    '''
    
    if pred_graph.numel() > 0:
        pred_graph_max = pred_graph.max().item()
    else:
        pred_graph_max = 0

    if truth_graph.numel() > 0:
        truth_graph_max = truth_graph.max().item()
    else:
        truth_graph_max = 0

    array_size = max(pred_graph_max, truth_graph_max) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = scipy.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = scipy.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list
        del l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = (
        torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col]))
        .long()
        .to(device)
    )
    y = torch.from_numpy(e_intersection.data > 0).to(device)
    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y
    
    
## ========================================================================================
def compare_adj(pred, true):
    if len(pred)==len(true):
        return torch.tensor(np.where((pred.flatten()==true.flatten())==True, 1, 0))
    else:
        print("ERROR:: predicted and true edge dense adjacency matrices with different size: %s vs %s"%(len(pred),len(true)))
        return None
    
## ========================================================================================
def get_true_edge_indexes(tracks_pv_key):
    ## Change all non associated tracks (value==-1) to it's: index*-1
    tracks_pv_key[np.where(tracks_pv_key <0)]= -1*np.where(tracks_pv_key <0)[0]
    ## Shift by one all tracks <-> PV association key (for key value>=0)
    tracks_pv_key[np.where(tracks_pv_key >=0)] = tracks_pv_key[np.where(tracks_pv_key >=0)]+1
    ## Make a matrix of shape [nTracks,nTracks] with ones for tracks with the same PV key
    ## and any other value for tracks with different key association
    mat = np.array(tracks_pv_key[:, np.newaxis]) / np.array((tracks_pv_key[np.newaxis, :]))
    ## Convert all values different from 1 to 0.
    mat_ones = np.where(mat == 1, 1, 0)
    ## Convert into torch tensor format and assign as dense adjacency matrix
    adj = torch.tensor(mat_ones)
    ## Get the sparse matrix from dense_to_sparse pytorch method
    sparse  = dense_to_sparse(adj)
    '''coo_tensor = to_torch_coo_tensor(sparse[0])'''
    return sparse, adj#, coo_tensor
    
## ========================================================================================
def connected_components_cpu_np(edge_index, max_node_idx: int):
    """Apply the connected components algorithm on CPU."""
    if edge_index.shape[0]:
        import scipy.sparse as sps

        sparse_edges = sps.coo_matrix((np.ones(edge_index[0].shape),(edge_index[0], edge_index[1]),),(max_node_idx + 1, max_node_idx + 1),)

        _, candidate_labels = sps.csgraph.connected_components(sparse_edges, directed=False, return_labels=True)

        return pd.DataFrame({"track_idx": np.arange(max_node_idx + 1),
                             "group_id": candidate_labels,
                            })
    else:
        return pd.DataFrame({"track_idx": [],
                             "group_id": [],
                            })   
    
## ========================================================================================
def get_sp_adj(x,y,z,max_d):
    m_coords    = np.column_stack((x,y,z))
    m_dist      = cdist(m_coords, m_coords)
    m_dist_filt = np.where(m_dist < max_d, 1, 0)
    m_adj       = torch.tensor(m_dist_filt)
    m_sp_adj    = dense_to_sparse(m_adj)
    return m_sp_adj, m_adj

## ========================================================================================
def tracks_isolation(tracks_z, tracks_sz, min_dz=0.5, max_sz=5):
    tracks_iso = np.ones(ak.to_numpy(tracks_z).shape)
        
    for iTrack in range(0,len(tracks_z)-1):
        dz = tracks_z[iTrack+1]-tracks_z[iTrack]
        if dz<min_dz:
            tracks_iso[iTrack+1] = 0
            tracks_iso[iTrack]   = 0
            
    for iTrack in range(0,len(tracks_z)):
        if tracks_sz[iTrack]>max_sz:
            tracks_iso[iTrack]   = 1
        
    return tracks_iso

## ========================================================================================
def get_GNN_data(iEvt,
                 group_idx,
                 global_configs,
                 nFeatures,
                 tracks_x,  tracks_y,  tracks_z, 
                 tracks_sx, tracks_sy, tracks_sz, 
                 rho_xy, rho_xz, rho_yz,
                 recon_tx, recon_ty,
                 true_links,
                 tracks_cat,
                 tracks_ov_x,
                 tracks_ov_y,
                 tracks_ov_z,
                 true_pv_x,
                 true_pv_y,
                 true_pv_z,
                 true_pv_n,
                 true_pv_key,
                 edge_construction,
                 verbose=False):

    ## ----------------------------------------------------
    ## Get the configuration parameters
    min_d            = global_configs["min_dist_group_tracks"]
    
    doGroups         = global_configs["doGroups"]
    if not doGroups:
        min_d = (global_configs["z_max"]-global_configs["z_min"])
    
    addSecondVertex  = global_configs["addSecondVertex"]
    
    iso_d            = global_configs["min_dist_iso_track"]
    max_sz           = global_configs["max_sigma_z"]
    min_tracks_group = global_configs["group_min_n_tracks"]
    min_d_group      = global_configs["min_dist_within_group_tracks"]
    
    ## ----------------------------------------------------
    ## Create the original tracks index
    original_idx = np.arange(0, len(tracks_x), 1)
        
    FP_TYPE = np.float32
            
    tracks_x   = ak.to_numpy(tracks_x).astype(FP_TYPE)
    tracks_y   = ak.to_numpy(tracks_y).astype(FP_TYPE)
    tracks_z   = ak.to_numpy(tracks_z).astype(FP_TYPE)
    tracks_sx  = ak.to_numpy(tracks_sx).astype(FP_TYPE)
    tracks_sy  = ak.to_numpy(tracks_sy).astype(FP_TYPE)
    tracks_sz  = ak.to_numpy(tracks_sz).astype(FP_TYPE)
    if nFeatures == 9:
        rho_xy     = ak.to_numpy(rho_xy).astype(FP_TYPE)
        rho_xz     = ak.to_numpy(rho_xz).astype(FP_TYPE)
        rho_yz     = ak.to_numpy(rho_yz).astype(FP_TYPE)
    if nFeatures == 11:
        rho_xy     = ak.to_numpy(rho_xy).astype(FP_TYPE)
        rho_xz     = ak.to_numpy(rho_xz).astype(FP_TYPE)
        rho_yz     = ak.to_numpy(rho_yz).astype(FP_TYPE)
        recon_tx   = ak.to_numpy(recon_tx).astype(FP_TYPE)
        recon_ty   = ak.to_numpy(recon_ty).astype(FP_TYPE)
    true_links = ak.to_numpy(true_links).astype(FP_TYPE)
    
    tracks_cat     = ak.to_numpy(tracks_cat).astype(FP_TYPE)
    tracks_ov_x    = ak.to_numpy(tracks_ov_x).astype(FP_TYPE)
    tracks_ov_y    = ak.to_numpy(tracks_ov_y).astype(FP_TYPE)
    tracks_ov_z    = ak.to_numpy(tracks_ov_z).astype(FP_TYPE)
    
    true_pv_x   = ak.to_numpy(true_pv_x).astype(FP_TYPE)
    true_pv_y   = ak.to_numpy(true_pv_y).astype(FP_TYPE)
    true_pv_z   = ak.to_numpy(true_pv_z).astype(FP_TYPE)
    true_pv_n   = ak.to_numpy(true_pv_n).astype(FP_TYPE)
    true_pv_key = ak.to_numpy(true_pv_key).astype(FP_TYPE)
    

    ## ----------------------------------------------------
    ## Filter isolated tracks and tracks with large Z uncertainty 
    filt = tracks_isolation(tracks_z, tracks_sz, iso_d, max_sz)
    
    '''
    print("Initial number of tracks",len(tracks_x))
    '''
    
    tracks_x  = tracks_x[filt==False]
    tracks_y  = tracks_y[filt==False]
    tracks_z  = tracks_z[filt==False]
    tracks_sx = tracks_sx[filt==False]
    tracks_sy = tracks_sy[filt==False]
    tracks_sz = tracks_sz[filt==False]
    if nFeatures == 9:
        rho_xy    = rho_xy[filt==False]
        rho_xz    = rho_xz[filt==False]
        rho_yz    = rho_yz[filt==False]
    if nFeatures == 11:
        rho_xy    = rho_xy[filt==False]
        rho_xz    = rho_xz[filt==False]
        rho_yz    = rho_yz[filt==False]
        recon_tx  = recon_tx[filt==False]
        recon_ty  = recon_ty[filt==False]
    '''
    print("Filtered number of tracks",len(tracks_x))
    '''
    
    tracks_cat     = tracks_cat[filt==False]
    tracks_ov_x    = tracks_ov_x[filt==False]
    tracks_ov_y    = tracks_ov_y[filt==False]
    tracks_ov_z    = tracks_ov_z[filt==False]
    
    original_idx = original_idx[filt==False]
    
    '''
    print("tracks_x[-2:]",tracks_x[-2:])
    print("tracks_y[-2:]",tracks_y[-2:])
    print("tracks_z[-2:]",tracks_z[-2:])
    print("tracks_sx[-2:]",tracks_sx[-2:])
    print("tracks_sy[-2:]",tracks_sy[-2:])
    print("tracks_sz[-2:]",tracks_sz[-2:])
    '''
    
    ## ----------------------------------------------------
    ## Get the sparse adjacency matrix
    m_sp_adj, m_adj = get_sp_adj(tracks_x, tracks_y, tracks_z, min_d)
    '''
    print("m_sp_adj",m_sp_adj)
    print("m_adj",m_adj)
    '''
    
    ## ----------------------------------------------------
    ## Get the linked tracks using the connected_components
    tracks_links = connected_components_cpu_np(m_sp_adj[0],len(tracks_x)-1)
    
        
    ## ----------------------------------------------------
    ## Get the indices of the tracks forming groups of more than "N=min_tracks_group" tracks
    grouped_tracks_idx = tracks_links.sort_index().groupby('group_id').filter(lambda group: len(group) >= min_tracks_group)["track_idx"].tolist()

    #print("grouped_tracks_idx",grouped_tracks_idx)
    #print("tracks_links[group_id]",tracks_links["group_id"].tolist())
    
    ## ----------------------------------------------------
    ## Get the predicted pvs identifier (group of more than min_tracks_group tracks)
    groups_id = tracks_links["group_id"][grouped_tracks_idx].unique()
    
    #print("groups_id",groups_id)
        
    l_data = []
    for iGroup in range(len(groups_id)):
               
        ## ----------------------------------------------------
        ## Get the tracks index corresponding to the considered group of tracks
        tracks_idx = tracks_links["track_idx"][tracks_links["group_id"]==groups_id[iGroup]].tolist()
        
        start = tracks_idx[0]
        end   = tracks_idx[-1]+1
        
        ## ----------------------------------------------------
        ## Get the group tracks info
        tracks_x_group  = tracks_x[tracks_idx]
        tracks_y_group  = tracks_y[tracks_idx]
        tracks_z_group  = tracks_z[tracks_idx]
        tracks_sx_group = tracks_sx[tracks_idx]
        tracks_sy_group = tracks_sy[tracks_idx]
        tracks_sz_group = tracks_sz[tracks_idx]
        if nFeatures == 9:
            rho_xy_group    = rho_xy[tracks_idx]
            rho_xz_group    = rho_xz[tracks_idx]
            rho_yz_group    = rho_yz[tracks_idx]
        if nFeatures == 11:
            rho_xy_group    = rho_xy[tracks_idx]
            rho_xz_group    = rho_xz[tracks_idx]
            rho_yz_group    = rho_yz[tracks_idx]
            recon_tx_group  = recon_tx[tracks_idx]
            recon_ty_group  = recon_ty[tracks_idx]
        
        tracks_cat_group     = tracks_cat[tracks_idx]
        tracks_ov_x_group    = tracks_ov_x[tracks_idx]
        tracks_ov_y_group    = tracks_ov_y[tracks_idx]
        tracks_ov_z_group    = tracks_ov_z[tracks_idx]
        
        ## ----------------------------------------------------
        ## Get the tracks <-> pv keys from the original track index array
        true_links_group = true_links[original_idx[tracks_idx]]
        
        #print("true_links_group",true_links_group)
        
        ## ----------------------------------------------------
        ## Get the group limits in z
        z_extension = 1. # in mm
        z_min_group = int(np.floor(tracks_z_group[0] - z_extension))
        z_max_group = int(np.floor(tracks_z_group[-1]+ z_extension))                    
        
        ## ----------------------------------------------------
        ## Get the group range in z
        z_range_group = z_max_group - z_min_group
                
        #print("_z_min = ",_z_min)
        #print("_z_max = ",_z_max)
        #print("true_pv_z : ",true_pv_z)
        ## ----------------------------------------------------
        ## Get the indixes of the true PVs in the range
        true_pvs_idx_in_group_range = np.array(np.where(np.logical_and(true_pv_z>=z_min_group, true_pv_z<=z_max_group)))[0].tolist()
        #print("true_pvs_idx_in_group_range",true_pvs_idx_in_group_range)
        
        ## ----------------------------------------------------
        ## Get the Z values of the true PVs in the range
        pvs_x_group = true_pv_x[true_pvs_idx_in_group_range]
        pvs_y_group = true_pv_y[true_pvs_idx_in_group_range]
        pvs_z_group = true_pv_z[true_pvs_idx_in_group_range]
        ## Get the N tracks of the true PVs in the range
        pvs_n_group = true_pv_n[true_pvs_idx_in_group_range]
        pvs_key_group = true_pv_key[true_pvs_idx_in_group_range]
        #print("_true_pvs_key",_true_pvs_key)
        #print("_true_pvs_z",_true_pvs_z)
        
                        
        ## ======================================================================
        ## Get the coordinates of each track in current group
        m_coords    = np.column_stack((tracks_x_group,tracks_y_group,tracks_z_group))            
        ## ----------------------------------------------------
        ## Get the number of tracks in current group
        n_tracks = len(tracks_x_group)
        ## ----------------------------------------------------
        ## Set the array of distance of each node wrt it's PV
        y_dist = np.zeros(n_tracks, dtype=np.float32)+0.00001
        ## ----------------------------------------------------
        ## Construct the tracks true PV position
        y_pos = np.zeros(3*n_tracks, dtype=np.float32)
        y_pos.shape = (n_tracks,3)
        ## ----------------------------------------------------
        ## Construct the normalized tracks true PV position
        y_pos_norm = np.zeros(3*n_tracks, dtype=np.float32)
        y_pos_norm.shape = (n_tracks,3)

        ## ----------------------------------------------------
        ## Construct the tracks true PV AND SV position
        y_pos_ov_added = np.zeros(6*n_tracks, dtype=np.float32)
        y_pos_ov_added.shape = (n_tracks,6)
        ## ----------------------------------------------------
        ## Construct the normalized tracks true PV AND SV position
        y_pos_ov_added_norm = np.zeros(6*n_tracks, dtype=np.float32)
        y_pos_ov_added_norm.shape = (n_tracks,6)
        
        ## ======================================================================
        ## ----------------------------------------------------
        ## Construct the tracks PV position
        
        for iTrack in range(len(true_links_group)):   
            if true_links_group[iTrack]>=0:
                ## ----------------------------------------------------
                ## track is linked, so simply get the coordinates of the linked PV
                pos_filt = np.where(true_links_group[iTrack]==true_pv_key)
                y_pos[iTrack][0] = true_pv_x[pos_filt]
                y_pos[iTrack][1] = true_pv_y[pos_filt]
                y_pos[iTrack][2] = true_pv_z[pos_filt]
                #print("y_pos[%d][2] = %.3f"%(iTrack,y_pos[iTrack][2]))
                
                y_pos_norm[iTrack][0] = y_pos[iTrack][0]
                y_pos_norm[iTrack][1] = y_pos[iTrack][1]
                ## ----------------------------------------------------
                ## shifting by z_min and dividing by the z range of the group
                y_pos_norm[iTrack][2] = (y_pos[iTrack][2]-z_min_group)/z_range_group
                #print("true_links[%s]: %s"%(iTrack,true_links[iEvt][iTrack]))
                #print("PV_z = ",pv_z[iEvt][np.where(true_links[iEvt][iTrack]==pv_key[iEvt])])
                
                y_pos_ov_added[iTrack][0] = y_pos[iTrack][0]
                y_pos_ov_added[iTrack][1] = y_pos[iTrack][1]
                y_pos_ov_added[iTrack][2] = y_pos[iTrack][2]

                y_pos_ov_added_norm[iTrack][0] = y_pos_norm[iTrack][0]
                y_pos_ov_added_norm[iTrack][1] = y_pos_norm[iTrack][1]
                y_pos_ov_added_norm[iTrack][2] = y_pos_norm[iTrack][2]

                y_pos_ov_added[iTrack][3] = tracks_ov_x_group[iTrack]
                y_pos_ov_added[iTrack][4] = tracks_ov_y_group[iTrack]
                y_pos_ov_added[iTrack][5] = tracks_ov_z_group[iTrack]

                y_pos_ov_added_norm[iTrack][3] = tracks_ov_x_group[iTrack]
                y_pos_ov_added_norm[iTrack][4] = tracks_ov_y_group[iTrack]
                y_pos_ov_added_norm[iTrack][5] = (tracks_ov_z_group[iTrack]-z_min_group)/z_range_group
                
            else:
                ## ----------------------------------------------------
                ## WARNING !!!
                ## track is not linked, so get the coordinates of the closest true PV
                ## this happens rarely so should not be an issue...
                closest_pv_idx = np.argmin(abs(true_pv_z - tracks_z_group[iTrack]))
                y_pos[iTrack][0] = true_pv_x[closest_pv_idx]
                y_pos[iTrack][1] = true_pv_y[closest_pv_idx]
                y_pos[iTrack][2] = true_pv_z[closest_pv_idx]
                
                y_pos_norm[iTrack][0] = y_pos[iTrack][0]
                y_pos_norm[iTrack][1] = y_pos[iTrack][1]
                ## ----------------------------------------------------
                ## shifting Z by z_min and dividing by the z range of the group
                y_pos_norm[iTrack][2] = (y_pos[iTrack][2]-z_min_group)/z_range_group
                                
                
                y_pos_ov_added[iTrack][0] = y_pos[iTrack][0]
                y_pos_ov_added[iTrack][1] = y_pos[iTrack][1]
                y_pos_ov_added[iTrack][2] = y_pos[iTrack][2]

                y_pos_ov_added_norm[iTrack][0] = y_pos_norm[iTrack][0]
                y_pos_ov_added_norm[iTrack][1] = y_pos_norm[iTrack][1]
                y_pos_ov_added_norm[iTrack][2] = y_pos_norm[iTrack][2]

                y_pos_ov_added[iTrack][3] = y_pos[iTrack][0]
                y_pos_ov_added[iTrack][4] = y_pos[iTrack][1]
                y_pos_ov_added[iTrack][5] = y_pos[iTrack][2]

                y_pos_ov_added_norm[iTrack][3] = y_pos_norm[iTrack][0]
                y_pos_ov_added_norm[iTrack][4] = y_pos_norm[iTrack][1]
                y_pos_ov_added_norm[iTrack][5] = y_pos_norm[iTrack][2]
                
        y_pos = torch.tensor(y_pos)
        y_pos_norm = torch.tensor(y_pos_norm)

        y_pos_ov_added = torch.tensor(y_pos_ov_added)
        y_pos_ov_added_norm = torch.tensor(y_pos_ov_added_norm)
        
        ## ----------------------------------------------------
        ## Loop over each PV in the group to compute distance of each node to the PV
        for iPV in range(len(pvs_x_group)):
            #print("*********")
            #print("PV",iPV)            
            ipv_x = pvs_x_group[iPV]
            ipv_y = pvs_y_group[iPV]
            ipv_z = pvs_z_group[iPV]
            ## ----------------------------
            ## Get the PV coordinates
            m_coords_pv = np.column_stack((ipv_x,ipv_y,ipv_z))    
            ## ----------------------------
            ## Get the PV coordinates
            m_dist      = cdist(m_coords, m_coords_pv)
            y_dist += (1./m_dist).T[0]

        ## ----------------------------------------------------
        ## Normalize the distance so it's between 0 and 1
        y_dist_max = np.max(y_dist)
        if y_dist_max>0:
            y_dist = y_dist/y_dist_max
        
        y_dist = torch.tensor(y_dist)
        
        
        ## ======================================================================
        ## ----------------------------------------------------
        ## Get the group edge indices 
        if edge_construction==0:
            m_adj_group = np.ones(len(tracks_idx)*len(tracks_idx))
            m_adj_group.shape = (len(tracks_idx),len(tracks_idx))
            m_adj_group = torch.tensor(m_adj_group)
        elif edge_construction==1:            
            m_adj_group = m_adj[start:end,start:end]
        elif edge_construction==2:            
            ## ----------------------------------------------------
            ## Get the sparse adjacency matrix for the group
            m_sp_adj_group, m_adj_group = get_sp_adj(tracks_x_group, tracks_y_group, tracks_z_group, min_d_group)            
        ## ----------------------------------------------------
        ## Remove self-loops:
        m_adj_group = m_adj_group - np.eye(len(m_adj_group))
        edge_index_group = dense_to_sparse(m_adj_group)
        '''
        print(len(m_adj_group))
        '''

        if nFeatures == 6:
            ## ======================================================================
            ## Get the nodes input features
            tracksData_group = np.stack((tracks_x_group,
                                         tracks_y_group,
                                         tracks_z_group,
                                         tracks_sx_group,
                                         tracks_sy_group,
                                         tracks_sz_group,
                                        )
                                        ,axis=1)

            x_group = torch.tensor(tracksData_group)
            ## ======================================================================
            ## Get the nodes input features (with normalized z position)
            ## shifting Z by z_min and dividing by the z range of the group
            ## scaling the Z uncertainty by the z range of the group 
            tracksData_group_norm = np.stack((tracks_x_group,
                                              tracks_y_group,
                                              (tracks_z_group-z_min_group)/z_range_group,
                                              tracks_sx_group,
                                              tracks_sy_group,
                                              tracks_sz_group/z_range_group,
                                             )
                                             ,axis=1)

            x_group_norm = torch.tensor(tracksData_group_norm)
            
        elif nFeatures == 9:
            ## ======================================================================
            ## Get the nodes input features
            tracksData_group = np.stack((tracks_x_group,
                                         tracks_y_group,
                                         tracks_z_group,
                                         tracks_sx_group,
                                         tracks_sy_group,
                                         tracks_sz_group,
                                         rho_xy_group,
                                         rho_xz_group,
                                         rho_yz_group,
                                        )
                                        ,axis=1)

            x_group = torch.tensor(tracksData_group)
            ## ======================================================================
            ## Get the nodes input features (with normalized z position)
            ## shifting Z by z_min and dividing by the z range of the group
            ## scaling the Z uncertainty by the z range of the group 
            tracksData_group_norm = np.stack((tracks_x_group,
                                              tracks_y_group,
                                              (tracks_z_group-z_min_group)/z_range_group,
                                              tracks_sx_group,
                                              tracks_sy_group,
                                              tracks_sz_group/z_range_group,
                                              rho_xy_group,
                                              rho_xz_group,
                                              rho_yz_group,
                                             )
                                             ,axis=1)

            x_group_norm = torch.tensor(tracksData_group_norm)
            
        elif nFeatures == 11:
            ## ======================================================================
            ## Get the nodes input features
            tracksData_group = np.stack((tracks_x_group,
                                         tracks_y_group,
                                         tracks_z_group,
                                         tracks_sx_group,
                                         tracks_sy_group,
                                         tracks_sz_group,
                                         rho_xy_group,
                                         rho_xz_group,
                                         rho_yz_group,
                                         recon_tx_group,
                                         recon_ty_group,
                                        )
                                        ,axis=1)

            x_group = torch.tensor(tracksData_group)
            ## ======================================================================
            ## Get the nodes input features (with normalized z position)
            ## shifting Z by z_min and dividing by the z range of the group
            ## scaling the Z uncertainty by the z range of the group 
            tracksData_group_norm = np.stack((tracks_x_group,
                                              tracks_y_group,
                                              (tracks_z_group-z_min_group)/z_range_group,
                                              tracks_sx_group,
                                              tracks_sy_group,
                                              tracks_sz_group/z_range_group,
                                              rho_xy_group,
                                              rho_xz_group,
                                              rho_yz_group,
                                              recon_tx_group,
                                              recon_ty_group,
                                             )
                                             ,axis=1)

            x_group_norm = torch.tensor(tracksData_group_norm)
        
        
        if edge_construction==0:
            ## ==================================================
            ## Get the pred <-> true edge matching tensor
            # matching_tensor = compare_adj(m_adj_group,m_adj_true)
            
            if addSecondVertex:
                ## Append the data object to the list
                l_data.append(Data(x=x_group, 
                                   x_norm=x_group_norm,
                                   edge_index=edge_index_group[0], 
                                   #y_edge=matching_tensor, 
                                   evt=iEvt, evt_group_idx=iGroup, global_group_idx=group_idx, 
                                   z_min=z_min_group, z_max=z_max_group, 
                                   true_pvs_x = pvs_x_group, 
                                   true_pvs_y = pvs_y_group, 
                                   true_pvs_z = pvs_z_group, 
                                   true_pvs_n = pvs_n_group,
                                   y_dist     = y_dist,
                                   y_dist_max = y_dist_max,
                                   y_pos      = y_pos,
                                   y_pos_norm = y_pos_norm,
                                   y_pos_ov_added      = y_pos_ov_added,
                                   y_pos_ov_added_norm = y_pos_ov_added_norm,
                                   tracks_cat = torch.tensor(tracks_cat_group)
                                   #y_hist=pv_hist
                                  ))                
            else:
                ## Append the data object to the list
                l_data.append(Data(x=x_group, 
                                   x_norm=x_group_norm,
                                   edge_index=edge_index_group[0], 
                                   #y_edge=matching_tensor, 
                                   evt=iEvt, evt_group_idx=iGroup, global_group_idx=group_idx, 
                                   z_min=z_min_group, z_max=z_max_group, 
                                   true_pvs_x = pvs_x_group, 
                                   true_pvs_y = pvs_y_group, 
                                   true_pvs_z = pvs_z_group, 
                                   true_pvs_n = pvs_n_group,
                                   y_dist     = y_dist,
                                   y_dist_max = y_dist_max,
                                   y_pos      = y_pos,
                                   y_pos_norm = y_pos_norm,
                                   tracks_cat = torch.tensor(tracks_cat_group)
                                   #y_hist=pv_hist
                                  ))
        else: 
            ## ==================================================
            ## Get the pred <-> true edge matching tensor
            # edge_index_group, matching_tensor = graph_intersection(edge_index_group[0],y_group[0],device=None)

            if addSecondVertex:
                ## Append the data object to the list
                l_data.append(Data(x=x_group, 
                                   x_norm=x_group_norm,
                                   edge_index=edge_index_group[0], 
                                   #y_edge=matching_tensor, 
                                   evt=iEvt, evt_group_idx=iGroup, global_group_idx=group_idx, 
                                   z_min=z_min_group, z_max=z_max_group, 
                                   true_pvs_x = pvs_x_group, 
                                   true_pvs_y = pvs_y_group, 
                                   true_pvs_z = pvs_z_group, 
                                   true_pvs_n = pvs_n_group,
                                   y_dist     = y_dist,
                                   y_dist_max = y_dist_max,
                                   y_pos      = y_pos,
                                   y_pos_norm = y_pos_norm,
                                   y_pos_ov_added      = y_pos_ov_added,
                                   y_pos_ov_added_norm = y_pos_ov_added_norm,
                                   tracks_cat = torch.tensor(tracks_cat_group)
                                   #y_hist=pv_hist
                                  ))                
            else:            
                ## Append the data object to the list
                l_data.append(Data(x=x_group, 
                                   x_norm=x_group_norm,
                                   edge_index=edge_index_group[0], 
                                   #y_edge=matching_tensor, 
                                   evt=iEvt, evt_group_idx=iGroup, global_group_idx=group_idx, 
                                   z_min=z_min_group, z_max=z_max_group, 
                                   true_pvs_x = pvs_x_group, 
                                   true_pvs_y = pvs_y_group, 
                                   true_pvs_z = pvs_z_group, 
                                   true_pvs_n = pvs_n_group,
                                   y_dist     = y_dist,
                                   y_dist_max = y_dist_max,
                                   y_pos      = y_pos,
                                   y_pos_norm = y_pos_norm,
                                   tracks_cat = torch.tensor(tracks_cat_group)
                                   #y_hist=pv_hist
                                  ))
            
                
        ## Iterate the global index value
        group_idx+=1
        
        
    return l_data, group_idx


## ========================================================================================
def collectData_GNN(configs, l_files, l_slice_evts, split=False, verbose=False, **kargs):
    """
    Method to load data from HDF5 files for the GNN model 
    
    Arguments:
    
    - configs:     configuration dictionnary
    
    - files_prefix: list of files (only the prefix to be filled with final suffix for
                    retriving the tracks, KDE and target hists tensors
        
    - split:      boolean to switch between returning either a split dataset for training, 
                  validation and testing purposes
        
    - slice_evts: slice in terms of number of events. Is automatically converted into number 
                  of intervals if using them
                       
    """
    
    
    ## -----------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## -----------------------------------------------------------------
    ## Get the global configuration 
    global_configs = configs['global_configs'][Exp]      
    
    # -------------------------------------------------------------------------
    ## Get the edge construction value:
    ## 0: Fully connected edge_index (all tracks within group are automatically linked)
    ## 1: BUGGY ==> do not use for the moment
    ## 2: DEFAULT ==> use the pre-predicted edge_index to create the truth level edge_index
    edge_construction=global_configs["edge_construction"]

    # -------------------------------------------------------------------------
    ## Get the number of input features to use for the dataset construction:
    model_type = configs["model_type"]
    if "InteractionGNN_node_PV_position" in model_type:
        model_config = configs["models_config"]["InteractionGNN_node_PV_position"]
    else:
        model_config = configs["models_config"][model_type]
        
    nFeatures=model_config["n_InputFeatures"]
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## List of geometric Data object for each HDF5 file
    ## that will be used to build the DataLoader after
    ## the loop over the list of files
    l_Data = []
    
    ## In case spliting the data into train, validation and test samples
    l_Data_train = []
    l_Data_val   = []
    l_Data_test  = []    
        
    nEvts_total = 0
    nEvts_train = 0
    nEvts_val   = 0
    nEvts_test  = 0
        
        
    group_idx = 0
    evt_idx   = 0
    
    group_idx_train = 0
    evt_idx_train   = 0
    
    group_idx_val = 0
    evt_idx_val   = 0
    
    group_idx_test = 0
    evt_idx_test   = 0    
    
    for file_idx, file in enumerate(l_files):

        checkFileExists(file)
            
        print("")
        print("*"*100)
        print("*"*100)               
        with Timer(start="\n Collecting data from input hdf5 file \n"):
            print("   `--> ",file)
            print("")
            print("-"*100)
            print("")
            
            
            ## ------------------------------------------------------------
            ## Open input HDF5 file
            input_file = h5py.File(file, "r")
            
            
            # -------------------------------------------------------------------------
            # Get the tracks level information 
            poca_x = Get_ak_from_hdf5(input_file, "poca_x")            
            poca_y = Get_ak_from_hdf5(input_file, "poca_y")
            poca_z = Get_ak_from_hdf5(input_file, "poca_z")
            poca_A = Get_ak_from_hdf5(input_file, "poca_A")
            poca_B = Get_ak_from_hdf5(input_file, "poca_B")
            poca_C = Get_ak_from_hdf5(input_file, "poca_C")
            poca_D = Get_ak_from_hdf5(input_file, "poca_D")
            poca_E = Get_ak_from_hdf5(input_file, "poca_E")
            poca_F = Get_ak_from_hdf5(input_file, "poca_F")

            recon_tx  = Get_ak_from_hdf5(input_file, "recon_tx")
            recon_ty  = Get_ak_from_hdf5(input_file, "recon_ty")

            poca_sigma_x = 1./np.sqrt(poca_A)
            poca_sigma_y = 1./np.sqrt(poca_B)
            poca_sigma_z = 1./np.sqrt(poca_C)
            
            rho_xy = poca_D/np.sqrt(poca_A*poca_B)
            rho_xz = poca_E/np.sqrt(poca_A*poca_C)
            rho_yz = poca_F/np.sqrt(poca_B*poca_C)
            
            tracks_pv_key = Get_ak_from_hdf5(input_file, "recon_pv_key")
            tracks_cat    = Get_ak_from_hdf5(input_file, "recon_cat")
            tracks_ov_x   = Get_ak_from_hdf5(input_file, "recon_ov_x")
            tracks_ov_y   = Get_ak_from_hdf5(input_file, "recon_ov_y")
            tracks_ov_z   = Get_ak_from_hdf5(input_file, "recon_ov_z")
            
            pv_x = Get_ak_from_hdf5(input_file, "pv_x")
            pv_y = Get_ak_from_hdf5(input_file, "pv_y")
            pv_z = Get_ak_from_hdf5(input_file, "pv_z")
            pv_n = Get_ak_from_hdf5(input_file, "pv_ntracks")
            pv_key = Get_ak_from_hdf5(input_file, "pv_key")
                                    
            # -------------------------------------------------------------------------
            # Get the requested slice of events for this file
            slice_evts = l_slice_evts[file_idx]
            
            
            # -------------------------------------------------------------------------
            # Get the total number of events in the input data
            nEvts_data = len(poca_x)
                                   
            # -------------------------------------------------------------------------
            # Check if the requested number of events to be used is not larger than the actual
            # total number of events in the input file
            if slice_evts.start > nEvts_data:
                print("Slice starting event (%s) larger than the maximum number of available events in the input file (%s)"%(slice_evts.start,nEvts_data))
                sys.exit()

            if slice_evts.stop > nEvts_data:
                print("WARNING:: Slice stopping event (%s) larger than the maximum number of available events in the input file (%s)"%(slice_evts.stop,nEvts_data))
                print("Slice stopping event set back to %s"%(nEvts_data))
                slice_evts = slice(slice_evts.start,nEvts_data)

            if slice_evts.stop == -1:
                slice_evts = slice(slice_evts.start,nEvts_data)
                
            nEvts_total += (slice_evts.stop - slice_evts.start)

            # -------------------------------------------------------------------------
            # Slice_entries is simply the slice_evts
            slice_entries   = slice_evts

            if not split:
                for iEvt in range(slice_entries.start,slice_entries.stop):
                                        
                    data, group_idx = get_GNN_data(iEvt,
                                                   group_idx,
                                                   global_configs,
                                                   nFeatures,
                                                   poca_x[iEvt],
                                                   poca_y[iEvt],
                                                   poca_z[iEvt],
                                                   poca_sigma_x[iEvt],
                                                   poca_sigma_y[iEvt],
                                                   poca_sigma_z[iEvt],
                                                   rho_xy[iEvt],
                                                   rho_xz[iEvt],
                                                   rho_yz[iEvt],
                                                   recon_tx[iEvt],
                                                   recon_ty[iEvt],
                                                   tracks_pv_key[iEvt],
                                                   tracks_cat[iEvt],
                                                   tracks_ov_x[iEvt],
                                                   tracks_ov_y[iEvt],
                                                   tracks_ov_z[iEvt],
                                                   pv_x[iEvt],
                                                   pv_y[iEvt],
                                                   pv_z[iEvt],
                                                   pv_n[iEvt],
                                                   pv_key[iEvt],
                                                   edge_construction,
                                                   verbose)
                    evt_idx += 1
                    for idata in data:
                        l_Data.append(idata)
                        
            else:
                slice_entries_train, slice_entries_val, slice_entries_test = get_split_slices(slice_evts, 
                                                                                              configs["training_configs"]["train_split"], 
                                                                                              False)
                nEvts_train += (slice_entries_train.stop - slice_entries_train.start)
                nEvts_val   += (slice_entries_val.stop   - slice_entries_val.start)
                nEvts_test  += (slice_entries_test.stop  - slice_entries_test.start)

                # ------------------------------
                ## Start with the training data...
                for iEvt in range(slice_entries_train.start,slice_entries_train.stop):
                    data_train, group_idx = get_GNN_data(iEvt,
                                                   group_idx,
                                                   global_configs,
                                                   nFeatures,
                                                   poca_x[iEvt],
                                                   poca_y[iEvt],
                                                   poca_z[iEvt],
                                                   poca_sigma_x[iEvt],
                                                   poca_sigma_y[iEvt],
                                                   poca_sigma_z[iEvt],
                                                   rho_xy[iEvt],
                                                   rho_xz[iEvt],
                                                   rho_yz[iEvt],
                                                   recon_tx[iEvt],
                                                   recon_ty[iEvt],
                                                   tracks_pv_key[iEvt],
                                                   tracks_cat[iEvt],
                                                   tracks_ov_x[iEvt],
                                                   tracks_ov_y[iEvt],
                                                   tracks_ov_z[iEvt],
                                                   pv_x[iEvt],
                                                   pv_y[iEvt],
                                                   pv_z[iEvt],
                                                   pv_n[iEvt],
                                                   pv_key[iEvt],
                                                   edge_construction,
                                                   verbose)
                    evt_idx_train += 1
                    for idata_train in data_train:
                        l_Data_train.append(idata_train)
                # ------------------------------
                ## ...then the validation data....
                for iEvt in range(slice_entries_val.start,slice_entries_val.stop):
                    data_val, group_idx = get_GNN_data(iEvt,
                                                   group_idx,
                                                   global_configs,
                                                   nFeatures,
                                                   poca_x[iEvt],
                                                   poca_y[iEvt],
                                                   poca_z[iEvt],
                                                   poca_sigma_x[iEvt],
                                                   poca_sigma_y[iEvt],
                                                   poca_sigma_z[iEvt],
                                                   rho_xy[iEvt],
                                                   rho_xz[iEvt],
                                                   rho_yz[iEvt],
                                                   recon_tx[iEvt],
                                                   recon_ty[iEvt],
                                                   tracks_pv_key[iEvt],
                                                   tracks_cat[iEvt],
                                                   tracks_ov_x[iEvt],
                                                   tracks_ov_y[iEvt],
                                                   tracks_ov_z[iEvt],
                                                   pv_x[iEvt],
                                                   pv_y[iEvt],
                                                   pv_z[iEvt],
                                                   pv_n[iEvt],
                                                   pv_key[iEvt],
                                                   edge_construction,
                                                   verbose)
                    evt_idx_val += 1
                    for idata_val in data_val:
                        l_Data_val.append(idata_val)
                # ------------------------------
                ## ...and finish with the test data...
                for iEvt in range(slice_entries_test.start,slice_entries_test.stop):
                    data_test, group_idx = get_GNN_data(iEvt,
                                                   group_idx,
                                                   global_configs,
                                                   nFeatures,
                                                   poca_x[iEvt],
                                                   poca_y[iEvt],
                                                   poca_z[iEvt],
                                                   poca_sigma_x[iEvt],
                                                   poca_sigma_y[iEvt],
                                                   poca_sigma_z[iEvt],
                                                   rho_xy[iEvt],
                                                   rho_xz[iEvt],
                                                   rho_yz[iEvt],
                                                   recon_tx[iEvt],
                                                   recon_ty[iEvt],
                                                   tracks_pv_key[iEvt],
                                                   tracks_cat[iEvt],
                                                   tracks_ov_x[iEvt],
                                                   tracks_ov_y[iEvt],
                                                   tracks_ov_z[iEvt],
                                                   pv_x[iEvt],
                                                   pv_y[iEvt],
                                                   pv_z[iEvt],
                                                   pv_n[iEvt],
                                                   pv_key[iEvt],
                                                   edge_construction,
                                                   verbose)
                    evt_idx_test += 1
                    for idata_test in data_test:
                        l_Data_test.append(idata_test)
                                                
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## Get the batch size from the configs
    batchSize = configs["training_configs"]["batch_size"]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ## By default, split the dataset in three data sets for training, validation and testing
    if split:        
        # -------------------------------------------------------------------------
        ## Transform the split datasets into three Data loader objects
        if verbose:
            print("")
            print("="*100)
            print("Constructing trainging, validation and testing DataLoaders from TensorDatasets")
        train_loader = DataLoader(dataset=l_Data_train, batch_size=batchSize, **kargs)
        val_loader   = DataLoader(dataset=l_Data_val,   batch_size=batchSize, **kargs)
        test_loader  = DataLoader(dataset=l_Data_test,  batch_size=batchSize, **kargs)

        # -------------------------------------------------------------------------
        ## Return the split input Data loaders
        return train_loader, val_loader, test_loader
        
    else:
        # -------------------------------------------------------------------------
        ## Create the data loader with the batch_size from configuration parameters file
        if verbose:
            print("")
            print("="*100)
            print("Constructing DataLoader from TensorDataset")
        loader = DataLoader(dataset=l_Data, batch_size=batchSize, **kargs)        
        ## Return the entire input data. This way one can use different data files for training, validation or testing  
        return loader        
                