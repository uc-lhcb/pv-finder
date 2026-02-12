import sys
import os
import pprint

import pandas as pd
import numpy as np
import torch

from utils.utilities import Get_ak_from_hdf5
import h5py as h5py


from utils.utilities import from_bins_to_z, get_device_from_model
from utils.plotting_utils import plot_z_hist
from utils.efficiency_utils import ValueSet, ValueSet_GNN, eval_efficiency
from utils.clustering_utils import *

def get_GNN_efficiency(configs, outPutFolder):
    
    dict_tag_event = str(outPutFolder / f'GNN_perf_event{configs["model_class"]}.npy')
    d_result_iEvt = np.load(dict_tag_event,allow_pickle='TRUE').item()
    
    S = 0
    M = 0
    FP = 0
    nEvts = 0
    for key in d_result_iEvt.keys():
        nEvts+=1
        for iPV in range(len(d_result_iEvt[key]["PV_match"])):
            if d_result_iEvt[key]["PV_match"][iPV]==1:
                S += 1
            elif d_result_iEvt[key]["PV_match"][iPV]==0:
                M += 1
            elif d_result_iEvt[key]["PV_match"][iPV]==-1:
                FP += 1
            else:
                print(d_result_iEvt[key]["PV_match"][iPV])
    '''            
    print(S)
    print(M)
    print(FP)
    print("nPVs=",(S+M))
    print("")
    print("Eff    = %.4f"%(S/float(S+M)))
    print("FP_pv  = %.4f"%(FP/float(FP+S)))
    print("FP_evt = %.4f"%(FP/float(nEvts)))
    '''
    eff = ValueSet_GNN(S, M, FP, nEvts)
    print(eff)
    return eff

def evaluate_GNN_model(model, test_data, inputFile, configs, weights_file, outPutFolder, Evts_display=[0,1], makePlots=False, verbose=False):
    
    
    file          = h5py.File(inputFile, "r")
    
    pv_x          = Get_ak_from_hdf5(file, "pv_x")
    pv_y          = Get_ak_from_hdf5(file, "pv_y")
    pv_z          = Get_ak_from_hdf5(file, "pv_z")
    pv_n          = Get_ak_from_hdf5(file, "pv_ntracks")
    pv_key        = Get_ak_from_hdf5(file, "pv_key")
    tracks_pv_key = Get_ak_from_hdf5(file, "recon_pv_key")
    tracks_z      = Get_ak_from_hdf5(file, "poca_z")

    ## ----------------------------------------------------------------
    ## Get the model weights (if not using test_model after a training)
    if torch.cuda.is_available():
        weights = torch.load(weights_file)
    else:
        weights = torch.load(weights_file, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    
    ## ----------------------------------------------------------------
    ## Get the current device
    device = get_device_from_model(model)
    ## Swich the model to evaluation mode and the model to 'cpu'
    '''
    if 'cuda' in str(device):
        model.to('cpu')
    model.eval()
    '''
    debug = False
    
    ## ----------------------------------------------------------------------
    ## Get the global boudaries of the z axis:
    ## ----------------------------------------------------------------------
    global_zmin = configs["global_configs"][configs["Experiment"]]["z_min"]
    global_zmax = configs["global_configs"][configs["Experiment"]]["z_max"]
    
    
    validPV_minTracks = configs["GNN_clusterin_config"]["validPV_minTracks"]
    
    ## ----------------------------------------------------------------------
    ## Initialize some counters:
    ## ----------------------------------------------------------------------
    nEvts = 0
    S_tot = 0
    M_tot = 0
    FP_tot = 0

    ## ----------------------------------------------------------------------
    ## Setup the performance dictionnary -- for groups of tracks here
    ## ----------------------------------------------------------------------
    d_result_group = {
        "True_PV_nTcks":[],
        "True_PV_z_mm":[],
        "True_PV_res_mm":[],
        "True_PV_dz_mm":[],
        "PV_match":[],
        "Pred_PV_matched_z_mm":[],
        "Pred_PV_FP_z_mm":[],
        "Pred_PV_matched_res_mm":[],
        "Pred_PV_FP_res_mm":[],
               }

    ## ----------------------------------------------------------------------
    ## Setup the performance dictionnary -- for events here
    ## ----------------------------------------------------------------------
    d_result_iEvt = {}

    with torch.no_grad():
        iEvt=0
        d_result_iEvt[iEvt] = {
                                "True_PV_nTcks":[],
                                "True_PV_z_mm":[],
                                "True_PV_res_mm":[],
                                "True_PV_dz_mm":[],
                                "PV_match":[],
                                "Pred_PV_matched_z_mm":[],
                                "Pred_PV_FP_z_mm":[],
                                "Pred_PV_matched_res_mm":[],
                                "Pred_PV_FP_res_mm":[]        
        }    
        
        
        ## ----------------------------------------------------------------------
        ## Looping over the examples in the test loader (i.e. groups of tracks)
        ## ----------------------------------------------------------------------
        n_iter_data = len(test_data.dataset)

        #for iGroup in range(10):
        
        for iGroup in range(n_iter_data):

            #if iEvt > 5: continue

            #if iEvt in [360,1508,1512,1854, 1859, 1860, 1888,2414,2457,2893]:
            #    debug=True
            #else:
            #    debug=False
            
            debug=False
            
            if iGroup%1000==0:
                print("At iGroup %d out of %d"%(iGroup,n_iter_data))
            data = test_data.dataset[iGroup].cpu()
            group_evt = data.evt
            if iEvt!=group_evt:

                if debug:
                    print("iEvt:",iEvt)

                prev_idx = len(d_result_group["True_PV_nTcks"]) - len(d_result_iEvt[iEvt]["True_PV_nTcks"])
                if debug:
                    print("prev_idx",prev_idx)
                ## Search and remove duplicate true PVs:
                dup_idx = []
                for i in range(len(d_result_iEvt[iEvt]["True_PV_z_mm"])):
                    for j in range(i+1,len(d_result_iEvt[iEvt]["True_PV_z_mm"])):
                        if d_result_iEvt[iEvt]["True_PV_z_mm"][i]==d_result_iEvt[iEvt]["True_PV_z_mm"][j]:
                            dup_idx.append(j)

                for idx in dup_idx:

                    if debug:
                        print("removing d_result[True_PV_z_mm]       = ",d_result_group["True_PV_z_mm"][idx+prev_idx])
                        print("removing d_result[iEvt][True_PV_z_mm] = ",d_result_iEvt[iEvt]["True_PV_z_mm"][idx])
                    d_result_group["True_PV_nTcks"].pop(idx+prev_idx)
                    d_result_group["True_PV_z_mm"].pop(idx+prev_idx)
                    d_result_group["True_PV_res_mm"].pop(idx+prev_idx)
                    d_result_group["True_PV_dz_mm"].pop(idx+prev_idx)
                    d_result_group["PV_match"].pop(idx+prev_idx)

                    d_result_group["Pred_PV_matched_z_mm"].pop(idx+prev_idx)
                    d_result_group["Pred_PV_matched_res_mm"].pop(idx+prev_idx)
                    d_result_group["Pred_PV_FP_z_mm"].pop(idx+prev_idx)
                    d_result_group["Pred_PV_FP_res_mm"].pop(idx+prev_idx)          


                    d_result_iEvt[iEvt]["True_PV_nTcks"].pop(idx)
                    d_result_iEvt[iEvt]["True_PV_z_mm"].pop(idx)
                    d_result_iEvt[iEvt]["True_PV_res_mm"].pop(idx)
                    d_result_iEvt[iEvt]["True_PV_dz_mm"].pop(idx)
                    d_result_iEvt[iEvt]["PV_match"].pop(idx)

                    d_result_iEvt[iEvt]["Pred_PV_matched_z_mm"].pop(idx)
                    d_result_iEvt[iEvt]["Pred_PV_matched_res_mm"].pop(idx)
                    d_result_iEvt[iEvt]["Pred_PV_FP_z_mm"].pop(idx)
                    d_result_iEvt[iEvt]["Pred_PV_FP_res_mm"].pop(idx)



                all_truth_z = ak.to_numpy(pv_z[iEvt])
                all_truth_n = ak.to_numpy(pv_n[iEvt])


                valid_all_truth  = np.where((all_truth_n>validPV_minTracks) & (all_truth_z>global_zmin) & (all_truth_z<global_zmax))
                all_truth_z  = all_truth_z[valid_all_truth]
                all_truth_n  = all_truth_n[valid_all_truth]
                all_truth_sz = get_res(all_truth_n, configs["global_configs"][configs["Experiment"]])


                truth_z      = np.array(d_result_iEvt[iEvt]["True_PV_z_mm"])
                filt_truth_z = np.where(truth_z!=-9)
                truth_z      = truth_z[filt_truth_z]

                if not len(all_truth_z)==len(truth_z):
                    if debug:
                        print("iEvt",iEvt)
                        print("all_truth_z",all_truth_z)
                        print("truth_z",truth_z)
                    for i_all_truth in range(len(all_truth_z)):
                        i_all_truth_z = round(all_truth_z[i_all_truth]*100)/100.
                        found=False 
                        for i_truth_z in truth_z:
                            i_truth_z = round(i_truth_z*100)/100.
                            if i_all_truth_z==i_truth_z:
                                found=True
                                break
                        if not found:
                            if debug:
                                print("Added missed PV with:")
                                print("n=",all_truth_n[i_all_truth])
                                print("z=",all_truth_z[i_all_truth])
                                print("sz=",all_truth_sz[i_all_truth])
                            True_PV_dz_mm = -9
                            matched = 0

                            d_result_group["True_PV_nTcks"].append(all_truth_n[i_all_truth])
                            d_result_group["True_PV_z_mm"].append(all_truth_z[i_all_truth])
                            d_result_group["True_PV_res_mm"].append(all_truth_sz[i_all_truth])
                            d_result_group["True_PV_dz_mm"].append(True_PV_dz_mm)
                            d_result_group["PV_match"].append(matched)

                            d_result_group["Pred_PV_matched_z_mm"].append(-9)
                            d_result_group["Pred_PV_matched_res_mm"].append(-9)
                            d_result_group["Pred_PV_FP_z_mm"].append(-9)
                            d_result_group["Pred_PV_FP_res_mm"].append(-9)

                            d_result_iEvt[iEvt]["True_PV_nTcks"].append(all_truth_n[i_all_truth])
                            d_result_iEvt[iEvt]["True_PV_z_mm"].append(all_truth_z[i_all_truth])
                            d_result_iEvt[iEvt]["True_PV_res_mm"].append(all_truth_sz[i_all_truth])
                            d_result_iEvt[iEvt]["True_PV_dz_mm"].append(True_PV_dz_mm)
                            d_result_iEvt[iEvt]["PV_match"].append(matched)

                            d_result_iEvt[iEvt]["Pred_PV_matched_z_mm"].append(-9)
                            d_result_iEvt[iEvt]["Pred_PV_matched_res_mm"].append(-9)
                            d_result_iEvt[iEvt]["Pred_PV_FP_z_mm"].append(-9)
                            d_result_iEvt[iEvt]["Pred_PV_FP_res_mm"].append(-9)

                iEvt = group_evt
                nEvts += 1
                d_result_iEvt[iEvt] = {
                                        "True_PV_nTcks":[],
                                        "True_PV_z_mm":[],
                                        "True_PV_res_mm":[],
                                        "True_PV_dz_mm":[],
                                        "PV_match":[],
                                        "Pred_PV_matched_z_mm":[],
                                        "Pred_PV_FP_z_mm":[],
                                        "Pred_PV_matched_res_mm":[],
                                        "Pred_PV_FP_res_mm":[]
                }    


            data.cuda()
            #data.cpu()

            ## ------------------------------
            ## Get the z range values
            zmin   = data.z_min
            zmax   = data.z_max
            zrange = zmax-zmin

            ## ------------------------------
            ## Filter out tracks from PV with less than "validPV_minTracks"
            tracks_x, tracks_cat, y, pred, true_pvs_z, true_pvs_n = skim_data_with_low_mutl_true_pv(data, model, validPV_minTracks)


            ## ------------------------------
            nTracks  = len(tracks_x)

            ## ------------------------------
            ## Set the z and sz values back from [0,1] to [zmin,zmax]
            tracks_x.T[2] = (tracks_x.T[2]*zrange)+zmin
            tracks_x.T[5] =  tracks_x.T[5]*zrange                

            ## ------------------------------
            ## Set the predicted PV z values back from [0,1] to [zmin,zmax]
            pred.T[2] = (pred.T[2]*zrange)+zmin        

            ## ------------------------------
            ## Concatenate the tracks parameters with the predicted or true PV z positions
            X_pred = np.concatenate([tracks_x,pred],axis=1)
            X_true = np.concatenate([tracks_x,y],axis=1)

            ## ------------------------------
            ## Sort by predicted pv position
            X_pred = X_pred[X_pred[:,8].argsort()]
            X_true = X_true[X_true[:,8].argsort()]

            ## It happens (rarelly) that some of the true tracks position are way out of the considered range.
            ## Because it messes up with the visual representation of true clusters, just remove them...
            ## This has no impact whatsoever, since we use the truth level of PV position for comparaison 
            ## with the predicted ones.
            X_true = X_true[np.where( (X_true[:,8]>zmin) & (X_true[:,8]<zmax))]

            pred_pvs_info, n_tracks_true_vis = get_info_and_plot_cluster(X_pred, X_true, 
                                                                         iEvt, iGroup, 
                                                                         zmin, zmax,
                                                                         configs,
                                                                         doPlot  = makePlots,
                                                                         verbose = verbose)

            #print("pred_pvs_info",pred_pvs_info)
            
            pred_pvs_z  = np.array([])
            pred_pvs_sz = np.array([])
            pred_pvs_n  = np.array([])
            for pred_pv_info in pred_pvs_info:
                pred_pvs_z  = np.append(pred_pvs_z,  pred_pv_info[2])
                pred_pvs_sz = np.append(pred_pvs_sz, pred_pv_info[5])
                pred_pvs_n  = np.append(pred_pvs_n,  pred_pv_info[9])

            sorting = np.argsort(pred_pvs_z)
            pred_pvs_z  = pred_pvs_z[sorting]
            pred_pvs_sz = pred_pvs_sz[sorting]
            pred_pvs_n  = pred_pvs_n[sorting]

            valid_pred  = np.where((pred_pvs_z>global_zmin) & (pred_pvs_z<global_zmax))
            pred_pvs_z  = pred_pvs_z[valid_pred]
            pred_pvs_sz = pred_pvs_sz[valid_pred]
            pred_pvs_n  = pred_pvs_n[valid_pred]
            if configs["GNN_clusterin_config"]["use_TDR_res"]:
                pred_pvs_sz = get_res(pred_pvs_n, configs["global_configs"][configs["Experiment"]])

            valid_true  = np.where((true_pvs_z>global_zmin) & (true_pvs_z<global_zmax))
            true_pvs_z  = true_pvs_z[valid_true]
            true_pvs_n  = true_pvs_n[valid_true]
            true_pvs_sz = get_res(true_pvs_n, configs["global_configs"][configs["Experiment"]])


            l_matched_true_idx = []
            l_matched_pred_idx = []


            if debug:            
                print("*"*100)
                print("Looping over true PVs with z ",true_pvs_z)
            for i_true_PV in range(len(true_pvs_z)):

                if debug: 
                    print("-"*100)
                    print("At true_PV[%s] with z = %.3f"%(i_true_PV,true_pvs_z[i_true_PV]))

                ## Check if current true PV has already been matched
                if i_true_PV in l_matched_true_idx: 
                    if debug:
                        print("True PV already matched, skip to next true PV...")
                    continue

                ## Bool to check if current true PV is matched to a pred PV
                matched = 0

                # Now looping over the predicted PVs.
                for i_pred_PV in range(len(pred_pvs_z)):

                    if debug: 
                        print("")
                        print("At pred_PV[%s] with z = %.3f"%(i_pred_PV,pred_pvs_z[i_pred_PV]))

                    ## Check if current predicted PV has already been matched
                    if i_pred_PV in l_matched_pred_idx: 
                        if debug:
                            print("Pred PV already matched, skip to next pred PV...")                    
                        continue

                    min_val = pred_pvs_z[i_pred_PV]-configs["GNN_clusterin_config"]["nSigma_match_pv"]*pred_pvs_sz[i_pred_PV]
                    max_val = pred_pvs_z[i_pred_PV]+configs["GNN_clusterin_config"]["nSigma_match_pv"]*pred_pvs_sz[i_pred_PV]

                    if debug:
                        print("Looking for a true PV in range [%.4f ; %.4f]"%(min_val,max_val))

                    if min_val <= true_pvs_z[i_true_PV] and true_pvs_z[i_true_PV] <= max_val:
                        # --------------------------------------------------
                        # --------------------------------------------------
                        ## Current true PV is matched to a pred PV
                        ## This is a found PV!!
                        # --------------------------------------------------

                        if debug:
                            print("PV[%s] is matched >> SUCCESS"%i_true_PV)

                        l_matched_pred_idx.append(i_pred_PV)
                        l_matched_true_idx.append(i_true_PV)

                        True_PV_dz_mm = true_pvs_z[i_true_PV] - pred_pvs_z[i_pred_PV]
                        matched = 1

                        d_result_group["True_PV_nTcks"].append(true_pvs_n[i_true_PV])
                        d_result_group["True_PV_z_mm"].append(true_pvs_z[i_true_PV])
                        d_result_group["True_PV_res_mm"].append(true_pvs_sz[i_true_PV])
                        d_result_group["True_PV_dz_mm"].append(True_PV_dz_mm)
                        d_result_group["PV_match"].append(matched)

                        d_result_group["Pred_PV_matched_z_mm"].append(pred_pvs_z[i_pred_PV])
                        d_result_group["Pred_PV_matched_res_mm"].append(pred_pvs_sz[i_pred_PV])
                        d_result_group["Pred_PV_FP_z_mm"].append(-9)
                        d_result_group["Pred_PV_FP_res_mm"].append(-9)

                        d_result_iEvt[iEvt]["True_PV_nTcks"].append(true_pvs_n[i_true_PV])
                        d_result_iEvt[iEvt]["True_PV_z_mm"].append(true_pvs_z[i_true_PV])
                        d_result_iEvt[iEvt]["True_PV_res_mm"].append(true_pvs_sz[i_true_PV])
                        d_result_iEvt[iEvt]["True_PV_dz_mm"].append(True_PV_dz_mm)
                        d_result_iEvt[iEvt]["PV_match"].append(matched)

                        d_result_iEvt[iEvt]["Pred_PV_matched_z_mm"].append(pred_pvs_z[i_pred_PV])
                        d_result_iEvt[iEvt]["Pred_PV_matched_res_mm"].append(pred_pvs_sz[i_pred_PV])
                        d_result_iEvt[iEvt]["Pred_PV_FP_z_mm"].append(-9)
                        d_result_iEvt[iEvt]["Pred_PV_FP_res_mm"].append(-9)


                        # Since a predicted PV and a true PV are matched, go to the next true PV
                        break

                if matched==0:
                    # --------------------------------------------------
                    ## Current true PV could not be matched to any pred PV
                    ## This is a missed PV!!
                    # --------------------------------------------------
                    True_PV_dz_mm = -9

                    if debug:
                        print("PV[%s] is not matched >> MISSED"%i_true_PV)

                    d_result_group["True_PV_nTcks"].append(true_pvs_n[i_true_PV])
                    d_result_group["True_PV_z_mm"].append(true_pvs_z[i_true_PV])
                    d_result_group["True_PV_res_mm"].append(true_pvs_sz[i_true_PV])
                    d_result_group["True_PV_dz_mm"].append(True_PV_dz_mm)
                    d_result_group["PV_match"].append(matched)

                    d_result_group["Pred_PV_matched_z_mm"].append(-9)
                    d_result_group["Pred_PV_matched_res_mm"].append(-9)
                    d_result_group["Pred_PV_FP_z_mm"].append(-9)
                    d_result_group["Pred_PV_FP_res_mm"].append(-9)

                    d_result_iEvt[iEvt]["True_PV_nTcks"].append(true_pvs_n[i_true_PV])
                    d_result_iEvt[iEvt]["True_PV_z_mm"].append(true_pvs_z[i_true_PV])
                    d_result_iEvt[iEvt]["True_PV_res_mm"].append(true_pvs_sz[i_true_PV])
                    d_result_iEvt[iEvt]["True_PV_dz_mm"].append(True_PV_dz_mm)
                    d_result_iEvt[iEvt]["PV_match"].append(matched)

                    d_result_iEvt[iEvt]["Pred_PV_matched_z_mm"].append(-9)
                    d_result_iEvt[iEvt]["Pred_PV_matched_res_mm"].append(-9)
                    d_result_iEvt[iEvt]["Pred_PV_FP_z_mm"].append(-9)
                    d_result_iEvt[iEvt]["Pred_PV_FP_res_mm"].append(-9)


            for i_pred_PV in range(len(pred_pvs_z)):

                # Check if pred PV was matched, and only consider unmatched predicted == FP
                if i_pred_PV in l_matched_pred_idx: continue

                if debug:
                    print("Now filling info for unmatched pred PV[%s] >> FALSE POSITIVE"%i_pred_PV)

                matched = -1
                d_result_group["True_PV_nTcks"].append(-9)
                d_result_group["True_PV_z_mm"].append(-9)
                d_result_group["True_PV_res_mm"].append(-9)
                d_result_group["True_PV_dz_mm"].append(-9)
                d_result_group["PV_match"].append(matched)

                d_result_group["Pred_PV_matched_z_mm"].append(-9)
                d_result_group["Pred_PV_matched_res_mm"].append(-9)
                d_result_group["Pred_PV_FP_z_mm"].append(pred_pvs_z[i_pred_PV])
                d_result_group["Pred_PV_FP_res_mm"].append(pred_pvs_sz[i_pred_PV])

                d_result_iEvt[iEvt]["True_PV_nTcks"].append(-9)
                d_result_iEvt[iEvt]["True_PV_z_mm"].append(-9)
                d_result_iEvt[iEvt]["True_PV_res_mm"].append(-9)
                d_result_iEvt[iEvt]["True_PV_dz_mm"].append(-9)
                d_result_iEvt[iEvt]["PV_match"].append(matched)

                d_result_iEvt[iEvt]["Pred_PV_matched_z_mm"].append(-9)
                d_result_iEvt[iEvt]["Pred_PV_matched_res_mm"].append(-9)
                d_result_iEvt[iEvt]["Pred_PV_FP_z_mm"].append(pred_pvs_z[i_pred_PV])
                d_result_iEvt[iEvt]["Pred_PV_FP_res_mm"].append(pred_pvs_sz[i_pred_PV])


    dict_tag_group = str(outPutFolder / f'GNN_perf_group{configs["model_class"]}.npy')
    dict_tag_event = str(outPutFolder / f'GNN_perf_event{configs["model_class"]}.npy')

    np.save(dict_tag_group, d_result_group)  
    np.save(dict_tag_event, d_result_iEvt)  
    
    return d_result_group, d_result_iEvt

## ============================================================
def test_model(model, test_data, configs, weights_file, outPutFolder_plots, Evts_display=[0,1], display=True):
        

    ## =================================================================
    ## We want to look into events and within each event the intervals
    ## Since the data is structured as an array of intervals, we 
    ## actually need to reconstruct the event by adding up each 
    ## corresponding interval.
    ## =================================================================

    ## -----------------------------------------------------------------
    ## Get the considered experiement: LHCb / ATLAS / CMS...
    Exp = configs['Experiment']
    
    ## -----------------------------------------------------------------
    ## Get the global configuration 
    global_configs = configs['global_configs'][Exp]
    
    ## -----------------------------------------------------------------
    ## Let start by getting back the number of intervals per event    
    nBinsKDE_perEvt     = global_configs["n_bins_poca_kde"]
    nBinsPerInterval    = global_configs["nBinsPerInterval"]
    nIntervals_perEvt   = int(nBinsKDE_perEvt/nBinsPerInterval)    
    
    ## =================================================================
    ## First get the model outputs (from inputs) and the labels from the test data
    ## =================================================================

    ## ----------------------------------------------------------------
    ## Get the model weights (if not using test_model after a training)
    if torch.cuda.is_available():
        weights = torch.load(weights_file)
    else:
        weights = torch.load(weights_file, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    
    ## ----------------------------------------------------------------
    ## Get the current device
    device = get_device_from_model(model)
    ## Swich the model to evaluation mode and the model to 'cpu'
    if 'cuda' in str(device):
        model.to('cpu')
    model.eval()

    ## =================================================================
    ## EFFICIENCY EVALUATION
    ## =================================================================
    if configs["model_class"]=="tracks-to-hist" or configs["model_class"]=="KDE-to-hist":

        ## ----------------------------------------------------------------
        ## Get the efficiency parameters dictionnary from the config file
        eff_params = configs["efficiency_config"]
        
        ## --------------------------------------------------------
        ## Reset the efficiency values 
        eff = ValueSet(0, 0, 0, 0)
        
        l_events_missed = []
        l_events_FP = []
        
        ## ----------------------------------------------------------------
        ## Splitting the inputs and outputs is usefull regarding to memory allocation...
        with torch.no_grad():
            ## Get the full test data for efficiency calculation
            model_preds_eff = model(test_data.dataset.tensors[0]).cpu().numpy()
            labels_eff      = test_data.dataset.tensors[1].cpu().numpy()

            ## Total number of available events in the test data
            totalEvts_test_data = int(len(labels_eff)/nIntervals_perEvt)

            ## If doing intervals, reshape the data to be shaped as events
            # Start by computing the number of intervals per event
            if configs["training_configs"]["doIntervals"]:
                model_preds_eff = model_preds_eff.reshape((totalEvts_test_data,nBinsKDE_perEvt))
                labels_eff      = labels_eff.reshape((totalEvts_test_data,nBinsKDE_perEvt))
            
            sumFound = 0
            sumMissed = 0
            sumFalsePositive = 0
            
            for iEvt in range(totalEvts_test_data):  
                iEvt_eff = eval_efficiency(labels_eff[iEvt], model_preds_eff[iEvt], **eff_params)
                Found = iEvt_eff.S
                Missed = iEvt_eff.MT
                FalsePositive = iEvt_eff.FP
                if Missed>0:
                    l_events_missed.append(iEvt)
                if FalsePositive>0:
                    l_events_FP.append(iEvt)
                sumFound += Found
                sumMissed += Missed
                sumFalsePositive += FalsePositive
                
                print("Event %d  --> Found, Missed, FalsePositive =  "%(iEvt),Found, Missed, FalsePositive,
                      " ;   sums = ", sumFound,sumMissed,sumFalsePositive)
                      
                eff += iEvt_eff
               
            print("")
            print("Events with missed PVs:\n",l_events_missed)
            print("")
            print("Events with FP signal: \n",l_events_FP)
            print("")
            ## Printing out efficiency results and parameters used for the evaluation
            print("*"*100)
            print(" Model efficiency on test data:\n",eff.pretty())
            print("")
            print("-"*100)
            print("")
            print(" Obtained with efficiency parameters:")
            pprint.pprint(eff_params, indent=10)
            print("*"*100)
            
    ## =================================================================
    ## PLOTTING PREDICTIONS AND LABELS
    ## =================================================================
    ## ----------------------------------------------------------------
    ## Splitting the inputs and outputs is usefull regarding to memory allocation...
    with torch.no_grad():
        inputs = test_data.dataset.tensors[0]
        labels = test_data.dataset.tensors[1]
                
        nSplit = []
        for ii in range(20):
            nSplit.append((ii+1)*100000)

        inputs_Split = torch.tensor_split(inputs, nSplit, dim=0)    
        labels_Split = torch.tensor_split(labels, nSplit, dim=0)
        
        defaultSplitSize = inputs_Split[0].shape[0]


    with torch.no_grad():
        ## Loop over the tensors stored in the split tuple of tensors
        for iChunk in range(1):
            
            labels      = labels_Split[iChunk].cpu().numpy()
            model_preds = model(inputs_Split[iChunk]).cpu().numpy()
    
    # model_pred = np.asarray(model_preds[0])
    
            
        
    ## Set the bin values for each interval (by definition from 0 to nBinsPerInterval)
    interval_bins      = np.arange(nBinsPerInterval)
    ## Set the bin values for each event (by definition from 0 to nBinsTotal)
    event_bins         = np.arange(nBinsKDE_perEvt)
    ## Get the z values for the current interval
    z_vals_evt         = from_bins_to_z(event_bins, global_configs)
    ## -----------------------------------------------------------------
    ## Now we loop over an arbitrary number of events defined as an 
    ## argument of test_model
    #listOfEvents = np.arange(nEvts_display)
    for iEvt in Evts_display:
        
        if display:
            print("*"*100)
            print("")
            print(" Event (%s) "%iEvt)
            print("")

        ## ----------------------------------------------
        ## Define empty label and model prediction
        ## that will be filled within the loop over
        ## the intervals (see below)
        event_label      = np.asarray([])
        event_model_pred = np.asarray([])
        
        ## ----------------------------------------------
        ## Let's now loop over the global intervals ID  
        ## and retreive the labels and model predictions
        ## Set the interval ID within each event to zero
        interval_thisEvt = 0
        maxs_interval = []
        for interval_global in range(iEvt*nIntervals_perEvt,iEvt*nIntervals_perEvt+nIntervals_perEvt):
            label      = np.asarray(labels[interval_global])
            model_pred = np.asarray(model_preds[interval_global])
            
            ## Get the z values for the current interval
            z_vals = from_bins_to_z(interval_bins, global_configs, True, interval_thisEvt, nBinsPerInterval)
            
            ## ----------------------------------------------
            ## We concatenate all intervals within one event (iEvt) 
            event_label           = np.concatenate((event_label,label))
            event_model_pred      = np.concatenate((event_model_pred,model_pred))
            
            ## ----------------------------------------------
            ## ymax is meant to define the maximum of the plot
            ## where both the model prediction and label
            ## will be superimposed
            ymax = max(np.max(label),np.max(model_pred))
            ## ----------------------------------------------
            ## Only make the interval plot if ymax is above a certain threshold
            ## as most of the intervals are empty: i.e. we only want ot check the ones 
            ## in where there is some activity
            if (ymax>0.1):
                if display:
                    print("="*50)
                    print(" Event (%s) -- Interval (%s) -- ymax (%5.2f) " %(iEvt,interval_thisEvt,ymax))
                ## For protting purposes let's redefine ymax
                ymax = max(ymax,1.0)
                maxs_interval.append(ymax)
                plot_tag = str(outPutFolder_plots / f'{configs["model_class"]}_Evt{iEvt}_Inter{interval_thisEvt}')
                plot_z_hist(ymax, z_vals, label, model_pred, configs, plot_tag, display)
                
            ## Iterate the interval id within each event    
            interval_thisEvt += 1
            
        ## ----------------------------------------------
        ## Finally we plot the event histogram
        ymax_evt = 1.1*max(maxs_interval)
        print("ymax_evt",ymax_evt)
        plot_tag = str(outPutFolder_plots / f'{configs["model_class"]}_Evt{iEvt}__AllInter')
        plot_z_hist(ymax_evt, z_vals_evt, event_label, event_model_pred, configs, plot_tag, display)
        if display:
            print("")
            print("*"*100)
            print("")

    ## ================================================================================
    # Save training plots
    print("")
    print("*"*100)
    print("Saved testing plots in:")
    print("")
    print(outPutFolder_plots)
    print("")
    print("*"*100)
    
    ## ----------------------------------------------------------------
    ## Swich the model back to 'cuda'
    if 'cuda' in str(device):
        model.to('cuda')
        