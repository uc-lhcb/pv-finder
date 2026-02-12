'''
Methods to split the data (KDE and target histograms) as well as grouping tracks in intervals along the z axis. This is required for the tracks-to-KDE and tracks-to-hist models.

NOTES:

1_ Equivalent of padding is performed here when building the tracks containers in each interval. 
   In indor for each batch to have identical shape when building the pytorch tensor data loader, 
   we need a fixe size tensor for the input data of shape [N_events, N_features, N_tracks]
   The number of tracks per event is therefore fix to maxTracksPerInterval (defined in config file).
   We first create empty containers for tracks features of shape
   
   [N_events*N_intervals,maxTracksPerInterval]
   
   with values = tracks_default_POCA (== -99.0) for all tracks parameters. 
   The choice of -99.0 is arbitrary and far away from any true tracks parameter values.
   Can be changed in config file.
   
   In the case one interval has N_tracks < maxTracksPerInterval, then the remaining entries
   will have values of -99.0. When constructing the NN model, a mask is applied to filter out the 
   entries with -99.0.
   
2_ When defining a given number of bins per interval, the could not be a divider of the total 
   number of bins of the KDE, leading to a non integer number of bins. In this case the remaining 
   bins at the last bins in z (bins with larger z values) are "scrubbed". See top part of the script.
      
3_ For a given interval tracks with z values overlapping with the lower and upper interval (in z) are also used to
   form the associated tracks for this interval. A +/- 2.5 mm window is considered (defined in config file) here.
   
4_ From all tracks associated to a given interval, not all tracks are persisted. 
   A selection based on the POCA_A, POCA_B and POCA_C values, corresponding to the diagonnal of the error 
   matrix (i.e. sigma_x, sigma_y and sigma_z, respectively), is applied.

5_ In order to have identical interval definitions, all tracks finally associated to a given interval are shifted
   in Z such that in each interval, the POCA_z ranges are identical. This is needed for the NN training as it allows 
   the NN not to care about the absolute positions of the tracks in Z. This operation assumes the PVs and tracks 
   parameters are invariant along Z.
   
'''

import awkward as ak
import numpy as np
from utils.utilities import Timer
from tools.named_tuples_def import InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple_intervals

## The method below is the same as code from at 'for i_Interval in range(nIntervals):', but using jit
## To be used in case the the classic "for loop" is too slow... Does not seem to be the case here, so 
## we do not use it for the moment.
from tools.get_tracks_info_intervals import get_tracks_info_intervals   
    
def split_data_intervals(input_tuple, config):

    
    ## ************************************************************    
    ## Let start by defining the intervals definition: 
    nBinsTotal         = config["n_bins_poca_kde"]
    nBinsPerInterval   = config["nBinsPerInterval"]
    nIntervals         = int(nBinsTotal/nBinsPerInterval)
    nBinsAllIntervals  = nBinsPerInterval*nIntervals
    ## In case the number of intervals is not a multiple of nBinsTotal
    ## we need to scrub the end of the KDE and target hist
    if (nBinsTotal != nBinsAllIntervals):
        print("-"*100)
        print("    ----------------------------------- ")
        print("    >>>>>>>>>>>>  WARNING  <<<<<<<<<<<< ")
        print("    ----------------------------------- ")
        print("")
        print("    Requested number of bins per interval:",nBinsPerInterval)
        print("    is not a multiple of the full histograms total number of bins:",nBinsTotal)
        print("    `---> this results in a number of intervals (%.2f) that is not an integer"%(nBins/binsPerInterval))
        print("")
        print("    Only the first %s bins (from low to high Z values) of the full KDE and target histograms will be used,"%nBinsAllIntervals)
        print("    corresponding to %s intervals of %s bins"%(nIntervals,nBinsPerInterval))
        print("")
        print("    Tracks containers will be split in %s intervals with +/- %.2f mm extensions on each interval edge"%(nIntervals,config["intervalExtension"]))
        print("    `---> This implies one track can be present in more than one inetrval!!!!")
        print("")
        print("    In each interval, tracks are selected according to the following criteria:")
        print("         ABS(POCA_x) / sigma(x) < %.1f     with sigma(POCA_x) = 1/sqrt(ABS(poca_A)) "%(config["maxSigmaX"]))
        print("         ABS(POCA_y) / sigma(y) < %.1f     with sigma(POCA_y) = 1/sqrt(ABS(poca_B)) "%(config["maxSigmaY"]))
        print("         sigma(z) < %.1f                   with sigma(POCA_z) = 1/sqrt(ABS(poca_C)) "%(config["maxSigmaZ"]))
        
        print("")
        print("    Maximum number of tracks to be persisted set to",config["maxTracksPerInterval"])
        print("    In case nSelTracks larger than maximum, only the %s tracks with smallest poca_C are persisted!"%(config["maxTracksPerInterval"]))
        print("")
        print("-"*100)
    else:
        print("-"*100)
        print("")
        print("    Each KDE and target histogram will be split in %s intervals of %s bins"%(nIntervals,nBinsPerInterval))
        print("")
        print("    Tracks containers will be split in %s intervals with +/- %.2f mm extensions on each interval edge"%(nIntervals,config["intervalExtension"]))
        print("    `---> This implies one track can be present in more than one inetrval!!!!")
        print("")
        print("    In each interval, tracks are selected according to the following criteria:")
        print("         ABS(POCA_x) / sigma(x) < %.1f     with sigma(POCA_x) = 1/sqrt(ABS(poca_A)) "%(config["maxSigmaX"]))
        print("         ABS(POCA_y) / sigma(y) < %.1f     with sigma(POCA_y) = 1/sqrt(ABS(poca_B)) "%(config["maxSigmaY"]))
        print("         sigma(z) < %.1f                   with sigma(POCA_z) = 1/sqrt(ABS(poca_C)) "%(config["maxSigmaZ"]))
        
        print("")
        print("    Maximum number of tracks to be persisted set to",config["maxTracksPerInterval"])
        print("    In case nSelTracks larger than maximum, only the %s tracks with smallest poca_C are persisted!"%(config["maxTracksPerInterval"]))
        print("")
        print("-"*100)
        
    ## ************************************************************    
    ## Get the POCA KDE histograms from the input tuple
    poca_KDE_A        = getattr(input_tuple, "poca_KDE_A")
    poca_KDE_A_xMax   = getattr(input_tuple, "poca_KDE_A_xMax")
    poca_KDE_A_yMax   = getattr(input_tuple, "poca_KDE_A_yMax")
    poca_KDE_B        = getattr(input_tuple, "poca_KDE_B")
    poca_KDE_B_xMax   = getattr(input_tuple, "poca_KDE_B_xMax")
    poca_KDE_B_yMax   = getattr(input_tuple, "poca_KDE_B_yMax")    
    ## Get the target histogram from the input tuple
    targetHists       = getattr(input_tuple, "targetHists")

    
    ## ************************************************************    
    ## Now let take the corresponding proportion of all the input full hists
    poca_KDE_A_scrub      = poca_KDE_A[:,:nBinsAllIntervals]
    poca_KDE_A_xMax_scrub = poca_KDE_A_xMax[:,:nBinsAllIntervals]
    poca_KDE_A_yMax_scrub = poca_KDE_A_yMax[:,:nBinsAllIntervals]
    poca_KDE_B_scrub      = poca_KDE_B[:,:nBinsAllIntervals]
    poca_KDE_B_xMax_scrub = poca_KDE_B_xMax[:,:nBinsAllIntervals]
    poca_KDE_B_yMax_scrub = poca_KDE_B_yMax[:,:nBinsAllIntervals]    
    targetHists_scrub     = targetHists[:,:,:nBinsAllIntervals]
    
    ## ************************************************************    
    ## And reshape them to have the correct shape:
    ## i.e. (nEvts*nIntervals,nBinsPerInterval)
    nEvts = len(poca_KDE_A)
    
    poca_KDE_A_intervals      = np.reshape(poca_KDE_A_scrub,(nEvts*nIntervals,nBinsPerInterval))
    poca_KDE_A_xMax_intervals = np.reshape(poca_KDE_A_xMax_scrub,(nEvts*nIntervals,nBinsPerInterval))
    poca_KDE_A_yMax_intervals = np.reshape(poca_KDE_A_yMax_scrub,(nEvts*nIntervals,nBinsPerInterval))
    poca_KDE_B_intervals      = np.reshape(poca_KDE_B_scrub,(nEvts*nIntervals,nBinsPerInterval))
    poca_KDE_B_xMax_intervals = np.reshape(poca_KDE_B_xMax_scrub,(nEvts*nIntervals,nBinsPerInterval))
    poca_KDE_B_yMax_intervals = np.reshape(poca_KDE_B_yMax_scrub,(nEvts*nIntervals,nBinsPerInterval))
    targetHists_intervals     = np.reshape(targetHists_scrub,(len(targetHists_scrub),nEvts*nIntervals,nBinsPerInterval))
    
    
    ## ************************************************************    
    ## Now lets deal with the tracks that also need to be split into intervals...
    ## Let start by retrieving the maximum number of tracks per interval
    ##   >>>>> Get the max number of tracks from config file
    maxTracksPerInterval = config["maxTracksPerInterval"]
    ## Also redefine the maximum value of Z given of the choosen number of intervals
    z_min   = config["z_min"]
    z_max   = config["z_max"]
    z_range = (z_max-z_min)
    z_max   = z_min + z_range*(nBinsTotal/nBinsAllIntervals)
    ## And redefine the Z range accordingly
    z_range = (z_max-z_min)
    ## define the Z range for each interval accordingly
    interval_range = z_range/nIntervals
    
    ## ************************************************************    
    ## Get the tracks information from the input tuple
    poca_x        = getattr(input_tuple, "poca_x")
    poca_y        = getattr(input_tuple, "poca_y")
    poca_z        = getattr(input_tuple, "poca_z")
    poca_A        = getattr(input_tuple, "poca_A")
    poca_B        = getattr(input_tuple, "poca_B")
    poca_C        = getattr(input_tuple, "poca_C")
    poca_D        = getattr(input_tuple, "poca_D")
    poca_E        = getattr(input_tuple, "poca_E")
    poca_F        = getattr(input_tuple, "poca_F")
            
    ## ************************************************************    
    ## Define the empty containers for the tracks to be saved limiting 
    ## the total number of tracks to maxTracksPerInterval
    poca_x_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_y_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_z_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_A_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_B_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_C_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_D_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_E_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
    poca_F_intervals   = np.zeros((nEvts*nIntervals,maxTracksPerInterval))+config["tracks_default_POCA"]
      
    ## ************************************************************    
    ## Now we need to select the tracks to be included in each interval...
    ##
    ## To do so, a few conditions are requested:
    ##
    ## - First, we start by selecting all tracks in the 
    ##   considered interval +/- a delta of each lower and higher Z intervals
    ##   !!! This implies that some tracks can be added to more than one interval !!!!
    ##
    ##   >>>>> Get the delta extension values from the config file:
    intervalExtension = config["intervalExtension"]
    ##    
    ## - Second, from the list of these tracks, only a fraction is persisted
    ##   based on the POCA uncertainties (diagonal elements only)
    ##
    ##   @ sigma(x) < maxSigmaX     with sigma(POCA_x) = 1/sqrt(poca_A)
    ##   @ sigma(y) < maxSigmaY     with sigma(POCA_y) = 1/sqrt(poca_B)
    ##   @ sigma(z) < maxSigmaZ     with sigma(POCA_z) = 1/sqrt(poca_C)
    ##
    ##   >>>>> Get the max sigma parameters' values from the config file
    maxSigmaX = config["maxSigmaX"]
    maxSigmaY = config["maxSigmaY"]
    maxSigmaZ = config["maxSigmaZ"]
    ##
    ## We need to loop around events and for each event we loop around the 
    ## number of intervals. This is relatively slow and can probably be improved
    ## Loop over events...

    l_nSelTracks = []
    print("Looping over %s events..."%nEvts)
    for i_Evt in range(nEvts):
        
        if i_Evt%1000==0:
            print("At event:",i_Evt)
            
        ## Get the tracks of event i_Evt, and transform the ak arrays into numpy arrays,
        ## as they as simpe arrays anyway
        poca_x_evt = ak.to_numpy(poca_x[i_Evt])
        poca_y_evt = ak.to_numpy(poca_y[i_Evt])
        poca_z_evt = ak.to_numpy(poca_z[i_Evt])
        poca_A_evt = ak.to_numpy(poca_A[i_Evt])
        poca_B_evt = ak.to_numpy(poca_B[i_Evt])
        poca_C_evt = ak.to_numpy(poca_C[i_Evt])
        poca_D_evt = ak.to_numpy(poca_D[i_Evt])
        poca_E_evt = ak.to_numpy(poca_E[i_Evt])
        poca_F_evt = ak.to_numpy(poca_F[i_Evt])

        ## For each event loop over intervals...
        for i_Interval in range(nIntervals):
            
            ## Define the index position for the tracks container
            indexPosition = i_Evt*nIntervals + i_Interval

            ## Get considered "true" interval range in z 
            interval_lowEdge  = z_min + i_Interval*interval_range
            interval_highEdge = interval_lowEdge + interval_range 
            ## Get considered "extended" interval range in z, the one from which we will 
            ## keep tracks to then be selected base on their uncertainties
            interval_z_min    = interval_lowEdge  - intervalExtension
            interval_z_max    = interval_highEdge + intervalExtension
            
            '''            
            ### THIS IS A TEST TO SEE IF USING JIT WOULD SPEED UP THE PROCESS...
            ### APPARENTLY THE "CLASSIC" APPROACH IS ALREADY FAST EANOUGH....
            
            poca_x_ints, poca_y_ints, poca_z_ints, poca_A_ints, poca_B_ints, poca_C_ints, poca_D_ints, poca_E_ints,poca_F_ints =        get_tracks_info_intervals(poca_x_evt, poca_y_evt, poca_z_evt, poca_A_evt, poca_B_evt, poca_C_evt, poca_D_evt, poca_E_evt, poca_F_evt, interval_lowEdge, interval_z_min, interval_z_max, maxTracksPerInterval, maxSigmaX, maxSigmaY, maxSigmaZ)
            
            ## Get the updated length of tracks containers in the case it is smaller than maxTracksPerInterval 
            fillingLength = len(poca_x_ints)
            #print("indexPosition",indexPosition)
            #print("fillingLength",fillingLength)
            #print("len(poca_z_ints)",len(poca_z_ints))
            ## Add the tracks information to the overall containers based on the index position
            poca_x_intervals[indexPosition,:fillingLength] = poca_x_ints
            poca_y_intervals[indexPosition,:fillingLength] = poca_y_ints
            poca_z_intervals[indexPosition,:fillingLength] = poca_z_ints
            poca_A_intervals[indexPosition,:fillingLength] = poca_A_ints
            poca_B_intervals[indexPosition,:fillingLength] = poca_B_ints
            poca_C_intervals[indexPosition,:fillingLength] = poca_C_ints
            poca_D_intervals[indexPosition,:fillingLength] = poca_D_ints
            poca_E_intervals[indexPosition,:fillingLength] = poca_E_ints
            poca_F_intervals[indexPosition,:fillingLength] = poca_F_ints

            '''

            ## Define filter for the tracks in the "extended" interval range 
            intervalRange = (poca_z_evt>interval_z_min) & (poca_z_evt<interval_z_max)

            ## Now apply the filter to the tracks variables
            poca_x_inRange = poca_x_evt[intervalRange]
            poca_y_inRange = poca_y_evt[intervalRange]
            poca_z_inRange = poca_z_evt[intervalRange]
            poca_A_inRange = poca_A_evt[intervalRange]
            poca_B_inRange = poca_B_evt[intervalRange]
            poca_C_inRange = poca_C_evt[intervalRange]
            poca_D_inRange = poca_D_evt[intervalRange]
            poca_E_inRange = poca_E_evt[intervalRange]
            poca_F_inRange = poca_F_evt[intervalRange]

            '''
            print("   Interval(%s) with %s tracks"%(i_Interval,len(poca_z_inRange)))
            '''

            ## Apply a shift to the poca_z values in each interval so that 
            ## all tracks z values are "invariant" across all intervals
            poca_z_inRange = poca_z_inRange - interval_lowEdge
        
            ## Now compute the selection to keep only tracks that have "low"
            ## uncertainties, i.e. good quality tracks
            sigmaX = np.sqrt(np.absolute(np.divide(1.,poca_A_inRange)))
            sigmaY = np.sqrt(np.absolute(np.divide(1.,poca_B_inRange)))
            sigmaZ = np.sqrt(np.absolute(np.divide(1.,poca_C_inRange)))

            sigmaXrel = np.divide(poca_x_inRange,sigmaX)
            sigmaYrel = np.divide(poca_y_inRange,sigmaY)

            ## Define a filter for the selection of good tracks
            goodTracks = (sigmaZ<maxSigmaZ) & (np.absolute(sigmaXrel)<maxSigmaX) & (np.absolute(sigmaYrel)<maxSigmaY)
            
            ## Apply the filter of good tracks
            poca_x_inRange_sel = poca_x_inRange[goodTracks]
            poca_y_inRange_sel = poca_y_inRange[goodTracks]
            poca_z_inRange_sel = poca_z_inRange[goodTracks]
            poca_A_inRange_sel = poca_A_inRange[goodTracks]
            poca_B_inRange_sel = poca_B_inRange[goodTracks]
            poca_C_inRange_sel = poca_C_inRange[goodTracks]
            poca_D_inRange_sel = poca_D_inRange[goodTracks]
            poca_E_inRange_sel = poca_E_inRange[goodTracks]
            poca_F_inRange_sel = poca_F_inRange[goodTracks]
            '''
            print("   Interval(%s) with %s selected tracks"%(i_Interval,len(poca_z_inRange_sel)))
            print("")
            '''
            ## Check if the number of selected tracks is larger or smaller 
            ## than the maximum number of tracks per interval defined in the 
            ## config file
            ## Get the number of selected tracks
            nSelTracks = len(poca_z_inRange_sel)
            ## Define the number of tracks to be persisted
            fillingLength = min(nSelTracks,maxTracksPerInterval)

            l_nSelTracks.append(nSelTracks)

            if nSelTracks > maxTracksPerInterval:
                ## Since the number of tracks remaining in the extended interval is larger than the
                ## defined maximum number of tracks to be kept, we need to remove tracks...
                ## This should be done according to the sigmaZ values. 
                print("At event %s, interval %s (Z in [%.2f, %.2f]):"%(i_Evt,i_Interval,interval_z_min,interval_z_max))
                print("`---> the number of selected tracks is %s; larger than the max number of tracks %s"%(nSelTracks,maxTracksPerInterval))
                print("`---> %s tracks will be removed according to their sigmaZ values (largest are discarded)"%(nSelTracks-maxTracksPerInterval))
                ## Let start by selecting the good tracks from the sigmaZ container
                sigmaZ_sel = sigmaZ[goodTracks]
                ## Then get the sorting indexes 
                sortIdx_sigmaZ = np.argsort(sigmaZ_sel)
                ## and apply the sorting to all tracks containers
                poca_x_inRange_sel_ord = poca_x_inRange_sel[sortIdx_sigmaZ]
                poca_y_inRange_sel_ord = poca_y_inRange_sel[sortIdx_sigmaZ]
                poca_z_inRange_sel_ord = poca_z_inRange_sel[sortIdx_sigmaZ]
                poca_A_inRange_sel_ord = poca_A_inRange_sel[sortIdx_sigmaZ]
                poca_B_inRange_sel_ord = poca_B_inRange_sel[sortIdx_sigmaZ]
                poca_C_inRange_sel_ord = poca_C_inRange_sel[sortIdx_sigmaZ]
                poca_D_inRange_sel_ord = poca_D_inRange_sel[sortIdx_sigmaZ]
                poca_E_inRange_sel_ord = poca_E_inRange_sel[sortIdx_sigmaZ]
                poca_F_inRange_sel_ord = poca_F_inRange_sel[sortIdx_sigmaZ]
                
                ## Keep tracks up to maxTracksPerInterval ranging from smallest values of sigmaZ 
                poca_x_inRange_sel_ord_scrub = poca_x_inRange_sel_ord[:fillingLength]
                poca_y_inRange_sel_ord_scrub = poca_y_inRange_sel_ord[:fillingLength]
                poca_z_inRange_sel_ord_scrub = poca_z_inRange_sel_ord[:fillingLength]
                poca_A_inRange_sel_ord_scrub = poca_A_inRange_sel_ord[:fillingLength]
                poca_B_inRange_sel_ord_scrub = poca_B_inRange_sel_ord[:fillingLength]
                poca_C_inRange_sel_ord_scrub = poca_C_inRange_sel_ord[:fillingLength]
                poca_D_inRange_sel_ord_scrub = poca_D_inRange_sel_ord[:fillingLength]
                poca_E_inRange_sel_ord_scrub = poca_E_inRange_sel_ord[:fillingLength]
                poca_F_inRange_sel_ord_scrub = poca_F_inRange_sel_ord[:fillingLength]

                ## Re-order tracks based on the z position
                sortIdx_pocaZ = np.argsort(poca_z_inRange_sel_ord_scrub)
                
                poca_x_inRange_sel_ord_scrub_reOrd = poca_x_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_y_inRange_sel_ord_scrub_reOrd = poca_y_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_z_inRange_sel_ord_scrub_reOrd = poca_z_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_A_inRange_sel_ord_scrub_reOrd = poca_A_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_B_inRange_sel_ord_scrub_reOrd = poca_B_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_C_inRange_sel_ord_scrub_reOrd = poca_C_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_D_inRange_sel_ord_scrub_reOrd = poca_D_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_E_inRange_sel_ord_scrub_reOrd = poca_E_inRange_sel_ord_scrub[sortIdx_pocaZ]
                poca_F_inRange_sel_ord_scrub_reOrd = poca_F_inRange_sel_ord_scrub[sortIdx_pocaZ]

                ## Add the tracks information to the overall containers based on the index position
                poca_x_intervals[indexPosition,:fillingLength] = poca_x_inRange_sel_ord_scrub_reOrd
                poca_y_intervals[indexPosition,:fillingLength] = poca_y_inRange_sel_ord_scrub_reOrd
                poca_z_intervals[indexPosition,:fillingLength] = poca_z_inRange_sel_ord_scrub_reOrd
                poca_A_intervals[indexPosition,:fillingLength] = poca_A_inRange_sel_ord_scrub_reOrd
                poca_B_intervals[indexPosition,:fillingLength] = poca_B_inRange_sel_ord_scrub_reOrd
                poca_C_intervals[indexPosition,:fillingLength] = poca_C_inRange_sel_ord_scrub_reOrd
                poca_D_intervals[indexPosition,:fillingLength] = poca_D_inRange_sel_ord_scrub_reOrd
                poca_E_intervals[indexPosition,:fillingLength] = poca_E_inRange_sel_ord_scrub_reOrd
                poca_F_intervals[indexPosition,:fillingLength] = poca_F_inRange_sel_ord_scrub_reOrd
                
            else:
                ## No need for a special treatment here, as the inRange_sel containers 
                ## are already of size fillingLength
                ## Add the tracks information to the overall containers based on the index position
                poca_x_intervals[indexPosition,:fillingLength] = poca_x_inRange_sel
                poca_y_intervals[indexPosition,:fillingLength] = poca_y_inRange_sel
                poca_z_intervals[indexPosition,:fillingLength] = poca_z_inRange_sel
                poca_A_intervals[indexPosition,:fillingLength] = poca_A_inRange_sel
                poca_B_intervals[indexPosition,:fillingLength] = poca_B_inRange_sel
                poca_C_intervals[indexPosition,:fillingLength] = poca_C_inRange_sel
                poca_D_intervals[indexPosition,:fillingLength] = poca_D_inRange_sel
                poca_E_intervals[indexPosition,:fillingLength] = poca_E_inRange_sel
                poca_F_intervals[indexPosition,:fillingLength] = poca_F_inRange_sel
                
            ## END of loop over intervals
        ## END of loop over events
        
    ## Reshape the tracks variables containers to be able to concatenate them when building the tensors (see scripts/Build_Torch_Tensors.py) 
    poca_x_intervals  = poca_x_intervals[:,np.newaxis,:]
    poca_y_intervals  = poca_y_intervals[:,np.newaxis,:]
    poca_z_intervals  = poca_z_intervals[:,np.newaxis,:]
    poca_A_intervals  = poca_A_intervals[:,np.newaxis,:]
    poca_B_intervals  = poca_B_intervals[:,np.newaxis,:]
    poca_C_intervals  = poca_C_intervals[:,np.newaxis,:]
    poca_D_intervals  = poca_D_intervals[:,np.newaxis,:]
    poca_E_intervals  = poca_E_intervals[:,np.newaxis,:]
    poca_F_intervals  = poca_F_intervals[:,np.newaxis,:]
      
    ## Print some statistics from the 
    print("*"*100)
    print("")
    print("Statistics from tracks to intervals splitting:")
    print("")
    l_nSelTracks_nonzero = list(filter(lambda a: a != 0, l_nSelTracks))    
    print("Mean number of tracks per interval: ",np.mean(l_nSelTracks))    
    print("Mean number of tracks (zeros removed) per interval: ",np.mean(l_nSelTracks_nonzero))    
    print("Max  number of tracks per interval: ",np.max(l_nSelTracks))    
    print("Number of intervals: ",len(l_nSelTracks))    
    print("Number of intervals with more than one track: ",len(l_nSelTracks)-np.count_nonzero(l_nSelTracks))    
    print("Number of intervals per event: ",len(l_nSelTracks)/float(nEvts))    
    print("Number of intervals with more than one track per event: ",(len(l_nSelTracks)-np.count_nonzero(l_nSelTracks))/float(nEvts))    
    print("")
    print("*"*100)

    return InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple_intervals(
        getattr(input_tuple, "pv_x"),
        getattr(input_tuple, "pv_y"),
        getattr(input_tuple, "pv_z"),
        getattr(input_tuple, "pv_ntracks"),
        getattr(input_tuple, "pv_cat"),
        getattr(input_tuple, "pv_key"),
        
        getattr(input_tuple, "sv_x"),
        getattr(input_tuple, "sv_y"),
        getattr(input_tuple, "sv_z"),
        getattr(input_tuple, "sv_ntracks"),
        
        poca_x,               
        poca_y,
        poca_z,
        
        poca_A,               
        poca_B,              
        poca_C,              
        poca_D,               
        poca_E,              
        poca_F,
        
        poca_x_intervals,
        poca_y_intervals,
        poca_z_intervals,
        
        poca_A_intervals,
        poca_B_intervals,
        poca_C_intervals,
        poca_D_intervals,
        poca_E_intervals,
        poca_F_intervals,
                
        getattr(input_tuple, "major_axis_x"),        
        getattr(input_tuple, "major_axis_y"),        
        getattr(input_tuple, "major_axis_z"),        
        getattr(input_tuple, "minor_axis1_x"),       
        getattr(input_tuple, "minor_axis1_y"),       
        getattr(input_tuple, "minor_axis1_z"),       
        getattr(input_tuple, "minor_axis2_x"),       
        getattr(input_tuple, "minor_axis2_y"),       
        getattr(input_tuple, "minor_axis2_z"),      
        
        getattr(input_tuple, "recon_pv_key"),
        getattr(input_tuple, "recon_cat"), 
        getattr(input_tuple, "recon_ov_x"), 
        getattr(input_tuple, "recon_ov_y"), 
        getattr(input_tuple, "recon_ov_z"), 
        
        getattr(input_tuple, "recon_tx"), 
        getattr(input_tuple, "recon_ty"),
        
        getattr(input_tuple, "IP_KDE"),
        getattr(input_tuple, "IP_KDE_xMax"),
        getattr(input_tuple, "IP_KDE_yMax"),
        
        poca_KDE_A,
        poca_KDE_A_xMax,
        poca_KDE_A_yMax,
        poca_KDE_B,
        poca_KDE_B_xMax,
        poca_KDE_B_yMax,
        
        targetHists,
        
        poca_KDE_A_intervals,
        poca_KDE_A_xMax_intervals,
        poca_KDE_A_yMax_intervals,
        poca_KDE_B_intervals,
        poca_KDE_B_xMax_intervals,
        poca_KDE_B_yMax_intervals,
        
        targetHists_intervals
    ) 
    
    
    