import numpy as np
from numba import jit

@jit(nopython=True)
def get_tracks_info_intervals(poca_x_evt, 
                              poca_y_evt, 
                              poca_z_evt, 
                              poca_A_evt, 
                              poca_B_evt, 
                              poca_C_evt, 
                              poca_D_evt, 
                              poca_E_evt, 
                              poca_F_evt,
                              interval_lowEdge,
                              interval_z_min, 
                              interval_z_max, 
                              maxTracksPerInterval,
                              maxSigmaX, 
                              maxSigmaY, 
                              maxSigmaZ):
    
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

    ## Apply a shift to the poca_z values in each interval so that 
    ## all tracks z values are "invariant" across all intervals
    poca_z_inRange = poca_z_inRange - interval_lowEdge

    ## Now compute the selection to keep only tracks that have "low"
    ## uncertainties, i.e. good quality tracks
    sigmaX = np.sqrt(np.divide(1.,poca_A_inRange))
    sigmaY = np.sqrt(np.divide(1.,poca_B_inRange))
    sigmaZ = np.sqrt(np.divide(1.,poca_C_inRange))

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

    ## Check if the number of selected tracks is larger or smaller 
    ## than the maximum number of tracks per interval defined in the 
    ## config file
    ## Get the number of selected tracks
    nSelTracks = len(poca_z_inRange_sel)
    ## Define the number of tracks to be persisted
    fillingLength = min(nSelTracks,maxTracksPerInterval)


    if nSelTracks > maxTracksPerInterval:
        ## Since the number of tracks remaining in the extended interval is larger than the
        ## defined maximum number of tracks to be kept, we need to remove tracks...
        ## This should be done according to the sigmaZ values. 
        '''
        PRINT STATEMENT IN JIT ARE NOT AVAILABLE AS THIS >> TO BE CHECKED
        print("At event %s, interval %s (Z in [%.2f, %.2f]):"%(i_Evt,i_Interval,interval_z_min,interval_z_max))
        print("`---> the number of selected tracks is %s; larger than the max number of tracks %s"%(nSelTracks,maxTracksPerInterval))
        print("`---> %s tracks will be removed according to their sigmaZ values (largest are discarded)"%(nSelTracks-maxTracksPerInterval))
        '''
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
        return poca_x_inRange_sel_ord_scrub_reOrd, poca_y_inRange_sel_ord_scrub_reOrd, poca_z_inRange_sel_ord_scrub_reOrd, poca_A_inRange_sel_ord_scrub_reOrd, poca_B_inRange_sel_ord_scrub_reOrd, poca_C_inRange_sel_ord_scrub_reOrd, poca_D_inRange_sel_ord_scrub_reOrd, poca_E_inRange_sel_ord_scrub_reOrd, poca_F_inRange_sel_ord_scrub_reOrd
    

    else:
        ## No need for a special treatment here, as the inRange_sel containers 
        ## are already of size fillingLength
        ## Add the tracks information to the overall containers based on the index position
        return poca_x_inRange_sel, poca_y_inRange_sel, poca_z_inRange_sel, poca_A_inRange_sel, poca_B_inRange_sel, poca_C_inRange_sel, poca_D_inRange_sel, poca_E_inRange_sel, poca_F_inRange_sel
    
