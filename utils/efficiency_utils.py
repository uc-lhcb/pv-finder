import numba
import numpy as np
from typing import NamedTuple
from collections import Counter
from math import sqrt as sqrt

class ValueSet(NamedTuple):
    """
    Class to represent a results tuple. Adding and printing supported,
    along with a few convenience properties.
    """

    S:  int
    MT: int
    FP: int
    events: int = 1

    @property
    def real_pvs(self):
        return self.S + self.MT

    @property
    def eff_rate(self):
        if self.real_pvs==0:
            return self.real_pvs
        else:
            return self.S / self.real_pvs

    @property
    def fp_rate(self):
        return self.FP / self.events

    def __repr__(self):
        message = f"Found {self.S} of {self.real_pvs}, added {self.FP} (eff {self.eff_rate:.2%})"
        if self.events > 1:
            message += f" ({self.fp_rate:.4} FP/event)"
        #if self.S != self.Sp:
        #    message += f" ({self.S} != {self.Sp})"
        return message

    def __add__(self, other):
        return self.__class__(
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
            self[3] + other[3],
        )

    def pretty(self):
        s_message = f"Successes: {self.S:,}"
        return f"""\        
Real PVs in validation set: {self.real_pvs:,}
{s_message}
Missed true PVs: {self.MT:,}
False positives: {self.FP:,}
Efficiency of detecting real PVs: {self.eff_rate:.2%}
False positive rate: {self.fp_rate:.4}"""

    
class ValueSet_GNN(NamedTuple):
    """
    Class to represent a results tuple. Adding and printing supported,
    along with a few convenience properties.
    """

    S:  int
    MT: int
    FP: int
    events: int = 1

    @property
    def real_pvs(self):
        return self.S + self.MT

    @property
    def pred_pvs(self):
        return self.S + self.FP

    @property
    def eff_rate(self):
        if self.real_pvs==0:
            return self.real_pvs
        else:
            return self.S / self.real_pvs

    @property
    def fp_rate(self):
        return self.FP / self.events

    @property
    def fp_rate_pvs(self):
        return self.FP / self.pred_pvs

    def __repr__(self):
        message = f"Found {self.S} PVs out of {self.real_pvs} true PVs (Eff = {self.eff_rate:.2%}), \nAdded {self.FP} false PVs"
        if self.events > 1:
            message += f" corresponding to {self.fp_rate:.4} [FP/event] or"
            message += f" {self.fp_rate_pvs:.4} [FP/(tot. pred. PVs)]"
        #if self.S != self.Sp:
        #    message += f" ({self.S} != {self.Sp})"
        return message

    def __add__(self, other):
        return self.__class__(
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
            self[3] + other[3],
        )

    def pretty(self):
        s_message = f"Successes: {self.S:,}"
        return f"""\        
Real PVs in validation set: {self.real_pvs:,}
{s_message}
Missed true PVs: {self.MT:,}
False positives: {self.FP:,}
Efficiency of detecting real PVs: {self.eff_rate:.2%}
False positive rate [/evt]: {self.fp_rate:.4}
False positive rate [/tot. pred. pvs]: {self.fp_rate_pvs:.4}"""


    
#####################################################################################
@numba.jit(
    numba.float32[:](
        numba.float32[:],
        numba.float32[:]
    ), 
    nopython=True,
)
def filter_nans(
    items,
    mask
):
    """
    Method to mask bins in the predicted KDE array if the corresponding bin in the true KDE array is 'nan'.
    
    Inputs:
      * items: 
          Numpy array of predicted PV z positions

      * mask: 
          Numpy array of KDE values (true PVs)


    Returns:
      * Numpy array of predicted PV z positions
      
    """
    # Create empty array with shape array of predicted PV z positions
    retval = np.empty_like(items)
    # Counter of 
    max_index = 0
    # Loop over the predicted PV z positions
    for item in items:
        index = int(round(item))
        not_valid = np.isnan(mask[index])
        if not not_valid:
            retval[max_index] = item
            max_index += 1

    return retval[:max_index]
#####################################################################################
    
#####################################################################################
@numba.jit(
    numba.float32[:](
        numba.float32[:],
        numba.float32[:],
        numba.float32,
        numba.float32,
        numba.int16
    ), 
    nopython=True,
)
def remove_ghosts_PVs(
    pred_PVs_loc,
    predict,
    z_ghosts,
    h_ghosts, 
    debug
):
    
    """
    Return the list or pred_PVs_loc after ghosts being removed based on two variables:
         
         - z_ghosts (in number of bins): 
              
             2 predicted PVs that are close by each other (less than z_ghosts) 

         - h_ghosts: 
          
             AND where one hist signal is significantly higher than the other (h_ghosts)
            
    Inputs:
      * pred_PVs_loc: 
          Numpy array of computed z positions of the predicted PVs (using KDEs)

      * predict: 
          Numpy array of predictions

      * z_ghosts: 
          Window in which one of 2 predicted PVs could be removed
          
      * h_ghosts: 
          Difference threshold in KDE max values between the two predicted PVs to decide 
          if the smallest needs to be removed 

      * debug: 
          flag to print output for debugging purposes


    Ouputs: 
        Numpy array of filtered predicted PVs z position.
    """

    if debug:
        print("pred_PVs_loc",pred_PVs_loc)
    
    # List of PVs to be removed at the end (index from pred_PVs_loc)
    l_removed_ipvs=[]

    # Only consider the case with at least 2 PVs
    if len(pred_PVs_loc)>1:
        
        # Loop over the predicted PVs z location in bin number
        for PV_index in range(len(pred_PVs_loc)-1):
            
            if debug:
                print("Looking at PV index",PV_index)
                
            if PV_index in l_removed_ipvs:
                # The considered PV has already been removed
                if debug:
                    print("Considered PV index",PV_index)
                    print("already removed. Do nothing..")
                
                continue
                    

            # Get the centered bin number of the considered predicted PV
            pred_PV_loc_ibin = int(pred_PVs_loc[PV_index])

            # Get the max KDE of this 
            pred_PV_max = predict[pred_PV_loc_ibin]

            # Get the next predicted PV bin (centered on the max)
            next_pred_PV_loc_ibin = int(pred_PVs_loc[PV_index+1])
         
            # Now get the actual closest matched PV max and max ibin
            next_pred_PV_max = predict[next_pred_PV_loc_ibin]
            next_pred_PV_max_ibin = next_pred_PV_loc_ibin

            # Check if real max isn't actually the next or previous bin in the KDE hist
            for ibin in range(next_pred_PV_loc_ibin-1,next_pred_PV_loc_ibin+1):
                if predict[ibin] > next_pred_PV_max:
                    next_pred_PV_max = predict[ibin]
                    next_pred_PV_loc_ibin = ibin

            if debug:
                print("pred_PV_loc_ibin",pred_PV_loc_ibin)
                print("pred_PV_max",pred_PV_max)
                print("next_pred_PV_loc_ibin",next_pred_PV_loc_ibin)
                print("next_pred_PV_max",next_pred_PV_max)
                
                print("Delta between PVs",abs(next_pred_PV_loc_ibin-pred_PV_loc_ibin))
                    
            # Check if the next predicted PV is in the region to be considered as a ghosts
            if abs(next_pred_PV_loc_ibin-pred_PV_loc_ibin)<z_ghosts:

                # Compute the ratio of the closest_pred_PV_max over the pred_PV_max
                r_max = 0
                if not pred_PV_max==0:
                    r_max=next_pred_PV_max/pred_PV_max

                if debug:
                    print("r_max",r_max)
                    print("h_ghosts",h_ghosts)
                    if abs(h_ghosts)>0:
                        print("1./h_ghosts",1./h_ghosts)
                    
                # If the ratio is above the high threshold (h_ghosts)
                # then tag the predicted PV with the "smallest" hist max as to be removed 
                if r_max>h_ghosts:
                    l_removed_ipvs.append(PV_index)
                    if debug:
                        print("adding PV with index",PV_index)
                        print(" to the list of PVs to be removed")
                if abs(h_ghosts)>0 and r_max<(1./h_ghosts):
                    l_removed_ipvs.append(PV_index+1)
                    if debug:
                        print("adding PV with index",(PV_index+1))
                        print(" to the list of PVs to be removed")
                
    # Initally set the array of PVs to be returned (after ghosts removal) 
    # as equal to the input array of reconstructed PVs
    filtered_pred_PVs_loc = pred_PVs_loc
    
    # then loop over the list of indexes of PVs in the input array that needs to be removed
    # and remove them from the array of PVs to be returned
    for ipv in l_removed_ipvs:
        filtered_pred_PVs_loc = np.delete(filtered_pred_PVs_loc,[ipv])
    
    return filtered_pred_PVs_loc
#####################################################################################



#####################################################################################
@numba.jit(
    numba.float32[:](
        numba.float32[:],
        numba.float32,
        numba.float32,
        numba.int32
    ),
    locals={
        "integral": numba.float32,
        "sum_weights_locs": numba.float32
    },
    nopython=True,
)
def pv_locations_updated(
    targets,
    threshold,
    integral_threshold,
    min_width
):
    """
    Compute the z positions from the input KDE using the parsed criteria.
    
    Inputs:
      * targets: 
          Numpy array of KDE values (predicted or true)

      * threshold: 
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold: 
          The total integral required to trigger a hit - such as 0.2

      * min_width: 
          The minimum width (in bins) of a feature - such as 2

    Returns:
      * array of float32 values corresponding to the PV z positions
      
    """
    # Counter of "active bins" i.e. with values above input threshold value
    state = 0
    # Sum of active bin values
    integral = 0.0
    # Weighted Sum of active bin values weighted by the bin location
    sum_weights_locs = 0.0

    # Make an empty array and manually track the size (faster than python array)
    items = np.empty(150, np.float32)
    # Number of recorded PVs
    nitems = 0

    # Account for special case where two close PV merge KDE so that
    # targets[i] never goes below the threshold before the two PVs are scanned through
    peak_passed = False
    
    # Loop over the bins in the KDE histogram
    for i in range(len(targets)):
        # If bin value above 'threshold', then trigger
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_weights_locs += i * targets[i]  # weight times location

            if targets[i-1]>targets[i]:
                peak_passed = True
            
        if (targets[i] < threshold or i == len(targets) - 1 or (targets[i-1]<targets[i] and peak_passed)) and state > 0:
            #if (targets[i] < threshold or i == len(targets) - 1) and state > 0:

            # Record a PV only if 
            if state >= min_width and integral >= integral_threshold:
                # Adding '+0.5' to account for the bin width (i.e. 50 microns)
                items[nitems] = (sum_weights_locs / integral) + 0.5 
                nitems += 1

            # reset state
            state = 0
            integral = 0.0
            sum_weights_locs = 0.0
            peak_passed=False

    # Special case for final item (very rare or never occuring)
    # handled by above if len

    return items[:nitems]
#####################################################################################


#####################################################################################
@numba.jit(
    numba.float32[:](
        numba.float32[:],
        numba.float32,
        numba.float32,
        numba.int32
    ),
    locals={
        "integral": numba.float32,
        "sum_weights_locs": numba.float32
    },
    nopython=True,
)
def pv_locations(
    targets,
    threshold,
    integral_threshold,
    min_width
):
    """
    Compute the z positions from the input KDE using the parsed criteria.
    
    Inputs:
      * targets: 
          Numpy array of KDE values (predicted or true)

      * threshold: 
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold: 
          The total integral required to trigger a hit - such as 0.2

      * min_width: 
          The minimum width (in bins) of a feature - such as 2

    Returns:
      * array of float32 values corresponding to the PV z positions
      
    """
    # Counter of "active bins" i.e. with values above input threshold value
    state = 0
    # Sum of active bin values
    integral = 0.0
    # Weighted Sum of active bin values weighted by the bin location
    sum_weights_locs = 0.0

    # Make an empty array and manually track the size (faster than python array)
    items = np.empty(150, np.float32)
    # Number of recorded PVs
    nitems = 0

    # Loop over the bins in the KDE histogram
    for i in range(len(targets)):
        # If bin value above 'threshold', then trigger
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_weights_locs += i * targets[i]  # weight times location

        if (targets[i] < threshold or i == len(targets) - 1) and state > 0:

            # Record a PV only if 
            if state >= min_width and integral >= integral_threshold:
                # Adding '+0.5' to account for the bin width (i.e. 50 microns)
                items[nitems] = (sum_weights_locs / integral) + 0.5 
                nitems += 1

            # reset state
            state = 0
            integral = 0.0
            sum_weights_locs = 0.0

    # Special case for final item (very rare or never occuring)
    # handled by above if len

    return items[:nitems]
#####################################################################################


#####################################################################################
@numba.jit(
    numba.float32[:](
        numba.float32[:],
        numba.float32[:],
        numba.float32,
        numba.float32,
        numba.int16,
        numba.int16
    ), 
    nopython=True,
)
def get_resolution(
    pred_PVs_loc,
    predict,
    nsig_res,
    f_ratio_window,
    nbins_lookup,
    debug
):

    std_res = np.empty_like(pred_PVs_loc)

    max_bin = len(predict)-1
    
    for i_pred_PVs in range(len(pred_PVs_loc)):
        
        # First check whether the bin with the maxKDE is actually the one reported in pred_PVs_loc, 
        # as it is already a weighted value. Just check previous abd next bins
        pred_PV_loc_ibin = int(pred_PVs_loc[i_pred_PVs])
        if predict[pred_PV_loc_ibin-1]>predict[pred_PV_loc_ibin]:
            pred_PV_loc_ibin = pred_PV_loc_ibin-1
            if debug:
                print("Actual maximum shift to previous bin")
        if predict[pred_PV_loc_ibin+1]>predict[pred_PV_loc_ibin]:
            pred_PV_loc_ibin = pred_PV_loc_ibin+1
            if debug:
                print("Actual maximum shift to next bin")

        bins = []
        weights = []
        sum_bin_prod_weights = 0
        sum_weights = 0

        maxKDE = predict[pred_PV_loc_ibin]
        maxKDE_ratio = f_ratio_window*maxKDE
        if debug:
            print("maxKDE",maxKDE)
            print("bin(maxKDE)",pred_PV_loc_ibin)

        # Start by adding the values for the bin where KDE is maximum:
        bins.append(pred_PV_loc_ibin)
        weights.append(maxKDE)
        sum_bin_prod_weights += pred_PV_loc_ibin*maxKDE
        sum_weights += maxKDE
        
        # Now scan the "left side" (lower bin values) of the peak and add values to compute the KDE hist std value 
        # if the predicted hist is higher than f_ratio_window*maxKDE 
        # -- OR --
        # if predict[ibin]>predict[ibin+1] which means there is another peak on the "left" of the considered one...
        for ibin in range(pred_PV_loc_ibin-1,pred_PV_loc_ibin-nbins_lookup,-1):
            if debug:
                print("ibin",ibin)
            if predict[ibin]<maxKDE_ratio or predict[ibin]>predict[ibin+1] or ibin==0:
                if debug:
                    print("before",ibin,predict[ibin])
                    if predict[ibin]>predict[ibin+1]:
                        print("predict[ibin]>predict[ibin+1]::ibin,predict[ibin],predict[ibin+1]", ibin,predict[ibin],predict[ibin+1])
                    print("break")
                break
            else:
                if debug:
                    print("inside window",ibin,predict[ibin])
                bins.append(ibin)
                weights.append(predict[ibin])
                sum_bin_prod_weights += ibin*predict[ibin]
                sum_weights += predict[ibin]
                
        # Finally scan the "right side" (higher bin values) of the peak and add values to compute the KDE hist std value 
        # if the predicted hist is higher than f_ratio_window*maxKDE 
        # -- OR --
        # if predict[ibin]>predict[ibin-1] which means there is another peak on the "right" of the considered one...
        for ibin in range(pred_PV_loc_ibin+1,pred_PV_loc_ibin+nbins_lookup):
            if debug:
                print("ibin",ibin)
            if predict[ibin]<maxKDE_ratio or predict[ibin]>predict[ibin-1] or ibin==max_bin:
                if debug:
                    print("after",ibin,predict[ibin])
                    if predict[ibin]>predict[ibin-1]:
                        print("predict[ibin]>predict[ibin-1]::ibin,predict[ibin-1],predict[ibin]",ibin,predict[ibin-1],predict[ibin])
                    
                    print("break")
                break
            else:
                if debug:
                    print("inside window",ibin,predict[ibin])
                bins.append(ibin)
                weights.append(predict[ibin])
                sum_bin_prod_weights += ibin*predict[ibin]
                sum_weights += predict[ibin]

        mean = sum_bin_prod_weights/sum_weights
        
        #mean = sum(weights*bins)/sum(weights)
        if debug:
            print("weighted mean =",mean)
        #computed_mean[i_pred_PVs] = mean
        
        sum_diff_sq_prod_w = 0
        for i in range(len(bins)):
            #delta_sq.append((bins[i]-mean)*(bins[i]-mean)*weights[i])
            sum_diff_sq_prod_w += (bins[i]-mean)*(bins[i]-mean)*weights[i]
                    
        std = sqrt(sum_diff_sq_prod_w/sum_weights)        

        std_res[i_pred_PVs] = nsig_res*std
    
    return std_res
#####################################################################################


#####################################################################################
@numba.jit(
    numba.types.UniTuple(numba.int32, 3)(
        numba.float32[:],
        numba.float32[:],
        numba.float32[:],
        numba.int16
    ),
    nopython=True,
)
def compare(
    target_PVs_loc,
    pred_PVs_loc,
    reco_res,
    debug
):
    """
    Method to compute the efficiency counters: 
    - succeed    = number of successfully predicted PVs
    - missed     = number of missed true PVs
    - false_pos  = number of predicted PVs not matching any true PVs

    Inputs argument:
      * target_PVs_loc: 
          Numpy array of computed z positions of the true PVs (using KDEs)

      * pred_PVs_loc: 
          Numpy array of computed z positions of the predicted PVs (using KDEs)

      * reco_res: 
          Numpy array with the "reco" resolution as a function of width of predicted KDE signal 

      * debug: 
          flag to print output for debugging purposes
    
    
    Returns:
        succeed, missed, false_pos
    """
    
    # Counters that will be iterated and returned by this method
    succeed = 0
    missed = 0
    false_pos = 0
        
    # Get the number of predicted PVs
    len_pred_PVs_loc = len(pred_PVs_loc)
    # Get the number of true PVs 
    len_target_PVs_loc = len(target_PVs_loc)

    # Decide whether we have predicted equally or more PVs than trully present
    # this is important, because the logic for counting the MT an FP depend on this
    if len_pred_PVs_loc >= len_target_PVs_loc:
        if debug:
            print("In len(pred_PVs_loc) >= len(target_PVs_loc)")

        # Since we have N(pred_PVs) >= N(true_PVs), 
        # we loop over the pred_PVs, and check each one of them to decide 
        # whether they should be labelled as S, FP. 
        # The number of MT is computed as: N(true_PVs) - S
        # Here the number of iteration is fixed to the original number of predicted PVs
        for i in range(len_pred_PVs_loc):
            if debug:
                print("pred_PVs_loc = ",pred_PVs_loc[i])
            # flag to check if the predicted PV is being matched to a true PV
            matched = 0

            # Get the window of interest: [min_val, max_val] 
            # The window is obtained from the value of z of the true PV 'j'
            # +/- the resolution as a function of the number of tracks for the true PV 'j'
            min_val = pred_PVs_loc[i]-reco_res[i]
            max_val = pred_PVs_loc[i]+reco_res[i]
            if debug:
                print("resolution = ",(max_val-min_val)/2.)
                print("min_val = ",min_val)
                print("max_val = ",max_val)

            # Now looping over the true PVs.
            for j in range(len(target_PVs_loc)):
                # If condition is met, then the predicted PV is labelled as 'matched', 
                # and the number of success is incremented by 1
                if min_val <= target_PVs_loc[j] and target_PVs_loc[j] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ",succeed)
                    # the true PV is removed from the original array to avoid associating 
                    # one predicted PV to multiple true PVs
                    # (this could happen for PVs with close z values)
                    target_PVs_loc = np.delete(target_PVs_loc,[j])
                    # Since a predicted PV and a true PV where matched, go to the next predicted PV 'i'
                    break
            # In case, no true PV could be associated with the predicted PV 'i'
            # then it is assigned as a FP answer
            if not matched:                
                false_pos +=1
                if debug:
                    print("false_pos = ",false_pos)
        # the number of missed true PVs is simply the difference between the original 
        # number of true PVs and the number of successfully matched true PVs
        missed = (len_target_PVs_loc-succeed)
        if debug:
            print("missed = ",missed)

    else:
        if debug:
            print("In len(pred_PVs_loc) < len(target_PVs_loc)")
        # Since we have N(pred_PVs) < N(true_PVs), 
        # we loop over the true_PVs, and check each one of them to decide 
        # whether they should be labelled as S, MT. 
        # The number of FP is computed as: N(pred_PVs) - S
        # Here the number of iteration is fixed to the original number of true PVs
        for i in range(len_target_PVs_loc):
            if debug:
                print("target_PVs_loc = ",target_PVs_loc[i])
            # flag to check if the true PV is being matched to a predicted PV
            matched = 0
            # Now looping over the predicted PVs.
            for j in range(len(pred_PVs_loc)):                
                # Get the window of interest: [min_val, max_val] 
                # The window is obtained from the value of z of the true PV 'i'
                # +/- the resolution as a function of the number of tracks for the true PV 'i'
                min_val = pred_PVs_loc[j]-reco_res[j]
                max_val = pred_PVs_loc[j]+reco_res[j]
                if debug:
                    print("pred_PVs_loc = ",pred_PVs_loc[j])
                    print("resolution = ",(max_val-min_val)/2.)
                    print("min_val = ",min_val)
                    print("max_val = ",max_val)
                # If condition is met, then the true PV is labelled as 'matched', 
                # and the number of success is incremented by 1
                if min_val <= target_PVs_loc[i] and target_PVs_loc[i] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ",succeed)
                    # the predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    pred_PVs_loc = np.delete(pred_PVs_loc,[j])
                    # Since a predicted PV and a true PV where matched, go to the next true PV 'i'
                    reco_res = np.delete(reco_res,[j])
                    break
            # In case, no predicted PV could be associated with the true PV 'i'
            # then it is assigned as a MT answer
            if not matched:
                missed += 1
                if debug:
                    print("missed = ",missed)
                    
        # the number of false positive predicted PVs is simply the difference between the original 
        # number of predicted PVs and the number of successfully matched predicted PVs
        false_pos = (len_pred_PVs_loc - succeed)
        if debug:
            print("false_pos = ",false_pos)

    return succeed, missed, false_pos
#####################################################################################


#####################################################################################
@numba.jit(
    numba.int16[:](
        numba.int16,
        
        numba.float32[:],
        numba.float32[:],
        
        numba.int16,
        numba.float32,
        numba.float32,
        
        numba.float32,
        numba.float32,
        numba.float32,
        
        numba.int32, 
        numba.int16
    ),
    nopython=True,
)
def get_PVs_label(
    get_Preds,

    truth, 
    predict, 

    use_locations_updated,
    nsig_res_std, 
    f_ratio_window,

    nbins_lookup,
    threshold, 
    integral_threshold, 

    min_width, 
    debug
):
    
    """
    Method to obtain the PVs labels (i.e. list of true PV that are matched to predicted PV, or vice-versa).

    Inputs:
      * truth: 
          Numpy array of truth values

      * predict: 
          Numpy array of predictions

      * true_PVs_nTracks: 
          Numpy array with the number of tracks originating from the true PV 
          Ordered from the generator level (random in z)

      * true_PVs_z: 
          Numpy array with the z position of the true PVs.
          Ordered from the generator level (random in z) 
          It is necessary when computing the resolution (association between 
          the correct true PV and the corresponding number of tracks)

      * nsig_res_std: 
          Number of sigma of var(peak) 

      * min_res: 
          Minimal resolution value (in terms of bins) for the search window - such as 3

      * threshold: 
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold: 
          The total integral required to trigger a hit - such as 0.2

      * min_width: 
          The minimum width (in bins) of a feature - such as 2

      * debug: 
          flag to print output for debugging purposes


    Ouputs: 
        Numpy array of '0=not matched' or '1=matched' acting like booleans.
    """
    
    # Get the z position from the true KDEs distribution
    target_PVs_loc = pv_locations(truth, threshold, integral_threshold, min_width)
    # Get the z position from the predicted KDEs distribution
    pred_PVs_loc = pv_locations(predict, threshold, integral_threshold, min_width)

    if use_locations_updated:
        # Get the z position from the true KDEs distribution
        target_PVs_loc = pv_locations_updated(truth, threshold, integral_threshold, min_width)
        # Get the z position from the predicted KDEs distribution
        pred_PVs_loc = pv_locations_updated(predict, threshold, integral_threshold, min_width)
    
        
    # Using the filter_nans_res method to mask the PVs in 'pred_PVs_loc' 
    # where the corresponding bins in truth are 'nan' 
    filtered_pred_PVs_loc = filter_nans(pred_PVs_loc, truth)

    if debug:
        print("target_PVs_loc:         ",target_PVs_loc)
        print("filtered_pred_PVs_loc:  ",filtered_pred_PVs_loc)
    
    # Get the true PV resolutions, sorted by ascending z value position
    # The sorting in z values is important, because the arrays target_PVs_loc 
    # and pred_PVs_loc obtained from 'pv_locations_updated_res' are sorted by ascending z values 
    # (by construction from the KDEs histograms)
    filtered_and_sorted = get_resolution(filtered_pred_PVs_loc, predict, nsig_res_std, f_ratio_window, nbins_lookup, 0)
    
    if debug:
        print("filtered_and_sorted:  ",filtered_and_sorted)

    # Get the number of predicted PVs
    len_pred_PVs_loc = len(filtered_pred_PVs_loc)
    # Get the number of true PVs 
    len_target_PVs_loc = len(target_PVs_loc)
        
    # ========================================================================
    # Check if the target true PVs are matched to a predicted PV
    # if not these would be considered as missed
    if get_Preds==0:
        
        # Initialize the array to an array of zeros with target_PVs_loc shape 
        PVs_label = np.zeros(target_PVs_loc.shape,dtype=numba.int16)
        
        # Looping over the true PVs.
        for i in range(len_target_PVs_loc):
            if debug:
                print("target_PVs_loc = ",target_PVs_loc[i])
            
            # Now looping over the predicted PVs.
            for j in range(len(filtered_pred_PVs_loc)):
                if debug:
                    print("pred_PVs_loc = ",filtered_pred_PVs_loc[j])
                min_val = filtered_pred_PVs_loc[j]-filtered_and_sorted[j]
                max_val = filtered_pred_PVs_loc[j]+filtered_and_sorted[j]
                if debug:
                    print("resolution = ",(max_val-min_val)/2.)
                    print("min_val = ",min_val)
                    print("max_val = ",max_val)

                # If condition is met, then the true PV is labelled as 'matched', 
                # and PVs_label[i] is set to 1
                if min_val <= target_PVs_loc[i] and target_PVs_loc[i] <= max_val:
                    PVs_label[i] = 1
                    if debug:
                        print("succeed = ",PVs_label[i])
                    # the predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    filtered_pred_PVs_loc = np.delete(filtered_pred_PVs_loc,[j])
                    # Also remove the associated predicted PV resolution:
                    filtered_and_sorted = np.delete(filtered_and_sorted,[j])
                    # Since a predicted PV and a true PV where matched, go to the next true PV 'i'
                    break
                    
        return PVs_label
        
    # ========================================================================
    # Check if the outpout predicted PVs are matched to a true PV
    # if not these would be considered as false positive
    elif get_Preds==1:

        # Initialize the array to an array of zeros with target_PVs_loc shape 
        PVs_label = np.zeros(filtered_pred_PVs_loc.shape,dtype=numba.int16)
        
        # Looping over the true PVs.
        for i in range(len_pred_PVs_loc):
            if debug:
                print("filtered_pred_PVs_loc = ",filtered_pred_PVs_loc[i])
            
            # Now looping over the true PVs.
            for j in range(len(target_PVs_loc)):
                if debug:
                    print("target_PVs_loc = ",target_PVs_loc[j])
                min_val = filtered_pred_PVs_loc[i]-filtered_and_sorted[i]
                max_val = filtered_pred_PVs_loc[i]+filtered_and_sorted[i]
                if debug:
                    print("resolution = ",(max_val-min_val)/2.)
                    print("min_val = ",min_val)
                    print("max_val = ",max_val)

                # If condition is met, then the true PV is labelled as 'matched', 
                # and PVs_label[i] is set to 1
                if min_val <= target_PVs_loc[j] and target_PVs_loc[j] <= max_val:
                    PVs_label[i] = 1
                    if debug:
                        print("succeed = ",PVs_label[i])
                    # the predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    target_PVs_loc = np.delete(target_PVs_loc,[j])
                    # Since a predicted PV and a true PV where matched, go to the next true PV 'i'
                    break
                    
        return PVs_label

    else:
        PVs_label = np.zeros(filtered_pred_PVs_loc.shape,dtype=numba.int16)
        print("Wrong value for the first argument in get_PVs_label")
        print("Needs to be either 0 (get true PV labels) or 1 (predicted PV labels)")
        
        return PVs_label
#####################################################################################

#####################################################################################
@numba.jit(
    numba.types.UniTuple(numba.int32, 3)(
        numba.float32[:],
        numba.float32[:],

        numba.float32,
        numba.float32,
        numba.float32,
        
        numba.float32,
        numba.float32,
        numba.int16,
        
        numba.int16,
        
        numba.int16,
        numba.float32,
        numba.float32,

        numba.int16
    ),
    nopython=True,
)
def numba_efficiency(
    truth,
    predict,
    
    threshold,
    integral_threshold,
    min_width,
    
    nsig_res,
    f_ratio_window,
    nbins_lookup,
    
    use_locations_updated,

    remove_ghosts,
    z_ghosts,
    h_ghosts,

    debug    
):
    """
    Function copied from pv-finder 'efficiency.py', which now returns 3 values instead of 4. Two values of counted successes are computed in 'efficiency.py', S and Sp, the latter being the number of successes when using the filter_nans method on the true KDE inputs. By default, the number of successes (S in this script) is being computed using the filter_nans method.
    """

    # Get the z position from the true KDEs distribution
    true_values = pv_locations(truth, threshold, integral_threshold, min_width)
    # Get the z position from the predicted KDEs distribution
    predict_values = pv_locations(predict, threshold, integral_threshold, min_width)

    if use_locations_updated:
        # Get the z position from the true KDEs distribution
        true_values = pv_locations_updated(truth, threshold, integral_threshold, min_width)
        # Get the z position from the predicted KDEs distribution
        predict_values = pv_locations_updated(predict, threshold, integral_threshold, min_width)

        
    # Using the filter_nans method to 'mask' the bins in 'predict_values' 
    # where the corresponding bins in truth are 'nan' 
    filtered_predict_values = filter_nans(predict_values, truth)
    
    # Get the efficiency values: 
    # - S (number of successfully predicted PVs); 
    # - MT (missed true PVs); 
    # - FP (false positive predicted PVs)
    
    # ======================================================================
    # Use the resolution as a the RMS of the predicted hist
    # ======================================================================

    # Remove the predicted PVs which are likelly to be ghosts: 
    # i.e. two predicted PVs that are close by each other (less than z_ghosts) 
    #      AND where one hist signal is significantly higher than the other (h_ghosts)
    if remove_ghosts:
        filtered_predict_values = remove_ghosts_PVs(filtered_predict_values, predict, z_ghosts, h_ghosts, debug)

    # WARNING:  get_resolution needs to take as argument the filtered list of predicted PVs location

    res = get_resolution(filtered_predict_values, predict, nsig_res, f_ratio_window, nbins_lookup, debug)

    S, MT, FP = compare(true_values, filtered_predict_values, res, debug)
    return S, MT, FP
            
#####################################################################################


#####################################################################################
def eval_efficiency(
    truth,
    predict,
    
    threshold,
    integral_threshold,
    min_width,

    nsig_res,
    f_ratio_window,
    nbins_lookup,

    use_locations_updated,

    remove_ghosts,
    z_ghosts,
    h_ghosts,

    debug
):
    """
    Compute three values: The number of succeses (S), the number of missed true
    values (MT), and the number of missed false values (FP). Note that the number of successes
    is computed twice, and both values are returned.

    ########################
    Inputs:
      ------------
      * truth:    Numpy array of truth values (target hist)

      * predict:  Numpy array of predictions (from the trained model)

      ------------
      * Resolution parameters:
          
          - nsig_res:       number of sigmas from which we multiply the resolution 
                            in the search for a true PV nearby a predicted one
                      
          - f_ratio_window: fraction (between 0 and 1) of the maximum predicted peak. 
                            Acts as a condition when performing the scan to compute the weighted
                            standard deviation.
          
          - nbins_lookup:   numbers of bins to scan (symmetric window around the pedicted PV position in z) 
                            to compute the resolution as the weighted standard deviation of the predicted peak 
                          
      ------------
      * Valid predicted peak parameters:
      
          - threshold:          idividual bin value for starting considering an "on" value - such as 1e-2

          - integral_threshold: "on" bins integral required to trigger a valid predicted PV - such as 0.2

          - min_width:          minimum width (in bins) of a feature - such as 2


      ------------
      * Ghosts removal function:
      
          Allow removing predicted PVs which are likelly to be ghosts based on two variables:
          
          - remove_ghosts: bool to use or not this functionnality
            
          - z_ghosts: distance for which two predicted PVs that are close by each other 
                           (less than z_ghosts) should be removed

          - h_ghosts: if one predicted hist is significantly higher than a nearby predicted (h_ghosts)
        
      ------------
      * debug: 
      
          flag to print output for debugging purposes


    ########################
    Ouputs: 
    
        ValueSet(S, Sp, MT, FP)

    This algorithm computes the weighted mean, and uses that.
    This avoids small fluctionations in the input array by requiring .
    a minium total integrated value required to "turn it on"
    (integral_threshold=0.2) and min_width of 3 bins wide.
    """

    return ValueSet(
        *numba_efficiency(
            truth, predict,
            threshold, integral_threshold, min_width, 
            nsig_res, f_ratio_window, nbins_lookup,
            use_locations_updated,
            remove_ghosts, z_ghosts, h_ghosts, 
            debug
        )
    )
#####################################################################################
