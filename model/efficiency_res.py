import numba
import numpy as np
from typing import NamedTuple
from collections import Counter


class ValueSet_res(NamedTuple):
    """
    Class to represent a results tuple. Adding and printing supported,
    along with a few convenience properties.
    """

    S: int
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
            message += f" ({self.fp_rate:.3} FP/event)"
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
False positive rate: {self.fp_rate:.3}"""

    

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
def pv_locations_updated_res(
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
def pv_locations_res(
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
        numba.float32[:]
    ), 
    nopython=True,
)
def filter_nans_res(
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
        numba.uint16[:],
        numba.float64[:],
        #numba.float32[:], 
        numba.float32,
        numba.float32,
        numba.int16
    ), 
    nopython=True,
)
def get_resolution(
    target_PVs_loc,
    true_PVs_nTracks,
    true_PVs_z,
    nsig_res,
    min_res,
    debug
):
    
    """
    Compute the resolution as a function of true_PVs_nTracks

    Inputs:
      * target_PVs_loc: 
          Numpy array of computed z positions of the true PVs (using KDEs)

      * true_PVs_nTracks: 
          Numpy array with the number of tracks originating from the true PV 
          Ordered from the generator level (random in z)

      * true_PVs_z: 
          Numpy array with the z position of the true PVs (from generator).
          Ordered from the generator level (random in z) 
          It is necessary when computing the resolution (association between 
          the correct true PV and the corresponding number of tracks)

      * nsig_res: 
          Empirical value representing the number of sigma wrt to the std resolution 
          as a function of true_PVs_nTracks - such as 5

      * min_res: 
          Minimal resolution value (in terms of bins) for the search window - such as 3

      * debug: 
          flag to print output for debugging purposes


    Ouputs: 
        Numpy array of filtered (nTracks>4) and sorted (in z values) expected resolution on the true PVs z position.
    """
    
    # First get the number of tracks for true PVs with true_PVs_nTracks > 4, 
    # and sorted in ascending z value position:
    #filtered_and_sorted_true_PVs_nTracks = np.empty(len(true_PVs_z[true_PVs_nTracks > 4]), np.float32)
    filtered_and_sorted_true_PVs_nTracks = [i[1] for i in sorted( zip((true_PVs_z[true_PVs_nTracks > 4]), true_PVs_nTracks[true_PVs_nTracks > 4]))]

    if debug:
        print("Sorted number of tracks (get_resolution): ",filtered_and_sorted_true_PVs_nTracks)

    # then compute the resolution using the following constants 
    # used in calculating pvRes from Ref LHCb-PUB-2017-005 (original values in microns)
    A_res = 926.0
    B_res = 0.84
    C_res = 10.7

    ## scaling factor to changes units
    scale = 0.01 # This scale allows a correct conversion from the target histograms of 4000 bins of width 100 microns and used elsewhere in the code. 
    #scale = 1.0 #microns
    #scale = 0.001 #mm

    filtered_and_sorted_res = np.empty_like(target_PVs_loc)
    
    for i in range(len(filtered_and_sorted_true_PVs_nTracks)):
        filtered_and_sorted_res[i] = nsig_res * ( scale * (A_res * np.power(filtered_and_sorted_true_PVs_nTracks[i], -1 * B_res) + C_res))
    #filtered_and_sorted_res = (nsig_res*0.01* (A_res * np.power(filtered_and_sorted_true_PVs_nTracks, -1.0 * B_res) + C_res))

    # Replace resolution values below min_res by min_res itself
    filtered_and_sorted_res = np.where(filtered_and_sorted_res < min_res, min_res, filtered_and_sorted_res)
    
    return filtered_and_sorted_res
#####################################################################################



#####################################################################################
@numba.jit(
    numba.int16[:](
        numba.int16,
        numba.float32[:],
        numba.float32[:],
        numba.uint16[:],
        numba.float64[:],
        #numba.float32[:],
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
    true_PVs_nTracks, 
    true_PVs_z, 
    nsig_res, 
    min_res, 
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

      * nsig_res: 
          Empirical value representing the number of sigma wrt to the std resolution 
          as a function of true_PVs_nTracks - such as 5

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
    target_PVs_loc = pv_locations_res(truth, threshold, integral_threshold, min_width)
    # Get the z position from the predicted KDEs distribution
    pred_PVs_loc = pv_locations_res(predict, threshold, integral_threshold, min_width)

    # Using the filter_nans_res method to mask the PVs in 'pred_PVs_loc' 
    # where the corresponding bins in truth are 'nan' 
    filtered_pred_PVs_loc = filter_nans_res(pred_PVs_loc, truth)

    # Get the true PV resolutions, sorted by ascending z value position
    # The sorting in z values is important, because the arrays target_PVs_loc 
    # and pred_PVs_loc obtained from 'pv_locations_res' are sorted by ascending z values 
    # (by construction from the KDEs histograms)
    filtered_and_sorted_res = get_resolution(target_PVs_loc, true_PVs_nTracks, true_PVs_z, nsig_res, min_res, debug)
    
    
    if get_Preds==0:
        # Initialize the array to an array of zeros with target_PVs_loc shape 
        PVs_label = np.zeros(target_PVs_loc.shape,dtype=numba.int16)
        # Loop over the true PVs
        for i in range(len(target_PVs_loc)):
            # Get the window of interest: [min_val, max_val] 
            # The window is obtained from the value of z of the true PV 'i'
            # +/- the resolution as a function of the number of tracks for the true PV 'i'
            min_val = target_PVs_loc[i]-filtered_and_sorted_res[i]
            max_val = target_PVs_loc[i]+filtered_and_sorted_res[i]
            # Loop over the 'filtered' predicted PVs
            for j in range(len(filtered_pred_PVs_loc)):                
                if min_val <= filtered_pred_PVs_loc[j] and filtered_pred_PVs_loc[j] <= max_val:
                    # If condition is met, then the element 'i' 
                    # of the PVs_label array is set to '1' 
                    PVs_label[i] = 1
                    # the predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    filtered_pred_PVs_loc = np.delete(filtered_pred_PVs_loc,[j])
                    # Since a true PV and a predicted PV where matched, go to the next true PV 'i'
                    break
                else:
                    # In case, no predicted PV could be associated with the true PV 'i'
                    # then PVs_label[i] is set to '0'
                    PVs_label[i] = 0
        return PVs_label           
    
    elif get_Preds==1:
        PVs_label = np.zeros(filtered_pred_PVs_loc.shape,dtype=numba.int16)
        # Loop over the 'filtered' predicted PVs
        for i in range(len(filtered_pred_PVs_loc)):

            # Loop over the true PVs
            for j in range(len(target_PVs_loc)):                

                # Get the window of interest: [min_val, max_val] 
                # The window is obtained from the value of z of the true PV 'i'
                # +/- the resolution as a function of the number of tracks for the true PV 'i'
                min_val = target_PVs_loc[j]-filtered_and_sorted_res[j]
                max_val = target_PVs_loc[j]+filtered_and_sorted_res[j]

                if min_val <= filtered_pred_PVs_loc[i] and filtered_pred_PVs_loc[i] <= max_val:
                    # If condition is met, then the element 'i' 
                    # of the PVs_label array is set to '1' 
                    PVs_label[i] = 1

                    # The predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    target_PVs_loc = np.delete(target_PVs_loc,[j])
                    # also remove the associated resolution to avoid mis-matching the array dimensions
                    filtered_and_sorted_res = np.delete(filtered_and_sorted_res,[j])
                    
                    # Since a true PV and a predicted PV where matched, go to the next true PV 'i'
                    break
                else:
                    # In case, no predicted PV could be associated with the true PV 'i'
                    # then PVs_label[i] is set to '0'
                    PVs_label[i] = 0
        return PVs_label           

    else:
        PVs_label = np.zeros(filtered_pred_PVs_loc.shape,dtype=numba.int16)
        print("Wrong value for the first argument in get_PVs_label")
        print("Needs to be either 0 (get true PV labels) or 1 (predicted PV labels)")
        
        return PVs_label
#####################################################################################


#####################################################################################
@numba.jit(
    numba.int16[:](
        numba.float32[:],
        numba.uint16[:],
        numba.float64[:],
        #numba.float32[:],
        numba.int16
    ), 
    nopython=True,
)
def get_nTracks_sorted(
    target_PVs_loc,
    true_PVs_nTracks,
    true_PVs_z,
    debug
):
    """
    Method to obtain the filtered and sorted list of number of tracks associated to each PVs

    Input argument:
      * target_PVs_loc: 
          Numpy array of computed z positions of the true PVs (using KDEs)

      * true_PVs_nTracks: 
          Numpy array with the number of tracks originating from the true PV 
          Ordered from the generator level (random in z)

      * true_PVs_z: 
          Numpy array with the z position of the true PVs.
          Ordered from the generator level (random in z) 
          It is necessary when computing the resolution (association between 
          the correct true PV and the corresponding number of tracks)

      * debug: 
          flag to print output for debugging purposes
    """

    # Get an empty array fo shape target_PVs_loc with values of type 'int16'
    filtered_and_sorted_true_PVs_nTracks = np.empty(target_PVs_loc.shape,dtype=numba.int16)
    
    # Sorted loop over the true PVs using the z position in 'true_PVs_z' 
    # with the condition 'true_PVs_nTracks > 4'. 
    # Fill the array with the associated number of tracks.
    j=0
    for i in sorted( zip(true_PVs_z[true_PVs_nTracks > 4], true_PVs_nTracks[true_PVs_nTracks > 4])):
        filtered_and_sorted_true_PVs_nTracks[j] = int(i[1])
        j+=1
     
    if debug:
        print("Sorted number of tracks (get_nTracks_sorted): ",filtered_and_sorted_true_PVs_nTracks)

    return filtered_and_sorted_true_PVs_nTracks
#####################################################################################


#####################################################################################
@numba.jit(
    numba.types.UniTuple(numba.int32, 3)(
        numba.float32[:],
        numba.float32[:],
        numba.uint16[:],
        numba.float64[:],
        #numba.float32[:],
        numba.float32,
        numba.float32,
        numba.int16
    ),
    nopython=True,
)
def compare_res(
    target_PVs_loc,
    pred_PVs_loc,
    true_PVs_nTracks,
    true_PVs_z,
    nsig_res,
    min_res,
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

      * true_PVs_nTracks: 
          Numpy array with the number of tracks originating from the true PV 
          Ordered from the generator level (random in z)

      * true_PVs_z: 
          Numpy array with the z position of the true PVs.
          Ordered from the generator level (random in z) 
          It is necessary when computing the resolution (association between 
          the correct true PV and the corresponding number of tracks)

      * nsig_res: 
          Empirical value representing the number of sigma wrt to the std resolution 
          as a function of true_PVs_nTracks - such as 5

      * min_res: 
          Minimal resolution value (in terms of bins) for the search window - such as 3

      * debug: 
          flag to print output for debugging purposes
    
    
    Returns:
        succeed, missed, false_pos
    """
    
    # Counters that will be iterated and returned by this method
    succeed = 0
    missed = 0
    false_pos = 0
    
    # Get the true PV resolutions, sorted by ascending z value position
    # The sorting in z values is important, because the arrays target_PVs_loc 
    # and pred_PVs_loc obtained from 'pv_locations_res' are sorted by ascending z values 
    # (by construction from the KDEs histograms)
    filtered_and_sorted_res = get_resolution(target_PVs_loc, true_PVs_nTracks, true_PVs_z, nsig_res, min_res, debug)
    
    if debug:
        print("")
        print("pred_PVs_loc = ",pred_PVs_loc)
        print("target_PVs_loc = ",target_PVs_loc)
        print("resolutions : ",filtered_and_sorted_res)
        print("")
        print("len(pred_PVs_loc) =",len(pred_PVs_loc))
        print("len(target_PVs_loc) =",len(target_PVs_loc))
        print("")
        print("")

    
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
            # Now looping over the true PVs.
            for j in range(len(target_PVs_loc)):
                # Get the window of interest: [min_val, max_val] 
                # The window is obtained from the value of z of the true PV 'j'
                # +/- the resolution as a function of the number of tracks for the true PV 'j'
                min_val = target_PVs_loc[j]-filtered_and_sorted_res[j]
                max_val = target_PVs_loc[j]+filtered_and_sorted_res[j]
                if debug:
                    print("resolution = ",(max_val-min_val)/2.)
                    print("min_val = ",min_val)
                    print("max_val = ",max_val)
                # If condition is met, then the predicted PV is labelled as 'matched', 
                # and the number of success is incremented by 1
                if min_val <= pred_PVs_loc[i] and pred_PVs_loc[i] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ",succeed)
                    # the true PV is removed from the original array to avoid associating 
                    # one predicted PV to multiple true PVs
                    # (this could happen for PVs with close z values)
                    target_PVs_loc = np.delete(target_PVs_loc,[j])
                    # also remove the associated resolution to avoid mis-matching the array dimensions
                    filtered_and_sorted_res = np.delete(filtered_and_sorted_res,[j])
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
            # Get the window of interest: [min_val, max_val] 
            # The window is obtained from the value of z of the true PV 'i'
            # +/- the resolution as a function of the number of tracks for the true PV 'i'
            min_val = target_PVs_loc[i]-filtered_and_sorted_res[i]
            max_val = target_PVs_loc[i]+filtered_and_sorted_res[i]
            if debug:
                print("resolution = ",(max_val-min_val)/2.)
                print("min_val = ",min_val)
                print("max_val = ",max_val)
            # flag to check if the true PV is being matched to a predicted PV
            matched = 0
            # Now looping over the predicted PVs.
            for j in range(len(pred_PVs_loc)):                
                if debug:
                    print("pred_PVs_loc = ",pred_PVs_loc[j])
                # If condition is met, then the true PV is labelled as 'matched', 
                # and the number of success is incremented by 1
                if min_val <= pred_PVs_loc[j] and pred_PVs_loc[j] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ",succeed)
                    # the predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    pred_PVs_loc = np.delete(pred_PVs_loc,[j])
                    # Since a predicted PV and a true PV where matched, go to the next true PV 'i'
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
    numba.types.UniTuple(numba.int32, 3)(
        numba.float32[:],
        numba.float32[:],
        numba.uint16[:],
        numba.float64[:],
        #numba.float32[:],
        numba.float32,
        numba.float32,
        numba.float32,
        numba.float32,
        numba.int32,
        numba.int16
    ),
    nopython=True,
)
def numba_efficiency_res(
    truth,
    predict,
    true_PVs_nTracks,
    true_PVs_z,
    nsig_res,
    min_res,
    threshold,
    integral_threshold,
    min_width,
    debug
):
    """
    Function copied from 'efficiency.py', which now returns 3 values instead of 4. Two values of counted successes are computed in 'efficiency.py', S and Sp, the latter being the number of successes when using the filter_nans method on the true KDE inputs. By default, the number of successes (S in this script) is being computed using the filter_nans_res method.
    """

    # Get the z position from the true KDEs distribution
    true_values = pv_locations_res(truth, threshold, integral_threshold, min_width)
    # Get the z position from the predicted KDEs distribution
    predict_values = pv_locations_res(predict, threshold, integral_threshold, min_width)

    
    # Using the filter_nans_res method to 'mask' the bins in 'predict_values' 
    # where the corresponding bins in truth are 'nan' 
    filtered_predict_values = filter_nans_res(predict_values, truth)

    # Get the efficiency values: 
    # - S (number of successfully predicted PVs); 
    # - MT (missed true PVs); 
    # - FP (false positive predicted PVs)
    S, MT, FP = compare_res(true_values, filtered_predict_values, true_PVs_nTracks, true_PVs_z, nsig_res, min_res, debug)

    return S, MT, FP
#####################################################################################


#####################################################################################
def efficiency_res(
    truth,
    predict,
    true_PVs_nTracks,
    true_PVs_z, 
    nsig_res,
    min_res,
    threshold,
    integral_threshold,
    min_width,
    debug
):
    """
    Compute three values: The number of succeses (S), the number of missed true
    values (MT), and the number of missed false values (FP). Note that the number of successes
    is computed twice, and both values are returned.

    Inputs:
      * truth: 
          Numpy array of truth values

      * predict: 
          Numpy array of predictions

      * true_PVs_nTracks: 
          The number of tracks originating from the true PV used to compute diff(true_PVs_nTracks)

      * true_PVs_z: 
          Z position of the true PVs used to assign the correct true PV to true_PVs_nTracks

      * nsig_res: 
          Empirical value representing the number of sigma wrt to the std resolution 
          as a function of true_PVs_nTracks - such as 5

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
        ValueSet(S, Sp, MT, FP)

    This algorithm computes the weighted mean, and uses that.
    This avoids small fluctionations in the input array by requiring .
    a minium total integrated value required to "turn it on"
    (integral_threshold=0.2) and min_width of 3 bins wide.
    """

    return ValueSet_res(
        *numba_efficiency_res(
            truth, predict, true_PVs_nTracks, true_PVs_z, nsig_res, min_res, threshold, integral_threshold, min_width, debug
        )
    )
#####################################################################################


'''
#####################################################################################
def exact_efficiency_res(
    truth, outputs, difference, threshold, integral_threshold, min_width
):
    """
    Compute the exact efficency, as well as return successful and failed counters. Rather slow.

    Accepts:
      * truth: AkwardArray of exact truth values
      * predict: Numpy array of predictions
      * difference: The maximum difference to count a success, in bin widths - such as 5
      * threshold: The threshold for considering an "on" value - such as 1e-2
      * integral_threshold: The total integral required to trigger a hit - such as 0.2
      * min_width: The minimum width (in bins) of a feature - such as 2

    Returns: total_found, pvs_successful, pvs_failed
    """
    total_found = 0

    pvs_successful = Counter()
    pvs_failed = Counter()

    for i in range(len(outputs)):
        found_values = ( ( pv_locations_res(outputs[i], threshold, integral_threshold, min_width) / 10 ) - 100)

        for z, n in zip(truth.z[i], truth.n[i]):
            if len(found_values) == 0:
                continue
            closest = np.min(np.abs(z - found_values))
            found = closest < difference / 10

            # Require 5 or more tracks
            if n > 4:
                total_found += found

            if found:
                pvs_successful[n] += 1
            else:
                pvs_failed[n] += 1

    return total_found, pvs_successful, pvs_failed
'''