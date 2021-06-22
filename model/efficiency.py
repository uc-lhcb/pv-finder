import numba
import numpy as np
from typing import NamedTuple
from collections import Counter


class ValueSet(NamedTuple):
    """
    Class to represent a results tuple. Adding and printing supported,
    along with a few convenience properties.
    """

    S: int
    Sp: int
    MT: int
    FP: int
    events: int = 1

    @property
    def real_pvs(self):
        return self.S + self.MT

    @property
    def eff_rate(self):
        if self.real_pvs == 0:
            return 0
        return self.S / self.real_pvs

    @property
    def fp_rate(self):
        return self.FP / self.events

    def __repr__(self):
        message = f"Found {self.S} of {self.real_pvs}, added {self.FP} (eff {self.eff_rate:.2%})"
        if self.events > 1:
            message += f" ({self.fp_rate:.3} FP/event)"
        # if self.S != self.Sp:
        #    message += f" ({self.S} != {self.Sp})"
        return message

    def __add__(self, other):
        return self.__class__(
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
            self[3] + other[3],
            self[4] + other[4],
        )

    def pretty(self):
        if self.S != self.Sp:
            s_message = f"Successes: Either {self.S:,} or {self.Sp:,}, depending on how you count."
        else:
            s_message = f"Successes: {self.S:,}"

        return f"""\
Real PVs in validation set: {self.real_pvs:,}
{s_message}
Missed true PVs: {self.MT:,}
False positives: {self.FP:,}
Efficiency of detecting real PVs: {self.eff_rate:.2%}
False positive rate: {self.fp_rate:.3}"""


@numba.jit(
    numba.float32[:](numba.float32[:], numba.float32, numba.float32, numba.int32),
    locals={"integral": numba.float32, "sum_weights_locs": numba.float32},
    nopython=True,
)
def pv_locations(targets, threshold, integral_threshold, min_width):
    state = 0
    integral = 0.0
    sum_weights_locs = 0.0

    # Make an empty array and manually track the size (faster than python array)
    items = np.empty(150, np.float32)
    nitems = 0

    for i in range(len(targets)):
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_weights_locs += i * targets[i]  # weight times location

        if (targets[i] < threshold or i == len(targets) - 1) and state > 0:

            # Record only if
            if state >= min_width and integral >= integral_threshold:
                items[nitems] = sum_weights_locs / integral
                nitems += 1

            # reset state
            state = 0
            integral = 0.0
            sum_weights_locs = 0.0

    # Special case for final item (very rare or never occuring)
    # handled by above if len

    return items[:nitems]


@numba.jit(numba.float32[:](numba.float32[:], numba.float32[:]), nopython=True)
def filter_nans(items, mask):
    retval = np.empty_like(items)
    max_index = 0
    for item in items:
        index = int(round(item))
        not_valid = np.isnan(mask[index])
        if not not_valid:
            retval[max_index] = item
            max_index += 1

    return retval[:max_index]


@numba.jit(
    numba.types.UniTuple(numba.int32, 2)(
        numba.float32[:], numba.float32[:], numba.float32
    ),
    nopython=True,
)
def compare(a, b, diff):
    succeed = 0
    fail = 0

    if len(b) == 0:
        return 0, len(a)

    # Check for closest value
    for item in a:
        mindiff = np.abs(b - item).min()
        if mindiff > diff:
            fail += 1
        else:
            succeed += 1

    return succeed, fail


# This function does the calculation (the function after this is
# just a wrapper to produce pretty output)


@numba.jit(
    numba.types.UniTuple(numba.int32, 4)(
        numba.float32[:],
        numba.float32[:],
        numba.float32,
        numba.float32,
        numba.float32,
        numba.int32,
    ),
    nopython=True,
)
def numba_efficiency(
    truth, predict, difference, threshold, integral_threshold, min_width
):

    true_values = pv_locations(truth, threshold, integral_threshold, min_width)
    predict_values = pv_locations(predict, threshold, integral_threshold, min_width)

    filtered_predict_values = filter_nans(predict_values, truth)

    # Using the unfiltered here intentionally - might not make a difference
    S, MT = compare(true_values, predict_values, difference)

    Sp, FP = compare(filtered_predict_values, true_values, difference)

    return S, Sp, MT, FP


def efficiency(truth, predict, difference, threshold, integral_threshold, min_width):
    """
    Compute three values: The number of succeses (S), the number of missed true
    values (MT), and the number of missed false values (FP). Note that the number of successes
    is computed twice, and both values are returned.

    Accepts:
      * truth: Numpy array of truth values
      * predict: Numpy array of predictions
      * difference: The maximum difference to count a success, in bin widths - such as 5
      * threshold: The threshold for considering an "on" value - such as 1e-2
      * integral_threshold: The total integral required to trigger a hit - such as 0.2
      * min_width: The minimum width (in bins) of a feature - such as 2


    Returns: ValueSet(S, Sp, MT, FP)

    This algorithm computes the weighted mean, and uses that.
    This avoids small fluctionations in the input array by requiring .
    a minium total integrated value required to "turn it on"
    (integral_threshold=0.2) and min_width of 3 bins wide.
    """

    return ValueSet(
        *numba_efficiency(
            truth, predict, difference, threshold, integral_threshold, min_width
        )
    )


def exact_efficiency(
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
        found_values = (
            pv_locations(outputs[i], threshold, integral_threshold, min_width) / 10
            - 100
        )

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
