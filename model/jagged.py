try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward
import numpy as np


def concatenate(jaggedarrays):
    """
    Concatenate jagged arrays. Does not support alternate `axis` or `out=`. Requires 1 or more jaggedarrays.
    """

    # Support generators:
    jaggedarrays = list(jaggedarrays)

    # Propogate Awkward 0.8+ jagged array types
    first = jaggedarrays[0]
    JaggedArray = getattr(first, "JaggedArray", awkward.JaggedArray)

    # Perform the concatenation
    contents = np.concatenate([j.flatten() for j in jaggedarrays])
    counts = np.concatenate([j.counts for j in jaggedarrays])
    return JaggedArray.fromcounts(counts, contents)
