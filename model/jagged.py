import awkward
import numpy as np

def concatenate(jaggedarrays):
    '''
    Concatenate jagged arrays. Does not support alternate `axis` or `out=`.
    '''
    # Support generators:
    jaggedarrays = list(jaggedarrays)
    
    contents = np.concatenate([j.flatten() for j in jaggedarrays])
    counts = np.concatenate([j.counts for j in jaggedarrays])
    return awkward.JaggedArray.fromcounts(counts, contents)
