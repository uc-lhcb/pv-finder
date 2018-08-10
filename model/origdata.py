import uproot
import numpy as np
from scipy.stats import norm
from pathlib import Path
import sys

from utilities import import_progress_bar

dtype_X = np.float16
dtype_Y = np.float16

def process_root_file(filepath,
         sd_1 = 0.1,
         *, notebook = None,
            position = None):
    
    name = filepath.stem
    tree = uproot.open(str(filepath))['kernel']
         
    X = (tree['zdata'].array() / 2500.).astype(dtype_X)
    pv_loc = tree['pv_loc'].array()
    pv_cat = tree['pv_cat'].array()
    sv_loc = tree['sv_loc'].array()
    sv_cat = tree['sv_cat'].array()
    
    N_vals = len(X)
    zvals_range = (-99.95, 299.95)
    Y = np.zeros([4, N_vals, 4000], dtype=dtype_Y)
    edges = np.array([-0.05, 0.05])
    bins = np.arange(-3, 4)
    mat = 0.1*bins[np.newaxis,:] + edges[:,np.newaxis] - 99.95

    progress = import_progress_bar(notebook)
    iterator = progress(range(N_vals),
                        desc=name.replace('kernel_',''),
                        dynamic_ncols=True,
                        position=position,
                        mininterval=.2,
                        file=sys.stdout)
    
    if hasattr(iterator, 'write'):
        print = iterator.write
    
    for i in iterator:
        columns = (
            pv_loc[i][pv_cat[i]==1],
            pv_loc[i][pv_cat[i]!=1],
            sv_loc[i][sv_cat[i]==1],
            sv_loc[i][sv_cat[i]!=1]
        )
        for n, centers in enumerate(columns):
            # Centers of PVs
            centers = centers[(zvals_range[0] < centers) & (centers < zvals_range[1])]

            for mean in centers:
                # Compute bin number
                N_bin = int(np.floor((mean - zvals_range[0])*10))
                prob = norm(mean, sd_1)

                values = prob.cdf(N_bin/10 + mat)
                
                try:
                    Y[n, i, bins + N_bin] += values[1] - values[0]
                except IndexError:
                    print(f"{name}: ignoring one hit at bin {N_bin} at {mean} in event {i}, column {n}")
                
    return X, Y