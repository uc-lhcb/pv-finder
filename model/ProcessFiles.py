#!/usr/bin/env python3

import numpy as np
from origdata import process_root_file
from utilities import Timer
import argparse
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

def main(output, files):
    Xs = []
    Ys = []

    futures = []
    
    for f in files:
        assert f.exists(), f'{f} must be an existing file'

    with ProcessPoolExecutor(max_workers=min(len(files), 24)) as executor:
        for i, f in enumerate(files):
            futures.append(executor.submit(process_root_file, f, notebook=False, position=i))

    for future in futures:
        X, Y = future.result()
        Xs.append(X)
        Ys.append(Y)
        
    print()
    
    with Timer(start="Concatinating..."):
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys, 1)
    
    with Timer(start=f"Saving to {output}..."):
        np.savez_compressed(output, kernel=X, pv=Y[0], sv=Y[2], pv_other=Y[1], sv_other=Y[3])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "This processes files, in multiple processes. "
                                     "You should use it something like this: "
                                     "./ProcessFiles -o /data/schreihf/PvFinder/Aug_10_30000.npz "
                                     "/data/schreihf/PvFinder/kernel*")
    
    parser.add_argument('-o', '--output', required=True, help="Set the output file (.npz)")
    parser.add_argument('files', type=Path, nargs='+', help="The files to read in")
    
    args = parser.parse_args()
    
    main(args.output, args.files)