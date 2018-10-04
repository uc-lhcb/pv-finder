#!/usr/bin/env python3

import numpy as np
from origdata import process_root_file
from utilities import Timer
import argparse
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

def main(output, files):
    Xs = []
    Xmaxs = []
    Ymaxs = []
    Ys = []

    futures = []

    for f in files:
        assert f.exists(), f'{f} must be an existing file'

    with ProcessPoolExecutor(max_workers=min(len(files), 24)) as executor:
        for i, f in enumerate(files):
            futures.append(executor.submit(process_root_file, f, notebook=False, position=i))

    for future in futures:
        X, Y, Xmax, Ymax = future.result()
        Xs.append(X)
        Ys.append(Y)
        Xmaxs.append(Xmax)
        Ymaxs.append(Ymax)

    print()

    with Timer(start="Concatinating..."):
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys, 1)
        Xmax = np.concatenate(Xmaxs)
        Ymax = np.concatenate(Xmaxs)

    with Timer(start=f"Saving to {output}..."):
        if output.suffix == '.npz':
            np.savez_compressed(output, Xmax=Xmax, Ymax=Ymax, kernel=X, pv=Y[0], sv=Y[2], pv_other=Y[1], sv_other=Y[3])
        else:
            import h5py
            with h5py.File(str(output), "w") as hf:
                dset = hf.create_dataset("kernel", data=X, compression="gzip")
                hf.create_dataset("pv", data=Y[0], compression="gzip")
                hf.create_dataset("sv", data=Y[2], compression="gzip")
                hf.create_dataset("pv_other", data=Y[1], compression="gzip")
                hf.create_dataset("sv_other", data=Y[3], compression="gzip")
                hf.create_dataset("Xmax", data=Xmax, compression="gzip")
                hf.create_dataset("Ymax", data=Ymax, compression="gzip")

                dset.attrs["files"] = np.string_(",".join(str(s.stem) for s in (files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "This processes files, in multiple processes. "
                                     "You should use it something like this: "
                                     "./ProcessFiles -o /data/schreihf/PvFinder/Aug_10_30K_train.h5 "
                                     "/data/schreihf/PvFinder/kernel*")

    parser.add_argument('-o', '--output', type=Path, required=True, help="Set the output file (.npz or .h5)")
    parser.add_argument('files', type=Path, nargs='+', help="The files to read in")

    args = parser.parse_args()

    main(args.output, args.files)
