#!/usr/bin/env python3


##  derived from ProcessData_mdsB.py 22 Sept. 2020
##  uses origdata_mds_poca_KDE rather than origdata_mdsA/B as
##  new .root files have old KDEs plus 2 new poca KDEs, one
##  from the sum of probabilities and the other from the
##  sum of the probability square values
##  new KDEs from poca ellipsoids have a smaller dynamic range
##  than the original DKEs
import argparse
from pathlib import Path
import numpy as np
import warnings

# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

from model.origdata_mds_poca_KDE import (
    process_root_file,
    OutputData,
    concatenate_data,
    save_data_hdf5,
)

from model.utilities import Timer


def main(output_fname, files):
    print("output_fname = ",output_fname)
    print("files = ",files)
    outputs = []
    futures = []
    for f in files:
        print("f = ",f)
        assert f.exists(), f"{f} must be an existing file"

    outputs = [process_root_file(f) for f in files]
    print("len(outputs) = ",len(outputs))
    print("len(outputs[0]) = ",len(outputs[0]))

    # Convert list of OutputDatas to one OutData
    with Timer(start="Concatenating..."):
        outputs = concatenate_data(outputs)

    with Timer(start=f"Saving to {output_fname}..."):
        with h5py.File(str(output_fname), "w") as hf:
            save_data_hdf5(hf, outputs, files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This processes files, in multiple processes. "
        "You should use it something like this: "
        "./ProcessFiles -o /data/schreihf/PvFinder/Aug_10_30K_train.h5 "
        "/data/schreihf/PvFinder/kernel*"
    )

    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Set the output file (.h5)"
    )
    parser.add_argument("files", type=Path, nargs="+", help="The files to read in")

    args = parser.parse_args()

    main(args.output, args.files)
