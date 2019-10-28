#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import warnings

# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

from model.origdata import (
    process_root_file,
    OutputData,
    concatinate_data,
    save_data_hdf5,
)

from model.utilities import Timer


def main(output_fname, files):
    outputs = []
    futures = []

    for f in files:
        assert f.exists(), f"{f} must be an existing file"

    outputs = [process_root_file(f) for f in files]

    # Convert list of OutputDatas to one OutData
    with Timer(start="Concatinating..."):
        outputs = concatinate_data(outputs)

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
