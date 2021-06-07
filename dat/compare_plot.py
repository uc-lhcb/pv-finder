#!/usr/bin/env python3

from argparse import ArgumentParser
try:
    import uproot3 as uproot
except ModuleNotFoundError:
    import uproot
import numpy as np
import sys
from pathlib import Path
import numpy as np
import matplotlib

DIR = Path(__file__).resolve().parent


def compare(p: Path, orig: Path, key: str):
    f_new = uproot.open(p)["kernel"]
    f_old = uproot.open(orig)["kernel"]

    d_new = f_new[key].array()
    d_old = f_old[key].array()

    return plot(d_new, d_old, 2000)


def plot(n: np.ndarray, o: np.ndarray, v: int):
    fig, axs = plt.subplots(2, 1, figsize=(15, 2), sharex=True)
    axs[0].imshow(n[:, v : v + 200])
    axs[1].imshow(o[:, v : v + 200])
    return fig, axs


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compare a file with the original kernel (10 event run)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DIR / "kernel_10pvs.root",
        help="A ROOT file with a new run",
    )
    parser.add_argument(
        "--orig",
        type=Path,
        default=DIR / "result_10pvs.root",
        help="A ROOT file with the old run",
    )
    parser.add_argument(
        "--save", type=Path, help="A file to write to instead of a display"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="zdata",
        help="The key to plot, one of zdata, xmax, ymax",
    )
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, _ = compare(args.file, args.orig, args.key)

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()
