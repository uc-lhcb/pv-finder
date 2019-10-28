#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path

from model import models
from model.efficiency import efficiency, ValueSet
from model.collectdata import collect_data
from model.training import select_gpu, PARAM_EFF

# This bit of black magic pulls out all the Torch Models by name from the loaded models file.
MODELS = {
    x
    for x in dir(models)
    if not x.startswith("_")
    and isinstance(getattr(models, x), type)
    and torch.nn.Module in getattr(models, x).mro()
}


def main(model, dataset):
    model.eval()

    with torch.no_grad():
        outputs = model(dataset.tensors[0]).cpu().numpy()
        labels = dataset.tensors[1].cpu().numpy()

    total = ValueSet(0, 0, 0, 0)

    for label, output in zip(labels, outputs):
        total += efficiency(label, output, **PARAM_EFF)

    return total


if __name__ == "__main__":
    argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run example: ./RunEfficiency.py --model SimpleCNN3Layer "
        "output3/20180815_120000_3layer_{1,2,3,4,5,6}.pyt --gpu 1 --training --events=30000",
    )

    parser.add_argument(
        "model_dict", nargs="+", type=Path, help="The model dictionary(s)"
    )
    parser.add_argument(
        "-d",
        "--data",
        default="/data/schreihf/PvFinder/Aug_15_140000.npz",
        help="The data to read in, in npz format (for now)",
    )
    parser.add_argument(
        "--model", required=True, choices=MODELS, help="Model to train on"
    )
    parser.add_argument(
        "--gpu",
        help="Pick a GPU by bus order (you can use CUDA_VISIBLE_DEVICES instead)",
    )
    parser.add_argument(
        "--events", type=int, help="Limit the maximum number of events to read"
    )

    args = parser.parse_args()

    device = select_gpu(args.gpu)

    Model = getattr(models, args.model)
    model = Model().to(device)

    dataloader = collect_data(
        args.data, batchsize=1, slice=slice(args.events), device=device
    )

    for item in args.model_dict:
        print()
        print(item.stem)
        model.load_state_dict(torch.load(item))

        print(main(model, dataloader.dataset).pretty())
