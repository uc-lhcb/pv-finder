#!/usr/bin/env python3
# coding: utf-8

# Get the current script directory
from pathlib import Path

DIR = Path(__file__).parent.resolve()

# Plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(str(DIR / "pvfinder.mplstyle"))
# See https://matplotlib.org/users/style_sheets.html


import numpy as np
import argparse
import torch
import os


from model.collectdata import collect_data
from model.loss import Loss
from model.training import trainNet, select_gpu
from model import models
from model.plots import dual_train_plots

# This bit of black magic pulls out all the Torch Models by name from the loaded models file.
MODELS = {
    x
    for x in dir(models)
    if not x.startswith("_")
    and isinstance(getattr(models, x), type)
    and torch.nn.Module in getattr(models, x).mro()
}


def main(
    n_epochs,
    name,
    trainfile,
    valfile,
    batch_size,
    learning_rate,
    model,
    output,
    gpu=None,
):

    results = pd.DataFrame([], columns=Results._fields)

    device = select_gpu(gpu)

    Model = getattr(models, model)

    train_loader = collect_data(
        trainfile, batch_size=batch_size, device=device, shuffle=True, masking=True
    )
    val_loader = collect_data(
        valfile, batch_size=batch_size, device=device, shuffle=False, masking=True
    )

    model = Model()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    output.mkdir(exist_ok=True)

    # Create our optimizer function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create our loss function
    loss = Loss()

    # Run the epochs
    for result in trainNet(
        model, optimizer, loss, train_loader, val_loader, n_epochs, notebook=False
    ):

        results = results.append(pd.Series(result._asdict()), ignore_index=True)

        # Any prints in here are per iteration (use tqdm.write)

        # Save each model state dictionary
        torch.save(model.state_dict(), output / f"{name}_{result.epoch}.pyt")

    # Save final model and stats
    torch.save(model.state_dict(), output / f"{name}_final.pyt")
    np.savez(output / f"{name}_stats.npz")

    # Make a plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    dual_train_plots(
        results.index,
        results.cost,
        results.val,
        results["eff_val"].apply(lambda x: x.eff_rate),
        results["eff_val"].apply(lambda x: x.fp_rate),
        axs=axs,
    )
    plt.tight_layout()

    # Save the plot
    fig.savefig(str(output / f"{name}.png"))


if __name__ == "__main__":
    # Handy: https://gist.github.com/dsc/3855240

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run example: ./RunModel.py 20180801_30000_2layer --model SimpleCNN2Layer --gpu 0",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=200, help="Set the number of epochs to run"
    )
    parser.add_argument(
        "name", help="The name, such as date_numevents_model or similar"
    )
    parser.add_argument(
        "-t",
        "--train",
        default="/share/lazy/schreihf/PvFinder/Aug14_80K_train.h5",
        help="The training data to read in, in npz format",
    )
    parser.add_argument(
        "-v",
        "--val",
        default="/share/lazy/schreihf/PvFinder/Oct03_20K_val.npz",
        help="The validation data to read in, in h5 format",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32, dest="batch", help="The batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning",
        help="The learning rate",
    )
    parser.add_argument(
        "--model", required=True, choices=MODELS, help="Model to train on"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output"), help="Output directory"
    )
    parser.add_argument(
        "--gpu",
        help="Pick a GPU by bus order (you can use CUDA_VISIBLE_DEVICES instead)",
    )

    args = parser.parse_args()
    main(
        args.epochs,
        args.name,
        args.train,
        args.val,
        args.batch,
        args.learning,
        args.model,
        args.output,
        args.gpu,
    )
