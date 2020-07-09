#!/usr/bin/env python3
# coding: utf-8

# Please see RunModel.ipynb in notebooks for more descriptions.

# Get the current script and currrent working directory
from pathlib import Path

DIR = Path(__file__).parent.resolve()
CURDIR = Path(".").resolve()

# Plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(str(DIR / "pvfinder.mplstyle"))
# See https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html

import os
import numpy as np
import torch

# Model parameters
output = CURDIR / "output"  # output folder
name = "20180816_2Layer_120000"  # output name
trainfile = Path("/share/lazy/schreihf/PvFinder/Aug_14_80K.h5")
valfile = Path("/share/lazy/schreihf/PvFinder/Oct03_20K_val.h5")
n_epochs = 200
batch_size = 32
learning_rate = 1e-3

# This is in the same directory as the helper files, so no special path
# manipulation is needed
from model.collectdata import collect_data
from model.loss import Loss
from model.training import trainNet, select_gpu
from model.models import SimpleCNN2Layer as Model
from model.plots import dual_train_plots

results = pd.DataFrame([], columns=Results._fields)

# Device configuration
device = select_gpu()  # You can set a GPU number here or in CUDA_VISIBLE_DEVICES

train_loader = collect_data(
    trainfile, batch_size=batch_size, device=device, shuffle=True, masking=True
)
val_loader = collect_data(
    valfile, batch_size=batch_size, device=device, shuffle=False, masking=True
)

model = Model()
loss = Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Copy model weights to device
if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# Make the output directory if it does not exist
output.mkdir(exist_ok=True)

# Run the epochs
for result in trainNet(
    model, optimizer, loss, train_loader, val_loader, n_epochs, notebook=False
):

    results = results.append(pd.Series(result._asdict()), ignore_index=True)

    # Save each model state dictionary
    torch.save(model.state_dict(), output / f"{name}_{result.epoch}.pyt")

torch.save(model.state_dict(), output / f"{name}_final.pyt")

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
fig.savefig(str(output / f"{name}_stats_a.png"))
