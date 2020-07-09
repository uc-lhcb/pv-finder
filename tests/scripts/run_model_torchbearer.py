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
trainfile = Path("/share/lazy/schreihf/PvFinder/Aug_14_80K.npz")
valfile = Path("/share/lazy/schreihf/PvFinder/Oct03_20K_val.npz")
n_epochs = 200
batch_size = 32
learning_rate = 1e-3

# This is in the same directory as the helper files, so no special path
# manipulation is needed
from model.collectdata import collect_data
from model.loss import Loss
from model.models import SimpleCNN2Layer as OurModel
from model.training import select_gpu
import torchbearer
from torchbearer.callbacks import TensorBoard

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 is P100 on Goofy

# Device configuration
device = select_gpu()

trainfile = Path("/share/lazy/schreihf/PvFinder/Aug_14_80K.npz")
valfile = Path("/share/lazy/schreihf/PvFinder/Oct03_20K_val.npz")

train_loader = collect_data(
    trainfile, batch_size=batch_size, device=device, shuffle=True, masking=True
)
val_loader = collect_data(
    valfile, batch_size=batch_size, device=device, shuffle=False, masking=True
)

model = OurModel()
loss = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Make the output directory if it does not exist
output.mkdir(exist_ok=True)

# Make a TorchBearer model
torchbearer_model = torchbearer.Model(model, optimizer, loss, metrics=("loss",)).to(
    device
)
torchbearer_model.fit_generator(
    train_loader,
    epochs=n_epochs,
    validation_generator=val_loader,
    callbacks=[
        TensorBoard(
            write_graph=True, write_batch_metrics=True, write_epoch_metrics=True
        )
    ],
)
