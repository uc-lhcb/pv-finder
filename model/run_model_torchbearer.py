#!/usr/bin/env python3
# coding: utf-8

# Please see RunModel.ipynb in notebooks for more descriptions.

# Get the current script and currrent working directory
from pathlib import Path
DIR = Path(__file__).parent.resolve()
CURDIR = Path('.').resolve()

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(str(DIR / 'pvfinder.mplstyle'))
# See https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html

import os
import numpy as np
import torch

# Model parameters
output = CURDIR / 'output' # output folder
name = '20180810_2Layer_30000' # output name
datafile = Path('/data/schreihf/PvFinder/Aug_10_30000.npz')
n_epochs = 200
batch_size = 32
learning_rate = 1e-3

# This is in the same directory as the helper files, so no special path
# manipulation is needed
from collectdata import DataCollector
from loss import Loss
from models import SimpleCNN2Layer as OurModel
import torchbearer
from torchbearer.callbacks import TensorBoard

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 0 is P100 on Goofy

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

collector = DataCollector(datafile, 20_000, 5_000)
train_loader = collector.get_training(batch_size, 20_000, device=device, shuffle=True)
val_loader = collector.get_training(batch_size, 5_000, device=device, shuffle=False)

model = OurModel()
loss = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Make the output directory if it does not exist
output.mkdir(exist_ok=True)

# Make a TorchBearer model
torchbearer_model = torchbearer.Model(model, optimizer, loss, metrics=("loss",)).to(device)
torchbearer_model.fit_generator(train_loader, epochs=n_epochs, validation_generator=val_loader,
                                callbacks=[TensorBoard(write_graph=True, write_batch_metrics=True, write_epoch_metrics=True)])
