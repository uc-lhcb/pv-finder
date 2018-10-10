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
name = '20180816_2Layer_120000' # output name
trainfile = Path('/share/lazy/schreihf/PvFinder/Aug_14_80K.npz')
valfile = Path('/share/lazy/schreihf/PvFinder/Oct03_20K_val.npz')
n_epochs = 200
batch_size = 32
learning_rate = 1e-3

# This is in the same directory as the helper files, so no special path
# manipulation is needed
from collectdata import collect_data
from loss import Loss
from training import trainNet, select_gpu
from models import SimpleCNN2Layer as Model

# Device configuration
device = select_gpu() # You can set a GPU number here or in CUDA_VISIBLE_DEVICES

train_loader = collect_data(trainfile, batch_size=batch_size, device=device, shuffle=True, masking=True)
val_loader = collect_data(valfile, batch_size=batch_size, device=device, shuffle=False, masking=True)

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
for results in trainNet(model, optimizer, loss,
                        train_loader, val_loader,
                        n_epochs,
                        notebook = False):

    # Save each model state dictionary
    torch.save(model.state_dict(), output / f'{name}_{results.epoch}.pyt')

torch.save(model.state_dict(), output / f'{name}_final.pyt')

fig=plt.figure()
plt.plot(np.arange(len(results.cost))+1, results.cost, 'o-',label='Train')
plt.plot(np.arange(len(results.val))+1, results.val, 'o-' , label='Validation')
plt.xlabel('Number of epoch')
plt.ylabel('Average cost per bin of a batch')
plt.yscale('log')
plt.legend()
fig.savefig(output / f'{name}.png')


