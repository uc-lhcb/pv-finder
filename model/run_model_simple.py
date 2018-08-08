#!/usr/bin/env python3
# coding: utf-8

# Please see RunModel.ipynb in notebooks for more descriptions.

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = "18"
plt.rcParams["axes.labelweight"] = "bold"

import os
import numpy as np
import torch
from pathlib import Path

# Model parameters
output = Path('output')
name = '20180808_2Layer_75000'
data = Path('/data/schreihf/PvFinder/July_31_75000.npz')
n_epochs = 200
batch = 32
learning_rate = 1E-3

# This is in the same directory as the helper files, so no special path
# manipulation is needed
from collectdata import collect_data
from loss import Loss
from training import trainNet
from models import ModelCNN2Layer as Model

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 0 is P100 on Goofy

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset_train, dataset_val, _ = collect_data(
    data, 55_000, 10_000,
    device=device, verbose=True)

model = Model()
# Copy model weights to device
if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(device)

output.mkdir(exist_ok=True)


# Run the epochs
for results in trainNet(model, dataset_train, dataset_val,
                        Loss(), batch, range(n_epochs),
                        learning_rate=learning_rate,
                        verbose = True):

    # Save each model state dictionary
    torch.save(model.state_dict(), output / f'{name}_{results.epoch}.pyt')

torch.save(model.state_dict(), output / f'{name}_final.pyt')

fig=plt.figure()
fig.set_figheight(10)
fig.set_figwidth(15)
plt.plot(np.arange(len(results.cost))+1, results.cost, 'o-',color='r',label='Train')
plt.plot(np.arange(len(results.val))+1, results.val, 'o-' , color='b', label='Validation')
plt.xlabel('Number of epoch', weight='bold', size= 20)
plt.ylabel('Average cost per bin of a batch',  weight='bold', size= 20)
plt.yscale('log')
plt.tick_params('y', colors = 'k',labelsize=16 )
plt.tick_params('x', colors = 'k',labelsize=16 )
plt.legend()
fig.savefig(output / f'{name}.png')


