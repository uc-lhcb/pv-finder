#!/usr/bin/env python3
# coding: utf-8

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = "18"
plt.rcParams["axes.labelweight"] = "bold"


import numpy as np
import argparse
import torch
import os
from pathlib import Path

from collectdata import collect_data
from loss import Loss
from training import trainNet
import models

from tqdm import tqdm as progress_bar

MODELS = {'SimpleCNN2Layer', 'SimpleCNN3Layer'}

def main(n_epochs, name, data, batch, learning_rate, model, output, copyeach):
    
    if torch.cuda.is_available() and not 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('CUDA_VISIBLE_DEVICES is *required* when running with CUDA available')
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    Model = getattr(models, model)

    dataset_train, dataset_val, _ = collect_data(
        data, 55_000, 10_000,
        device=None if copyeach else device, verbose=True)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed);

    model = Model()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    if output:
        output.mkdir(exist_ok=True)
        
    # Make a progress bar
    progress = progress_bar(range(n_epochs))
    
    # Run the epochs
    for results in trainNet(model, dataset_train, dataset_val,
                            Loss(), batch, progress,
                            learning_rate=learning_rate):
        
        # Pretty print a description
        progress.set_postfix(train=results.cost[-1], val=results.val[-1])
        
        # Save each model state dictionary
        if output:
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

    
if __name__ == '__main__':
    # Handy: https://gist.github.com/dsc/3855240
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Set the number of epochs to run")
    parser.add_argument('name', help="The name, such as date_numevents_model or similar")
    parser.add_argument('-d', '--data', default='/data/schreihf/PvFinder/July_31_75000.npz',
                        help="The data to read in, in npz format (for now)")
    parser.add_argument('-b', '--batch-size', type=int, default=32, dest='batch', help="The batch size")
    parser.add_argument('--learning-rate', type=float, default=1e-3, dest='learning', help="The learning rate")
    parser.add_argument('--model', default='SimpleCNN2Layer', choices=MODELS, help="Model to train on")
    parser.add_argument('--output', type=Path, help="Output directory")
    parser.add_argument('--copy-each-time', dest='copyeach', action='store_true',
                        help='Copy the memory to the GPU each time')
    
    args = parser.parse_args()
    main(args.epochs, args.name, args.data, args.batch, args.learning, args.model, args.output, args.copyeach)