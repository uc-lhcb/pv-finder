#!/usr/bin/env python3
# coding: utf-8

# Get the current script directory
from pathlib import Path
DIR = Path(__file__).parent.resolve()

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(str(DIR / 'pvfinder.mplstyle'))
# See https://matplotlib.org/users/style_sheets.html


import numpy as np
import argparse
import torch
import os


from collectdata import collect_data
from loss import Loss
from training import trainNet
import models

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

    output.mkdir(exist_ok=True)
    
    # Run the epochs
    for results in trainNet(model, dataset_train, dataset_val,
                            Loss(), batch, n_epochs,
                            notebook = False,
                            learning_rate=learning_rate):
        
        # Any prints in here are per iteration
        
        # Save each model state dictionary
        if output:
            torch.save(model.state_dict(), output / f'{name}_{results.epoch}.pyt')

    torch.save(model.state_dict(), output / f'{name}_final.pyt')

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(np.arange(len(results.cost))+1, results.cost, 'o-',color='r',label='Train')
    ax.plot(np.arange(len(results.val))+1, results.val, 'o-' , color='b', label='Validation')
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Average cost per bin of a batch')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(str(output / f'{name}.png'))

    
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
    parser.add_argument('--output', type=Path, default=Path('output'), help="Output directory")
    parser.add_argument('--copy-each-time', dest='copyeach', action='store_true',
                        help='Copy the memory to the GPU each time')
    
    args = parser.parse_args()
    main(args.epochs, args.name, args.data, args.batch, args.learning, args.model, args.output, args.copyeach)