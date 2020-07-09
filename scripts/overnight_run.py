import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import mlflow

from model.collectdata_mdsA import collect_data
from model.alt_loss_A import Loss
from model.training import trainNet, select_gpu
from model.model_29June2020_B import UNet4SC as Model
from model.training import trainNet, select_gpu, Results
from model.utilities import load_full_state, count_parameters
from model.plots import dual_train_plots, replace_in_ax
from model.utilities import count_parameters, Params
import hiddenlayer as HL
from torchsummary import summary

device = torch.device('cuda:2')    
mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')
mlflow.set_experiment('Weird U-Net')

train_loader = collect_data('/share/lazy/sokoloff/ML-data_A/Aug14_80K_train.h5',
                            '/share/lazy/sokoloff/ML-data_AA/Oct03_80K_train.h5',
                           '/share/lazy/sokoloff/ML-data_AA/Oct03_40K_train.h5',
                            batch_size=64,
                            masking=True, shuffle=True,
                            load_XandXsq=False,
                            load_xy=False)

val_loader = collect_data('/share/lazy/sokoloff/ML-data_AA/Oct03_20K_val.h5',
                          batch_size=64,
                          slice=slice(256 * 39),
                          masking=True, shuffle=False,
                          load_XandXsq=False,
                          load_xy=False)

# params order - batch size, epochs, lr
runs = [
    #(Model(24).to(device), Params(64, 200, 5e-4, 0)),
    #(Model(16).to(device), Params(64, 200, 5e-4, 0)),
    (Model(12).to(device), Params(64, 200, 5e-4, 0)),
    (Model(24).to(device), Params(64, 200, 5e-3, 0)),
    (Model(16).to(device), Params(64, 200, 5e-3, 0)),
    (Model(12).to(device), Params(64, 200, 5e-3, 0)),
    (Model(24).to(device), Params(64, 200, 1e-3, 0)),
    (Model(16).to(device), Params(64, 200, 1e-3, 0)),
    (Model(12).to(device), Params(64, 200, 1e-3, 0))
]

# we need this for plots
ax, tax, lax, lines = dual_train_plots()
fig = ax.figure
plt.tight_layout()
results = pd.DataFrame([], columns=Results._fields)

# Define optimizer and loss
loss = Loss(epsilon=1e-5,coefficient=2.5)
eff_avg = 0
fp_avg = 0

for (model, args) in runs:
    run_name = 'No MaxPool, kernel size decay'

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run(run_name = run_name) as run:

        for key, value in vars(args).items():
            print(key, value)
            mlflow.log_param(key, value)

        mlflow.log_param('Parameters', count_parameters(model))
        

        for result in trainNet(model, optimizer, loss,
                                train_loader, val_loader,
                                args.epochs+args.epoch_start, epoch_start=args.epoch_start, notebook=False, device=device):

            result = result._asdict()
            
            # plotting code block ===============================
            results = results.append(pd.Series(result), ignore_index=True)
            xs = results.index
            # Update the plot above
            lines['train'].set_data(results.index,results.cost)
            lines['val'].set_data(results.index,results.val)
            #filter first cost epoch (can be really large)
            max_cost = max(max(results.cost if len(results.cost)<2 else results.cost[1:]), max(results.val))
            
            min_cost = min(min(results.cost), min(results.val))
            # The plot limits need updating too
            ax.set_ylim(min_cost*.9, max_cost*1.1)  
            ax.set_xlim(-.5, len(results.cost) - .5)
            replace_in_ax(lax, lines['eff'], xs, results['eff_val'].apply(lambda x: x.eff_rate))
            replace_in_ax(tax, lines['fp'], xs, results['eff_val'].apply(lambda x: x.fp_rate))
            # Redraw the figure
#            fig.canvas.draw()
            fig.savefig('plot.png')
            # plotting code block =============================== 

            # Log metrics
            mlflow.log_metric('Efficiency', result['eff_val'].eff_rate, result['epoch'])
            mlflow.log_metric('False Positive Rate',  result['eff_val'].fp_rate, result['epoch'])
            mlflow.log_metric('Validation Loss',  result['val']*2, result['epoch'])
            mlflow.log_metric('Training Loss',  result['cost']*2, result['epoch'])
            
            # Log tags
            mlflow.set_tag('Optimizer', 'Adam')
            mlflow.set_tag('Kernel size', 'Mixed')
            mlflow.set_tag('Skip connections', '4')
            mlflow.set_tag('Activation', 'Softplus')
            mlflow.set_tag('Mid Activation', 'Relu')

            # Save model AND optimizer state_dict AND epoch number. x
            torch.save({
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':args.epochs+result['epoch']
                }, 'run_stats.pyt')
            mlflow.log_artifact('run_stats.pyt')
            
            # save a diagram of the architecture
            HL.transforms.Fold("Conv > BatchNorm > LeakyRelu", "ConvBnRelu"),
            HL.build_graph(model, torch.zeros([args.batch_size, 1, 4000]).to(device)).save('architecture', format='png')
            mlflow.log_artifact('architecture.png')
        
            # log the code for the model architecture
            mlflow.log_artifact('architecture.txt')
        
            # save plot that mike likes
            mlflow.log_artifact('plot.png')