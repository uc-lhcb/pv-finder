import torch
import mlflow

## imports
from model.collectdata_poca_KDE import collect_data_poca
from model.alt_loss_A import Loss
from model.training import trainNet, select_gpu
from model.utilities import Params, save_to_mlflow

## model imports
from model.autoencoder_models import UNet
from model.autoencoder_models import UNetPlusPlus
from model.autoencoder_models import DenseNet
from model.autoencoder_models import PerturbativeUNet

## set arguments for experiment
args = Params(
    batch_size=64,
    device=select_gpu(0),
    epochs=300,
    lr=1e-7,
    experiment_name='June-2022',
    asymmetry_parameter=2.5,
    run_name='unet-train_no-xy-train_2'
)

##  pv_HLT1CPU_D0piMagUp_12Dec.h5 + pv_HLT1CPU_MinBiasMagDown_14Nov.h5 contain 138810 events
##  pv_HLT1CPU_MinBiasMagUp_14Nov.h5 contains 51349
##  choose which to "load" and slices to produce 180K event training sample
##  and 10159 event validation sample
train_loader = collect_data_poca(
                              '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagDown_14Nov.h5',
                              '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_JpsiPhiMagDown_12Dec.h5',
                              '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_D0piMagUp_12Dec.h5',
                              '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagUp_14Nov.h5',
                               slice = slice(None,260000),
                             batch_size=args.batch_size,
## if we are using a larger dataset (240K events, with the datasets above, and 11 GB of GPU memory),
## not the dataset will overflow the GPU memory; device=device will allow the data to move back
## and forth between the CPU and GPU memory. While this allows use of a larger dataset, it slows
## down performance by about 10%.  So comment out when not needed.
##                            device=args.device,
                            masking=True, shuffle=True,
                            load_A_and_B=True,
                            load_xy=True)

# Validation dataset. You can slice to reduce the size.
## dataAA -> /share/lazy/sokoloff/ML-data_AA/
val_loader = collect_data_poca(
##                          '/share/lazy/sokoloff/dataAA/pv_HLT1CPU_MinBiasMagDown_14Nov.h5',
                            '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagUp_14Nov.h5',
##                            '/share/lazy/sokoloff/dataAA/pv_HLT1CPU_D0piMagUp_12Dec.h5',
                          batch_size=args.batch_size,
                          slice=slice(33000,None),
##                          device=args.device,
                          masking=True, shuffle=False,
                          load_A_and_B=True,
                          load_xy=True)

## Set path for mlflow to save experiments
mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')
mlflow.set_experiment(args.experiment_name)

## Choose model of interest. This should be imported from another file or created in this script.
## When used without loading weights, this will randomly initialize the weights (i.e. train from
## scratch)
## Any parameters that need to be passed in should be done here
model = PerturbativeUNet()

## Use when loading pre-trained weights. Find the run_stats.pyt file path and copy here. 
## Comment out if not needed.
model = torch.load('/share/lazy/pv-finder_model_repo/31/2877a40c24ce447a894a4b89b4f2d58b/artifacts/run_stats.pyt')

## Load the opimitizer and loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss = Loss(epsilon=1e-5,coefficient=args.asymmetry_parameter)

## Calculates the number of parameters in the model being trained
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

## Variables for finding average efficiency and false-positive rate over the last N (usually N=10) 
## epochs. Useful if the efficiency/false-positive has large fluctuations from epoch to epoch.
# avgEff = 0.0
# avgFP = 0.0

## Move model to GPU/device
model = model.to(args.device)

## Begin training
train_iter = enumerate(trainNet(model, optimizer, loss, train_loader, val_loader, args.epochs, notebook=True))
with mlflow.start_run(run_name = args.run_name) as run:
    mlflow.log_artifact('script_train_lhcb-mc.py')
    for i, result in train_iter:
        print(result.cost)
        torch.save(model, 'run_stats.pyt')
        mlflow.log_artifact('run_stats.pyt')

        ## Save each epoch's model state dictionary to separate folder
        ## Use to load weights from specific epoch (choose using mlflow)
        output = '/share/lazy/pv-finder_model_repo/ML/' + args.run_name + '_' + str(result.epoch) + '.pyt'
        torch.save(model, output)
        mlflow.log_artifact(output)

        ### Average efficiency and false positive start ###
        ## If we are on the last 10 epochs
        # if(result.epoch >= args.epochs-10):
        #     avgEff += result.eff_val.eff_rate
        #     avgFP += result.eff_val.fp_rate

        # ## If we are on the last epoch
        # if(result.epoch == args.epochs-1):
        #     print('Averaging...\n')
        #     avgEff/=10
        #     avgFP/=10
        #     mlflow.log_metric('10 Eff Avg.', avgEff)
        #     mlflow.log_metric('10 FP Avg.', avgFP)
        #     print('Average Eff: ', avgEff)
        #     print('Average FP Rate: ', avgFP)
        ### Average efficiency and false positive end ###

        ## Save results to mlflow after each epoch
        save_to_mlflow({
            'Metric: Training loss':result.cost,
            'Metric: Validation loss':result.val,
            'Metric: Efficiency':result.eff_val.eff_rate,
            'Metric: False positive rate':result.eff_val.fp_rate,
            'Param: Parameters':parameters,
            'Param: Asymmetry':args.asymmetry_parameter,
            'Param: Batch Size':args.batch_size,
            'Param: Epochs':args.epochs,
            'Param: Learning Rate':args.lr,
        }, step=i)
