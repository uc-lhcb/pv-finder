import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import torch
import pandas as pd

from model.collectdata_kde_Ellipsoids import collect_t2kde_data
from model.collectdata_mdsA  import collect_truth
from model.models_kde import TracksToKDE_Ellipsoids_DDplus as Model

"""
making a function to take in a DDplus model (or I suppose any KDE to hist model) and output
some example hists centered on events, in order to see what the model is learning. Based on 
code from Plot_TrainedModel_Tracks_to_KDE in pv-finder/notebooks (iirc)

Best used in a notebook by running

->  data = load_data()
    
in one cell, and then in the next cell running 

->  plot_DDplus_hists( [list of state dicts], data)
    
in case you want to re-plot without reloading the data.

"""

def load_data(device='cpu'):  
    #val dataset
    batch_size = 16
    val_loader = collect_t2kde_data('dataAA/20K_POCA_kernel_evts_200926.h5',
                                    batch_size=batch_size,
                                    device=device,
                                    slice = slice(0,20))

    PV = collect_truth('dataAA/20K_POCA_kernel_evts_200926.h5', pvs=True)

    SV = collect_truth('dataAA/20K_POCA_kernel_evts_200926.h5', pvs=False)
    return (val_loader, PV, SV)

def plot_DDplus_hists(model_dicts, data, num_events=5, device='cpu'):

    val_loader = data[0]
    PV = data[1]
    SV = data[2]
    
    nOut1 = 50
    nOut2 = 50
    nOut3 = 50
    nOut4 = 50
    nOut5 = 50
    nOut6 = 50
    nOut7 = 50
    nOut8 = 50
    nOut9 = 50
    nOut10 = 50
    nOut11 = 50
    latentChannels = 4
                      
    outputs = {}
    y_pred = {}
    models = {}
    for ep, state_dict in model_dicts:
        model = Model(nOut1,nOut2,nOut3,nOut4,nOut5,nOut6,nOut7,nOut8,nOut9,nOut10,nOut11,latentChannels)
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models[ep] = model
        
    epochs = [tup[0] for tup in model_dicts]
    print(epochs)
        
    nFeatures = 4000

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=min(epochs), vmax=max(epochs))
                      
    with torch.no_grad():
        count = 0
        for inputs, labels in val_loader:
            #print("count",count)
            if count>0:
                continue
            count+=1
            if inputs.device != device:
                inputs, labels = inputs.to(device), labels.to(device)

            # Loop over the trained iterations to retrieve the predictions 
            for ep in epochs:
                outputs[ep] = models[ep](inputs)
                #print(outputs[it])

            # Get the number of events from the output of the first trained model 
            nEvts = outputs[0].shape[0]
            
            print('labels shape:', labels.size())
            
            # Do something on the true distribution (KDEs) 
            y = labels.view(nEvts,-1,nFeatures)
            
            print('nEvts:', nEvts, '| nFeatures:', nFeatures)
            print('y shape:', y.size())
            
            y = y.transpose(1,2)         
            
            print('y shape:', y.size())
            
#             for j in range(nEvts):
#                 plt.figure(figsize=(12,8))
#                 for i in range(3):
#                     plt.plot(y[j,:,i], label='%i'%i)
#                 plt.legend()

            # LOOP over the events
            for iEvt in range(min([nEvts, num_events])):

                # --- Retrieve the TRUE KDEs
                y_kde = y[iEvt,:,0].cpu().numpy()
                # print("y_kde.shape = ",y_kde.shape)

                # --- Retrieve the PREDICTED KDEs for each trained model (iteration over same model)
                for ep in epochs:
                    #print(it)
                    y_pred[ep] = outputs[ep][iEvt,:].cpu().numpy()
                    #print(y_pred)
                    #y_pred[it] = y_pred[it].cpu().numpy()
                
#                 if (iEvt<5):
                plt.figure(figsize=(15, 10))
                plt.title('Event %i'%(iEvt))
                #plt.ylim(0.0005,1.1*max(y_pred[ep]))
                #print(max(y_pred[ep]))
                plt.plot(y_kde, color="royalblue", label="True KDE")
                for ep in epochs:
                    #print(it)
                    plt.plot(y_pred[ep], color=cm.jet(( ep-min(epochs) )/( max(epochs)-min(epochs) )), alpha = 0.8, label='Epoch'+str(ep))
                #plt.legend()
                textstr = '\n'.join((
                r'Evt #%s' % (iEvt, ),     
                ))              
                props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.5) #
                #ax.text(0.05, 0.95, textstr, transform=ax.transAxes,verticalalignment='top', bbox=props,linespacing = 1.8)#, fontsize=14
                plt.xlabel("z (in bin #)")
                plt.ylabel("Arbitrary unit")
                plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet))
                
                plt.figure(figsize=(15, 10))
                plt.title('Event %i zoomed in'%(iEvt))
                #plt.ylim(0.0005,1.1*max(y_pred[ep]))
                #print(max(y_pred[ep]))
                plt.plot(y_kde, color="royalblue", label="True KDE")
                xmax = np.argmax(y_kde)
                plt.xlim((xmax-250, xmax+250))
                for ep in epochs:
                    #print(it)
                    plt.plot(y_pred[ep], color=cm.jet(( ep-min(epochs) )/( max(epochs)-min(epochs) )), alpha = 0.8, label='Epoch'+str(ep))
                #plt.legend()
                textstr = '\n'.join((
                r'Evt #%s' % (iEvt, ),     
                ))              
                props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.5) #
                #ax.text(0.05, 0.95, textstr, transform=ax.transAxes,verticalalignment='top', bbox=props,linespacing = 1.8)#, fontsize=14
                plt.xlabel("z (in bin #)")
                plt.ylabel("Arbitrary unit")
                plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet))
                #plt.show()
