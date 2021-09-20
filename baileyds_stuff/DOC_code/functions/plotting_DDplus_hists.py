import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import torch
import pandas as pd

from model.collectdata_kde_Ellipsoids import collect_t2kde_data
from model.collectdata_mdsA  import collect_truth

from functions.my_collectdata_kde_Ellipsoids import collect_t2kde_data_withtransform

from model.models_kde import TracksToKDE_Ellipsoids_DDplus as Model
"""
making a function to take in a DDplus model (or I suppose any KDE to hist model) and output
some example hists centered on events, in order to see what the model is learning. Based on 
code from Plot_TrainedModel_Tracks_to_KDE in pv-finder/notebooks (iirc)
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

def load_transform_data(device='cpu', noise=False, norm=False, smooth=False):  
    #val dataset
    batch_size = 16
    val_loader = collect_t2kde_data_withtransform('dataAA/20K_POCA_kernel_evts_200926.h5',
                                    batch_size=batch_size,
                                    device=device,
                                    slice = slice(0,20),
                                    noise=noise, norm=norm, smooth=smooth
                                    )

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


def plot_DDplus_hists_zoomedin(model_dicts, data, device='cpu'):
    
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
        
    iters = [tup[0] for tup in model_dicts]
        
    nFeatures = 4000

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=len(model_dicts))
                
    with torch.no_grad():

        #print("val_loader = ",val_loader)
        # Loop over the trained iterations to retrieve the predictions 
        for it in iters:
            outputs[it] = models[it](val_loader.dataset.tensors[0])

        # Get the number of events from the output of the first trained model 
        nEvts = outputs[iters[0]].shape[0]
        #nEvts = 5

        print("nEvts = ",nEvts)
        labels = val_loader.dataset.tensors[1]

        # Do something on the true distribution (KDEs) 
        y = labels.view(nEvts,-1,nFeatures)
        y = y.transpose(1,2)               
        #print("y.shape = ",y.shape)

        # ========================
        # ========================
        # --- LOOP over the events
        # ========================
        # ========================
        maxEvt = 20
        nTracks_good_PV = 5
        plotSV = True
        for iEvt in range(maxEvt):

            print("iEvt = ",iEvt)

            # --- Retrieve the TRUE KDEs
            y_kde = y[iEvt,:,0].cpu().numpy()

            # --- Retrieve the PREDICTED KDEs for each trained model (iteration over same model)
            for it in iters:
                y_pred[it] = outputs[it][iEvt,:]
                y_pred[it] = y_pred[it].cpu().numpy()

            # --- Containers usefull for plotting
            pv_sorted = sorted(zip(PV.z[iEvt],PV.cat[iEvt],PV.n[iEvt]), key=lambda x: x[0])
            n_pvs = len(pv_sorted)
            good_pvs = []
            good_pv_bins = []
            pv_sorted_bins = []

            # --- LOOP over all PVs
            for ii in range(n_pvs):

                pv_sorted_bins.append(np.floor(10.*(pv_sorted[ii][0]+100)))

                # PV with at least 5 tracks are considered as good PVs
                # Can also filter using "pv_sorted[ii][1]", which is a boolean at 1 if nTracks_good_PV is >= 5
                if pv_sorted[ii][2]>=nTracks_good_PV:
                    good_pvs.append(pv_sorted[ii])
                    good_pv_bins.append(np.floor(10.*(pv_sorted[ii][0]+100)))

            n_good_pv_bins = len(good_pv_bins)

            sv_sorted = sorted(zip(SV.z[iEvt],SV.cat[iEvt],SV.n[iEvt]), key=lambda x: x[0])
            n_svs = len(sv_sorted)
            sv_sorted_bins = []
            # --- LOOP over all SVs
            for ii in range(n_svs):
                sv_sorted_bins.append(np.floor(10.*(sv_sorted[ii][0]+100)))

            # ==============================
            # --- Start plotting things here
            # ==============================
            plt.figure(figsize=(15,10))
            max_y_kde = max(y_kde)
            max_y_pred = 0
            for it in iters:
                max_y_pred = max(y_pred[it])

            #print(max_y_kde,max_y_pred)
            max_y = max(max_y_kde,max_y_pred)
            #print(max_y)

            plt.ylim(0, 1.1*max(y_pred[iters[-1]]))
            #plt.ylim(0.0005,1.15*max_y)

            pvstr = "PVs" # tag for the text box info on the plot
            if len(good_pv_bins)<=1:
                pvstr = "PV"

            # --- Add the true KDE distribution
            plt.plot(y_kde, color="royalblue", label="True KDE")

            # --- Add the predicted KDE distributions (for each model iteration)
            for it in iters:
                if len(iters)>1:
                    plt.plot(y_pred[it], color=cm.jet(it/len(model_dicts)), alpha = 0.8)
                else:
                    plt.plot(y_pred[it], color=cm.jet(it/len(model_dicts)), alpha = 0.8)

            # --- Add markers showing the PVs position
            # - LOOP over all PVs
            for ipv in range(n_pvs):
                xPV = pv_sorted_bins[ipv]
                yPV = max(y_kde[int(xPV)-50:int(xPV)+50]) + 0.05*max_y
                plt.scatter((xPV),(yPV), s=61, marker = "o", color="blue", facecolor="cyan")
                #color = 'blue', marker = "o", markersize=8, markerfacecolor="cyan"
            # Add once more the last PV with the label for legend purpose (not ideal, but it works...)
            if n_pvs>n_good_pv_bins:
                plt.scatter((xPV),(yPV), s=61, marker = "o", color="blue", facecolor="cyan", label='%s (#Trks$<%s$)'%(pvstr,nTracks_good_PV))

            # - LOOP over good PVs
            for ipv in range(n_good_pv_bins):
                xPV = good_pv_bins[ipv]
                yPV = max(y_kde[int(xPV)-50:int(xPV)+50]) + 0.05*max_y
                plt.scatter((xPV),(yPV), s=61, marker="o", color="red", facecolor="orange")
            # Add once more the last goo PV with the label for legend purpose (not ideal, but it works...)
            plt.scatter((xPV),(yPV), s=61, marker="o", color="red", facecolor="orange", label='%s (#Trks$\geq%s$)'%(pvstr,nTracks_good_PV))

            # - LOOP over all SVs
            if plotSV:
                for isv in range(n_svs):
                    xSV = sv_sorted_bins[isv]
                    ySV = max(y_kde[int(xSV)-50:int(xSV)+50]) + 0.1*max_y
                    plt.scatter((xSV),(ySV), s=41, marker = "o", color="darkgreen", facecolor="limegreen")
                    #color = 'blue', marker = "o", markersize=8, markerfacecolor="cyan"
                # Add once more the last PV with the label for legend purpose (not ideal, but it works...)
                if n_svs>0:
                    plt.scatter((xSV),(ySV), s=41, marker = "o", color="darkgreen", facecolor="limegreen", label='SVs')

            # --- Add the legend (in the upper-right corner: 'loc=1')
            #plt.legend(loc=1, facecolor='white', edgecolor='white', fontsize=fsize)

            # --- Text box info added on the top left part of the plot, with
            # - event number
            # - number of good PVs out of the total number of PVs for the event
            textstr = '\n'.join((
            r'Toy MC simulation',     
            #r'',     
            #r'Evt #%s' % (iEvt, ),     
            #r'$-$ %s good %s (of %s)' %(len(good_pv_bins),pvstr,len(pv_sorted)) 
            ))              
            #props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.5) #
            #ax.text(0.05, 0.90, textstr, transform=ax.transAxes,verticalalignment='top', bbox=props,linespacing = 1.8, fontsize=fsize)

            # --- Set the plot x- and y-axis labels
            plt.xlabel("z (in bin #)", fontsize=fsize_ax)
            plt.ylabel("Arbitrary unit", fontsize=fsize_ax)
            plt.legend()
#             for tick in ax.xaxis.get_major_ticks():
#                 tick.label1.set_fontsize(fsize_ax)
#             for tick in ax.yaxis.get_major_ticks():
#                 tick.label1.set_fontsize(fsize_ax)

            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet))