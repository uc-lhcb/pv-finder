import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np

def plot_truth_vs_predict(truth, predict, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18,2))
        
    non_zero, = np.nonzero(np.round(truth + predict, 4))
        
    ax.plot(-truth, label='Truth')
    ax.plot(predict, label='Prediction')
    ax.set_xlim(min(non_zero) - 20, max(non_zero) + 400)
    ax.legend()
    return ax

mystyle = {      
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "font.size": 18,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
}

def plot_ruiplot(zvals, i, inputs, labels, outputs, width=25, ax=None):
    x_bins =np.round(zvals[i-width : i+width] - 0.05, 2)
    y_kernel = inputs.squeeze()[i-width : i+width]*2500
    y_target = labels.squeeze()[i-width : i+width]
    y_predicted = outputs.squeeze()[i-width : i+width]
    
    with plt.rc_context(mystyle):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,7))
            
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlim(zvals[i-width]-0.05, zvals[i+width]-0.05)
        ax.set_xlabel('z values [mm]')
        
        ax.bar(x_bins, y_kernel,
               width=0.1, alpha=0.3 ,color='g', label='Kernel Density')

        ax.legend(loc='upper left')
        
        ax.set_ylim(0, max(y_kernel)*1.2)
        ax.set_ylabel('Kernel Density', color='g')

        ax_prob=ax.twinx()
        
        p1=ax_prob.bar(x_bins, y_target,
                       width=0.1, alpha=0.6, color='b', label='Target')
        p2=ax_prob.bar(x_bins, y_predicted,
                       width=0.1, alpha=0.6, color='r', label='Predicted')

        ax_prob.set_ylim(0, max(0.8, 1.2*max(y_predicted)))
        ax_prob.set_ylabel('Probability', color='r')

        ax_prob.legend(loc='upper right')

    return ax, ax_prob