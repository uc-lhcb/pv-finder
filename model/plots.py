import matplotlib.pyplot as plt
import numpy as np

def plot_truth_vs_predict(predict, truth, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18,2))
        
    non_zero, = np.nonzero(np.round(truth + predict, 4))
        
    ax.plot(-truth, label='Truth')
    ax.plot(predict, label='Prediction')
    ax.set_xlim(min(non_zero) - 20, max(non_zero) + 400)
    ax.legend()
    return ax