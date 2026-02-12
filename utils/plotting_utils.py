import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import mplhep as hep
plt.style.use(hep.style.ROOT)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib.patches import Wedge, Rectangle, Ellipse
from pathlib import Path, PosixPath
from utils.utilities import pv_isolation, tracks_isolation_only, tracks_weighted_pos
import numpy as np

# ==========================================================================================
mystyle = {
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "font.size": 18,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
}

RUI_STYLES = {
    "kernel": dict(alpha=0.3, color="g"),
    "target": dict(alpha=0.6, color="b"),
    "predicted": dict(alpha=0.6, color="r"),
    "masked": dict(alpha=0.3, color="k"),
}

plot_style = {
    "fsize":18,
    "fsize_ax":20,
}

d_colors = {
  -1:'black',
   0:'red',
   1:'blue',
   2:'darkgreen',
   3:'sandybrown',
   4:'darkviolet',
   5:'dodgerblue',
   6:'darkorange',
   7:'deeppink',
   8:'teal',
   9:'purple',
  10:'orangered',
  11:'darkblue',
  12:'crimson',
  13:'cyan',
  14:'gold',
  15:'deepskyblue',
  16:'olive',
  17:'blueviolet',
  18:'tomato',
  19:'turquoise',
}

# ==========================================================================================
def cylinder_theta(z):
    return (z+100) * 2*np.pi / (300+100)

def cylinder_xy(z):     
    theta = cylinder_theta(z)
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y

def get_pv_res(nTrks):
    ## Parameters' values for the resolution as a function of the number 
    ## of tracks originating from the PV.
    A_res= 926.0
    B_res= 0.84
    C_res= 10.7
    
    ## Resolution function (in [mm]? --> to be checked!!)
    return (A_res * np.power(nTrks, -1 * B_res) + C_res) * 0.001

# ==========================================================================================
def plot_truth_vs_predict(truth, predict, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 2))

    non_zero, = np.nonzero(np.round(truth + predict, 4))

    ax.plot(-truth, label="Truth")
    ax.plot(predict, label="Prediction")
    ax.set_xlim(min(non_zero) - 20, max(non_zero) + 400)
    ax.legend()
    return ax

# ==========================================================================================
def replace_in_ax(ax, lines, x_values, y_values):
    lines.set_data(x_values, y_values)
    if np.max(y_values) > 0:
        ax.set_ylim(np.min(y_values) * 0.9, np.max(y_values) * 1.1)
    ax.set_xlim(-0.5, x_values[-1] + 0.5)

# ==========================================================================================
def get_color(style):
    color = style.get("color")
    if color is None:
        color = style.get("edgecolor")
    if color is None:
        color = "k"
    return color

# ==========================================================================================
def setAxes(ax, x_label="x_label", y_label="y_label",):    
    # --- Set the plot x- and y-axis labels
    ax.set_xlabel(x_label, fontsize=plot_style["fsize_ax"])
    ax.set_ylabel(y_label, fontsize=plot_style["fsize_ax"])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(plot_style["fsize_ax"])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(plot_style["fsize_ax"])
    return ax

# ==========================================================================================
def plot_ruiplot(
    zvals, i, inputs, labels, outputs, width=25, ax=None, styles=RUI_STYLES
):
    x_bins = np.round(zvals[i - width : i + width] - 0.05, 2)
    y_kernel = inputs.squeeze()[i - width : i + width] * 2500
    y_target = labels.squeeze()[i - width : i + width]
    y_predicted = outputs.squeeze()[i - width : i + width]

    with plt.rc_context(mystyle):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlim(zvals[i - width] - 0.05, zvals[i + width] - 0.05)
        ax.set_xlabel("z values [mm]")

        ax.bar(x_bins, y_kernel, width=0.1, **styles["kernel"], label="Kernel Density")

        ax.legend(loc="upper left")

        ax.set_ylim(0, max(y_kernel) * 1.2)

        ax.set_ylabel("Kernel Density", color=get_color(styles["kernel"]))

        ax_prob = ax.twinx()

        p1 = ax_prob.bar(
            x_bins, y_target, width=0.1, **styles["target"], label="Target"
        )
        p2 = ax_prob.bar(
            x_bins, y_predicted, width=0.1, **styles["predicted"], label="Predicted"
        )

        ax_prob.set_ylim(0, max(0.8, 1.2 * max(y_predicted)))
        ax_prob.set_ylabel("Probability", color=get_color(styles["predicted"]))

        if np.any(np.isnan(labels)):
            grey_y = np.isnan(y_target) * 0.2
            ax_prob.bar(x_bins, grey_y, width=0.1, **styles["masked"], label="Masked")

        ax_prob.legend(loc="upper right")

    return ax, ax_prob


# ==========================================================================================
def dual_train_plots(x=(), train=(), validation=(), eff=(), FP_rate=(), *, axs=None):

    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    ax, eff_ax = axs
    fp_ax = eff_ax.twinx()

    lines = dict()
    lines["train"], = ax.plot(x, train, "o-", label="Train")
    lines["val"],   = ax.plot(x, validation, "o-", label="Validation")

    lines["eff"], = eff_ax.plot(x, eff, "o-b", label="Eff")
    lines["fp"],  = fp_ax.plot(x, FP_rate, "o-r", label="FP rate")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cost")

    eff_ax.set_xlabel("Epochs")
    eff_ax.set_ylabel("Eff", color="b")
    fp_ax.set_ylabel("FP rate", color="r")

    ax.set_yscale("log")
    ax.legend()
    eff_ax.legend(loc="upper right")
    fp_ax.legend(loc="lower left")
    return ax, fp_ax, eff_ax, lines

# ==========================================================================================
def cost_train_plot(x=(), train=(), validation=(), *, axs=None):

    ## ----------------------------------------------------------------
    ## plotting parameters
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    
    if axs is None:
        fig, axs = plt.subplots()

    lines = dict()
    lines["train"], = axs.plot(x, train,      "o-", label="Train")
    lines["val"],   = axs.plot(x, validation, "o-", label="Validation")

    axs = setAxes(axs,x_label="Epochs",y_label="Cost")
        
    axs.legend()
    axs.legend(loc=1, facecolor='white', edgecolor='white', fontsize=plot_style["fsize"])
    
    return axs, lines

# ==========================================================================================
def plot_z_hist(ymax, z_vals, label, model_pred, configs, plot_tag, display=True):

    if not display:
        plt.ioff()
        
    ## ----------------------------------------------------------------
    ## plotting parameters
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size
    
    ## ----------------------------------------------------------------
    ## make fig 
    fig, ax = plt.subplots()

    ## ----------------------------------------------------------------
    ## plotting labels and pmodel predictions
    if configs["model_class"]=="tracks-to-KDE":
        plt.plot(z_vals, label, color='red',label='target KDE',linewidth=2.0)
        
    else:
        plt.bar(z_vals,height=label,     width=0.1,color='red',edgecolor='red',alpha=0.55)
        plt.plot(z_vals, label, color='red',label='target hist',linewidth=2.0)        

    if configs["model_class"]=="tracks-to-hist" or configs["model_class"]=="KDE-to-hist":
        plt.bar(z_vals,height=model_pred,width=0.1,color='b',  edgecolor='b',  alpha=0.55)
    plt.plot(z_vals, model_pred,color='b', label='model prediction',linewidth=2.0)
        
    ## ----------------------------------------------------------------
    ## Set the axes limits
    plt.xlim(z_vals[0],z_vals[len(z_vals)-1])
    plt.ylim((0.,1.1*ymax))
    
    ## ----------------------------------------------------------------
    ## Setting the axes labels
    ax.set_xlabel("z [mm]",                       fontsize=plot_style["fsize_ax"], loc="center")        
    if configs["model_class"]=="tracks-to-KDE":
        ax.set_ylabel("POCA KDE A\n[arbitrary unit]", fontsize=plot_style["fsize_ax"], loc="center") 
    else:
        ax.set_ylabel("target hists\n[arbitrary unit]", fontsize=plot_style["fsize_ax"], loc="center") 

    ## ----------------------------------------------------------------
    ## Add the legend
    plt.legend()

    ## ----------------------------------------------------------------
    ## Save the plots
    plt.savefig("%s.png"%plot_tag)
    plt.savefig("%s.pdf"%plot_tag)
    
    ## ----------------------------------------------------------------
    ## Display (or not the plots in notebook)
    if display:
        plt.show()     
    else:
        plt.close()        

## ===================================================================================================
def plot_xy_vsz_tracks_polar_inline(iEvt,
                                    l_pvs,
                                    pv_x,
                                    pv_y,
                                    pv_z,
                                    pv_key,
                                    z_min, z_max,
                                    tracks_x, 
                                    tracks_sigma_x, 
                                    tracks_y, 
                                    tracks_sigma_y, 
                                    tracks_z,
                                    tracks_sigma_z,
                                    tracks_pv_key,
                                    typeV,
                                    plot_tag,
                                    sliced_Z
                                   ):
        
    fig, ax = plt.subplots(1, 1, figsize=(10,10), sharey=False, tight_layout=True)
    
    plot_tag += "_"
    plot_tag += typeV
    plot_tag += "_Proj_sliceZ_%s"%sliced_Z
    
    if typeV=="X":
        ax.set_xlabel('$Z$ [mm]')
        ax.set_ylabel('$X$ [mm]')
    if typeV=="Y":
        ax.set_xlabel('$Z$ [mm]')
        ax.set_ylabel('$Y$ [mm]')
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size

    ## Set the plot frame size
    y_t_fram = +0.5 
    y_b_fram = -0.5
    x_r_fram = z_max
    x_l_fram = z_min
    z_center = (z_max+z_min)/2.
    print("z_center=",z_center)
    print("z_center-0.05*(z_center-x_l_fram)=",z_center-0.05*(z_center-x_l_fram))
    plt.xlim([x_l_fram, x_r_fram])    
    plt.ylim([y_b_fram, y_t_fram])
    #plt.invert_xaxis()
    
    x_l_col = x_l_fram+0.03*(x_r_fram-x_l_fram)
    
    l_rec = 0.015
    h_rec = 0.025*(x_r_fram-x_l_fram)
    
    d_rec_txt = 0.02*(x_r_fram-x_l_fram)
    
    y_t_col = y_t_fram-0.12

    
    plt.gca().annotate('Event # %s'%(iEvt),  xy=(z_center-0.10*(z_center-x_l_fram), y_t_fram-0.05), xycoords='data', fontsize=15, weight='bold')
    plt.gca().annotate('$%.2f<z<%.2f$ [mm]'%(z_min,z_max),  xy=(z_center-0.20*(z_center-x_l_fram), y_t_fram-0.08), xycoords='data', fontsize=13, weight='bold')

    ## ===================================================================================================
    ## ===================================================================================================

    ## ----------------------------------------
    ## Loop over tracks 
    for i_t in range(len(tracks_z)):
        if tracks_z[i_t]>z_min and tracks_z[i_t]<z_max and abs(tracks_x[i_t])<1 and abs(tracks_y[i_t])<1:
            if typeV=="X":
                ell = Ellipse((tracks_z[i_t], tracks_x[i_t]),tracks_sigma_z[i_t], tracks_sigma_x[i_t],color=d_colors[int(tracks_pv_key[i_t])], alpha=0.15)
            if typeV=="Y":
                ell = Ellipse((tracks_z[i_t], tracks_y[i_t]),tracks_sigma_z[i_t], tracks_sigma_y[i_t],color=d_colors[int(tracks_pv_key[i_t])], alpha=0.15)
            ## Add the tracks position (x,y) as an ellipse 
            ax.add_artist(ell)
    
    ## ----------------------------------------
    ## Loop over PVs 
    pos = 0
    for i_pv in l_pvs:        
        if typeV=="X":
            ax.scatter(pv_z[i_pv], pv_x[i_pv], s=200, marker="^", c=d_colors[i_pv], alpha=1.0)
        if typeV=="Y":
            ax.scatter(pv_z[i_pv], pv_y[i_pv], s=200, marker="^", c=d_colors[i_pv], alpha=1.0)

        # ------------------
        spaceUnit = ""
        if i_pv<10:
            spaceUnit = "  "
        
        # ------------------
        space=""
        if abs(pv_z[i_pv])<100:
            space="  "
        if abs(pv_z[i_pv])<10:
            space="    "                    

        ax.add_artist(Rectangle((x_l_col,y_t_col-pos*0.03),h_rec,l_rec,color=d_colors[int(pv_key[i_pv])], alpha=0.5))
        if pv_z[i_pv]<0:
            plt.gca().annotate(r'PV[%s]: %s$z =$ %s$%.2f$ [mm]'%(i_pv,spaceUnit,space,pv_z[i_pv]),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-pos*0.03), xycoords='data', fontsize=13)
        else:
            plt.gca().annotate(r'PV[%s]: %s$z =$ %s$+%.2f$ [mm]'%(i_pv,spaceUnit,space,pv_z[i_pv]),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-pos*0.03), xycoords='data', fontsize=13)
        pos+=1
        

    plt.savefig("%s.png"%plot_tag)
    plt.savefig("%s.pdf"%plot_tag)
            
    plt.show()            

    
## ===================================================================================================
def plot_xy_tracks_polar_inline(iEvt,
                                l_pvs,
                                pv_x,
                                pv_y,
                                pv_z,
                                pv_key,
                                z_min, z_max,
                                tracks_x, 
                                tracks_sigma_x, 
                                tracks_y, 
                                tracks_sigma_y, 
                                tracks_z,
                                tracks_sigma_z,
                                tracks_pv_key,
                                typeV,
                                plot_tag,
                                sliced_Z
                               ):
        
    fig, ax = plt.subplots(1, 1, figsize=(10,10), sharey=False, tight_layout=True)
    
    plot_tag += "_"
    plot_tag += typeV
    plot_tag += "_Proj_sliceZ_%s"%sliced_Z
    
    if typeV=="XY":
        ax.set_xlabel('$X$ [mm]')
        ax.set_ylabel('$Y$ [mm]')
    if typeV=="tXtY":
        ax.set_xlabel('$t_X$ [mm]')
        ax.set_ylabel('$t_Y$ [mm]')
    if typeV=="tXX":
        ax.set_xlabel('$t_X$ [mm]')
        ax.set_ylabel('$X$ [mm]')
    if typeV=="tYY":
        ax.set_xlabel('$t_Y$ [mm]')
        ax.set_ylabel('$Y$ [mm]')
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size

    ## Set the plot frame size
    y_t_fram = +0.5 
    y_b_fram = -0.5
    x_r_fram = (y_t_fram - y_b_fram)/2.
    x_l_fram = -x_r_fram
    plt.xlim([x_l_fram, x_r_fram])    
    plt.ylim([y_b_fram, y_t_fram])   
    
    
    x_l_col = x_l_fram+0.03
    
    l_rec = 0.015
    h_rec = 0.025
    
    d_rec_txt = 0.02
    
    y_t_col = y_t_fram-0.12

    
    plt.gca().annotate('Event # %s'%(iEvt),  xy=(-0.05, y_t_fram-0.05), xycoords='data', fontsize=15, weight='bold')
    plt.gca().annotate('$%.2f<z<%.2f$ [mm]'%(z_min,z_max),  xy=(-0.12, y_t_fram-0.08), xycoords='data', fontsize=13, weight='bold')

    ## ----------------------------------------
    ## Loop over tracks 
    for i_t in range(len(tracks_z)):
        if tracks_z[i_t]>z_min and tracks_z[i_t]<z_max and abs(tracks_x[i_t])<1 and abs(tracks_y[i_t])<1:
            ell = Ellipse((tracks_x[i_t], tracks_y[i_t]), tracks_sigma_x[i_t], tracks_sigma_y[i_t],color=d_colors[int(tracks_pv_key[i_t])], alpha=0.15)
            ## Add the tracks position (x,y) as an ellipse 
            ax.add_artist(ell)


    ## ===================================================================================================
    ## ===================================================================================================
    ## ----------------------------------------
    ## Loop over PVs 
    pos = 0
    for i_pv in l_pvs:        
        ax.scatter(pv_x[i_pv], pv_y[i_pv], s=200, marker="^", c=d_colors[i_pv], alpha=1.0)

        # ------------------
        spaceUnit = ""
        if i_pv<10:
            spaceUnit = "  "
        
        # ------------------
        space=""
        if abs(pv_z[i_pv])<100:
            space="  "
        if abs(pv_z[i_pv])<10:
            space="    "                    

        ax.add_artist(Rectangle((x_l_col,y_t_col-pos*0.03),h_rec,l_rec,color=d_colors[int(pv_key[i_pv])], alpha=0.5))
        if pv_z[i_pv]<0:
            plt.gca().annotate(r'PV[%s]: %s$z =$ %s$%.2f$ [mm]'%(i_pv,spaceUnit,space,pv_z[i_pv]),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-pos*0.03), xycoords='data', fontsize=13)
        else:
            plt.gca().annotate(r'PV[%s]: %s$z =$ %s$+%.2f$ [mm]'%(i_pv,spaceUnit,space,pv_z[i_pv]),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-pos*0.03), xycoords='data', fontsize=13)
        pos+=1
        
            
    plt.savefig("%s.png"%plot_tag)
    plt.savefig("%s.pdf"%plot_tag)
            
    plt.show()            
       
## ===================================================================================================
def plot_tracks_polar(iEvt, 
                      pv_x, pv_y, pv_z, pv_n, pv_key, 
                      tracks_x, tracks_sigma_x, 
                      tracks_y, tracks_sigma_y, 
                      tracks_z, tracks_sigma_z, 
                      tracks_tx, tracks_ty,
                      tracks_pv_key, 
                      dataType='',
                      doFilt=False, min_d_tracks=0.5, min_d_pv=5.0, 
                      inv_sigma_z=False, loc='/data/home/sakar/ML_dir/'):
    
    if dataType == '':
        print("Please enter a daat type: e.g. MinBias_MagUp...")
        sys.exit()
        
    n_tracks_total = len(tracks_z)
    if doFilt:
        filt = tracks_isolation_only(tracks_z, min_dz=min_d_tracks)
        tracks_x = tracks_x[filt==False]
        tracks_sigma_x = tracks_sigma_x[filt==False]
        tracks_y = tracks_y[filt==False]
        tracks_sigma_y = tracks_sigma_y[filt==False]
        tracks_z = tracks_z[filt==False]        
        tracks_sigma_z = tracks_sigma_z[filt==False]
        tracks_tx = tracks_tx[filt==False]
        tracks_ty = tracks_ty[filt==False]
        tracks_pv_key  = tracks_pv_key[filt==False]
        n_tracks_filt  = n_tracks_total - len(tracks_z)    
    
    ## ===================================================================================================
    ## ===================================================================================================
    ## ----------------------------------------
    ## Define plot style
    ## ----------------------------------------
    fig, ax = plt.subplots()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size

    ## Set the plot frame size
    y_t_fram = +2.4 + 0.1*len(pv_z)
    y_b_fram = -1.8
    x_r_fram = (y_t_fram - y_b_fram)/2.
    x_l_fram = -x_r_fram
    plt.xlim([x_l_fram, x_r_fram])    
    plt.ylim([y_b_fram, y_t_fram])   
    ## Get current axes
    a = plt.gca()
    ## Set visibility of x-axis as False
    xax = a.axes.get_xaxis()
    xax = xax.set_visible(False)
    ## Set visibility of y-axis as False
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)
    
    ## ===================================================================================================
    ## ===================================================================================================
    ## ----------------------------------------
    ## Define circles: wrapping z coordinates around circles
    ##
    ##   - Inner circle: r = r_inner_circle 
    ##       ==> PV Z position as wedges ; width proportional to resolution (using true number of tracks)
    ##   - Outer ring:   r in [r_outer_ring_min, r_outer_ring_max]
    ##       ==> tracks Z position as lines (very thin wedges)
    ##       ==> tracks resolution as circles centered 
    ## 
    ## ----------------------------------------
    scale = 1.3
    r_outer_ring_max = 1.2*scale
    r_outer_ring_min = 1.0*scale

    r_inner_ring_max = 0.90*scale
    r_inner_ring_min = 0.80*scale
    
    r_inner_circle = 0.75*scale
    
    circ = plt.Circle((0, 0), radius=r_outer_ring_max, edgecolor='black', facecolor='None')
    ax.add_patch(circ)

    circ = plt.Circle((0, 0), radius=r_outer_ring_min, edgecolor='black', facecolor='None')
    ax.add_patch(circ)

    circ = plt.Circle((0, 0), radius=r_inner_ring_max, edgecolor='black', facecolor='None')
    ax.add_patch(circ)

    circ = plt.Circle((0, 0), radius=r_inner_ring_min, edgecolor='black', facecolor='None')
    ax.add_patch(circ)

    circ = plt.Circle((0, 0), radius=r_inner_circle, edgecolor='black', facecolor='None')
    ax.add_patch(circ)


    ## ===================================================================================================
    ## ===================================================================================================
    ## ----------------------------------------
    ## Get the x and y coordinates (wrapping z around a circle) for each track and each PV
    x_t,  y_t  = cylinder_xy(tracks_z)
    x_pv, y_pv = cylinder_xy(pv_z)
    ## ----------------------------------------
    ## Get the pv resolutions
    pv_res     = get_pv_res(pv_n)
    
    
    ## ===================================================================================================
    ## ===================================================================================================

    ## ----------------------------------------
    ## Add event number and tracks info
    plt.gca().annotate('Event # %s'%(iEvt),  xy=(-0.2, y_t_fram-0.12), xycoords='data', fontsize=15, weight='bold')
    plt.gca().annotate('N tracks total: %s'%n_tracks_total,  xy=(-0.3, y_t_fram-0.25), xycoords='data', fontsize=13, weight='bold')
    
    if doFilt:
        plt.gca().annotate('Isolated track filtered with $ | \Delta z_{ij} | <%s$ : removed %s tracks'%(min_d_tracks,n_tracks_filt),  xy=(-1.00, y_t_fram-0.35), xycoords='data', fontsize=13, weight='bold')
        #plt.gca().annotate('$ | \Delta z_{ij} | <%s$',  xy=(-0.15, 2.35), xycoords='data', fontsize=13, weight='bold')
        #plt.gca().annotate(''%(n_tracks_filt),  xy=(-0.25, 2.25), xycoords='data', fontsize=13, weight='bold')
    
    x_l_col = x_l_fram+0.05
    x_r_col = x_l_fram+2.35
    
    l_rec = 0.05
    h_rec = 0.1
    
    d_rec_txt = 0.07
    
    y_t_col = y_t_fram-0.5
      
    ## ----------------------------------------
    ## Add Z coordinates around the outter ring
    ## ----------------------------------------
    plt.gca().annotate(r'$z = - 100$',  xy=(r_outer_ring_max+0.1, 0), xycoords='data', fontsize=15)
    plt.gca().annotate(r'$z = - 50$',   xy=(np.cos(1*np.pi/4.)*r_outer_ring_max+0.1, np.sin(1*np.pi/4.)*r_outer_ring_max+0.1), xycoords='data', fontsize=15)
    plt.gca().annotate(r'$z = 0 \, [mm]$',      xy=(-0.1, r_outer_ring_max+0.1), xycoords='data', fontsize=15)
    plt.gca().annotate(r'$z = 50$',     xy=(np.cos(3*np.pi/4.)*r_outer_ring_max-0.4, np.sin(3*np.pi/4.)*r_outer_ring_max+0.1), xycoords='data', fontsize=15)
    plt.gca().annotate(r'$z = 100$',    xy=(-r_outer_ring_max-0.5, 0), xycoords='data', fontsize=15)
    plt.gca().annotate(r'$z = 150$',    xy=(np.cos(5*np.pi/4.)*r_outer_ring_max-0.5, np.sin(5*np.pi/4.)*r_outer_ring_max-0.1), xycoords='data', fontsize=15)
    plt.gca().annotate(r'$z = 250$',    xy=(np.cos(7*np.pi/4.)*r_outer_ring_max+0.1, np.sin(7*np.pi/4.)*r_outer_ring_max-0.1), xycoords='data', fontsize=15)

    ## ----------------------------------------
    # Add the Z grid inside the circles
    ## ----------------------------------------
    # horizontal line
    plt.plot([-r_outer_ring_max, r_outer_ring_max], [0, 0], color = 'black', linewidth=0.5, linestyle = '--')
    # vertical line
    plt.plot([0, 0], [-r_outer_ring_max, r_outer_ring_max], color = 'black', linewidth=0.5, linestyle = '--')
    # diagonal line (top left to bottom right)
    plt.plot([np.cos(3*np.pi/4.)*r_outer_ring_max, np.cos(1*np.pi/4.)*r_outer_ring_max], [np.sin(1*np.pi/4.)*r_outer_ring_max, np.sin(5*np.pi/4.)*r_outer_ring_max], color = 'black', linewidth=0.25, linestyle = '--')
    # diagonal line (bottom left to top right)
    plt.plot([np.cos(1*np.pi/4.)*r_outer_ring_max, np.cos(3*np.pi/4.)*r_outer_ring_max], [np.sin(3*np.pi/4.)*r_outer_ring_max, np.sin(7*np.pi/4.)*r_outer_ring_max], color = 'black', linewidth=0.25, linestyle = '--')

    ## ----------------------------------------
    ## Add Z ticks around the outter ring
    ## ----------------------------------------
    ## Define long ticks spaced by pi/4. [-> 8 long ticks]
    z_ticks_len = 0.05
    for i in range(8):    
        theta_tick=i*np.pi/4.
        plt.plot([np.cos(theta_tick)*r_outer_ring_max, np.cos(theta_tick)*r_outer_ring_max+z_ticks_len*np.cos(theta_tick)], [np.sin(theta_tick)*r_outer_ring_max, np.sin(theta_tick)*r_outer_ring_max+z_ticks_len*np.sin(theta_tick)], color = 'black', linewidth=1)
    ## Define short ticks spaced by pi/20. [-> 40 short ticks]
    z_ticks_len = 0.015
    for i in range(40):
        if i%5==0: continue
        theta_tick=i*np.pi/20.
        plt.plot([np.cos(theta_tick)*r_outer_ring_max, np.cos(theta_tick)*r_outer_ring_max+z_ticks_len*np.cos(theta_tick)], [np.sin(theta_tick)*r_outer_ring_max, np.sin(theta_tick)*r_outer_ring_max+z_ticks_len*np.sin(theta_tick)], color = 'black', linewidth=1)

    
    ## ==============================================================================================================================
    ## ==============================================================================================================================
    ## ----------------------------------------
    ## Loop over tracks 
    for i_t in range(len(tracks_z)):
        if tracks_pv_key[i_t]>-2 and tracks_z[i_t]>-150 and tracks_z[i_t]<350:
            ## Add the tracks position as a line (very thin wedge) inside the outer ring
            wedge = Wedge((x_t[i_t]*r_outer_ring_min, y_t[i_t]*r_outer_ring_min), (r_outer_ring_max-r_outer_ring_min), np.rad2deg(cylinder_theta(tracks_z[i_t]-0.001)), np.rad2deg(cylinder_theta(tracks_z[i_t]+0.001)), color=d_colors[int(tracks_pv_key[i_t])], alpha=0.5)
            ax.add_artist(wedge)
            ## Add the tracks "weight" as a circle with area the inverse of the track resoltion in Z
            #ax.scatter(x_t[i_t], y_t[i_t], s=1./tracks_sigma_z[i_t], c=d_colors[int(tracks_pv_key[i_t])], alpha=0.5)
            ax.scatter(x_t[i_t]*r_outer_ring_min, y_t[i_t]*r_outer_ring_min, s=np.pi*np.power(tracks_sigma_z[i_t],2), c=d_colors[int(tracks_pv_key[i_t])], alpha=0.5)


    ## ----------------------------------------
    ## Loop over pvs 
    max_pv_legend = 15
    for i_pv in range(len(pv_z)):
        ## Add the pv position as a wedge with a "width" proportional to the pv resolution.
        ## Here the resolution is enhanced by a factor 10 to better visualize the differences between each pv resolution
        wedge = Wedge((0,0), r_inner_circle, np.rad2deg(cylinder_theta(pv_z[i_pv]-10*pv_res[i_pv])), np.rad2deg(cylinder_theta(pv_z[i_pv]+10*pv_res[i_pv])), color=d_colors[int(pv_key[i_pv])], alpha=0.5)
        ax.add_artist(wedge)

        # ------------------
        spaceUnit = ""
        if i_pv<10:
            spaceUnit = "  "
        
        # ------------------
        space=""
        if abs(pv_z[i_pv])<100:
            space="  "
        if abs(pv_z[i_pv])<10:
            space="    "            
        
        # ------------------
        spaceUnitTracksPV = ""
        if pv_n[i_pv]<10:
            spaceUnitTracksPV = "  "
        
        # ------------------
        label_tracks = ""
        
        
        if len(tracks_pv_key[tracks_pv_key==i_pv])>0:
            ## Get the weighted position of the tracks belonging to PV i_pv
            mu_x,mu_y,mu_z,s_x,s_y,s_z = tracks_weighted_pos(i_pv,
                                                             tracks_x,       tracks_y,       tracks_z,
                                                             tracks_sigma_x, tracks_sigma_y, tracks_sigma_z,
                                                             tracks_pv_key                           
                                                            )
            #print(mu_x,mu_y,mu_z,s_x,s_y,s_z)
            ## Transform the weighted z position into cylindrical coords
            mu_z_cyl_x,  mu_z_cyl_y  = cylinder_xy(mu_z)
            ## Add the weighted z position to the plot in the inner ring
            wedge = Wedge((mu_z_cyl_x*(r_inner_ring_min), mu_z_cyl_y*(r_inner_ring_min)), (r_inner_ring_max-r_inner_ring_min), np.rad2deg(cylinder_theta(mu_z-10*s_z)), np.rad2deg(cylinder_theta(mu_z+10*s_z)), color=d_colors[int(pv_key[i_pv])], alpha=0.95)        
            ax.add_artist(wedge)
            space=""
            if abs(mu_z)<100:
                space="  "
            if abs(mu_z)<10:
                space="    "
            spaceUnitTracks = ""
            if len(tracks_pv_key[tracks_pv_key==i_pv])<10:
                spaceUnitTracks = "  "
                
            if mu_z<0:
                label_tracks = "$<z>_{t} =$ %s$%.2f \pm %.2f$ %s(%s trks)"%(space,mu_z,s_z, spaceUnitTracks,len(tracks_pv_key[tracks_pv_key==i_pv]))
            else:
                label_tracks = "$<z>_{t} =$ %s$+%.2f \pm %.2f$ %s(%s trks)"%(space,mu_z,s_z, spaceUnitTracks,len(tracks_pv_key[tracks_pv_key==i_pv]))
                
            
        if i_pv<max_pv_legend:
            ax.add_artist(Rectangle((x_l_col,y_t_col-i_pv*0.1),h_rec,l_rec,color=d_colors[int(pv_key[i_pv])], alpha=0.5))
            if pv_z[i_pv]<0:
                plt.gca().annotate(r'PV[%s]: %s$z =$ %s$%.2f \pm %.2f$ %s(%s trks) ; %s'%(i_pv,spaceUnit,space,pv_z[i_pv],pv_res[i_pv], spaceUnitTracksPV, pv_n[i_pv], label_tracks),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-i_pv*0.1), xycoords='data', fontsize=13)
            else:
                plt.gca().annotate(r'PV[%s]: %s$z =$ %s$+%.2f \pm %.2f$ %s(%s trks) ; %s'%(i_pv,spaceUnit,space,pv_z[i_pv],pv_res[i_pv], spaceUnitTracksPV, pv_n[i_pv], label_tracks),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-i_pv*0.1), xycoords='data', fontsize=13)
        else:
            ax.add_artist(Rectangle((x_r_col,y_t_col-(i_pv-max_pv_legend)*0.1),h_rec,l_rec,color=d_colors[int(pv_key[i_pv])], alpha=0.5))
            if pv_z[i_pv]<0:
                plt.gca().annotate(r'PV[%s]: %s$z =$ %s$%.2f \pm %.2f$ (%s trks) ; %s'%(i_pv,spaceUnit,space,pv_z[i_pv],pv_res[i_pv],pv_n[i_pv], label_tracks),  xy=(x_r_col+h_rec+d_rec_txt, y_t_col-(i_pv-max_pv_legend)*0.1), xycoords='data', fontsize=13)
            else:
                plt.gca().annotate(r'PV[%s]: %s$z =$ %s$+%.2f \pm %.2f$ (%s trks) ; %s'%(i_pv,spaceUnit,space,pv_z[i_pv],pv_res[i_pv],pv_n[i_pv], label_tracks),  xy=(x_r_col+h_rec+d_rec_txt, y_t_col-(i_pv-max_pv_legend)*0.1), xycoords='data', fontsize=13)
                                
        
    if len(pv_z)<max_pv_legend:
        if len(tracks_pv_key[tracks_pv_key==-1]) > 0:
            ax.add_artist(Rectangle((x_l_col,y_t_col-(len(pv_z))*0.1),h_rec,l_rec,color=d_colors[-1], alpha=0.5))
            plt.gca().annotate(r'Non associated tracks (%s trks)'%(len(tracks_pv_key[tracks_pv_key==-1])),  xy=(x_l_col+h_rec+d_rec_txt, y_t_col-len(pv_z)*0.1), xycoords='data', fontsize=13)
    else:
        if len(tracks_pv_key[tracks_pv_key==-1]) > 0:
            ax.add_artist(Rectangle((x_r_col,y_t_col-(len(pv_z)-9)*0.1),h_rec,l_rec,color=d_colors[-1], alpha=0.5))            
            plt.gca().annotate(r'Non associated tracks (%s trks)'%(len(tracks_pv_key[tracks_pv_key==-1])),  xy=(x_r_col+h_rec+d_rec_txt, y_t_col-(len(pv_z)-9)*0.1), xycoords='data', fontsize=13)
        
    ax.scatter(x_l_fram+0.15, y_b_fram+0.26, s=1./0.002, c='black', alpha=0.5)
    #plt.gca().annotate(r': track resolution; $S = 1/\sigma(z)^2$',  xy=(-1.63, -1.27), xycoords='data', fontsize=12)
    plt.gca().annotate(r': track resolution; $S = \pi\sigma(z)^2$',  xy=(x_l_fram+0.22, y_b_fram+0.23), xycoords='data', fontsize=12)
    ax.add_artist(Wedge((x_l_fram+0.10,y_b_fram+0.05), 0.1, 10, 50, color='black', alpha=0.5))
    plt.gca().annotate(r': pv resolution; $\theta(wedge) = PV(z) \pm 10\delta z$',  xy=(x_l_fram+0.22, y_b_fram+0.07), xycoords='data', fontsize=12)
            
    ## ----------------------------------------------------------------
    ## Save the plots
    plot_tag = Path(loc) 
    plot_tag = plot_tag / 'Tracks_PV_Link/'
    ## Make the global output directory first (if not already done)
    plot_tag.mkdir(exist_ok=True)
    
    plot_tag = str(plot_tag) + '/'
    plot_tag += dataType
    plot_tag += '_iEvt%s'%iEvt
    if inv_sigma_z:
        plot_tag += '_sigma_z_inv'
    else:
        plot_tag += '_sigma_z'
    if doFilt:
        plot_tag += "_FiltTracks"
        plot_tag += "_minDz_%s"%min_d_tracks        
    else:
        plot_tag += "_AllTracks"
    
    if not doFilt:
        ## ------------------------------------------------------------------------
        ## ------------------------------------------------------------------------
        ## ------------------------------------------------------------------------
        ##
        ## Draw tracks X and Y info for PVs which are close in Z 
        ##
        ## ------------------------------------------------------------------------
        iso_pv = pv_isolation(pv_z, min_dz=min_d_pv)
        iso_pv_z = pv_z[iso_pv==False]
        sliced_Z=0
        for i_pv in range(len(pv_z)-1):
            if iso_pv[i_pv]==True: continue
            if pv_z[i_pv+1] - pv_z[i_pv]>min_d_pv: continue
            z_min = (pv_z[i_pv+1]+pv_z[i_pv])/2.-min_d_pv
            z_max = (pv_z[i_pv+1]+pv_z[i_pv])/2.+min_d_pv
            l_pvs = []
            l_pvs.append(i_pv)
            l_pvs.append(i_pv+1)
            if (i_pv+2)<len(pv_z) and pv_z[i_pv+2] - pv_z[i_pv+1]<min_d_pv:
                l_pvs.append(i_pv+2)

            plot_xy_tracks_polar_inline(iEvt,
                                        l_pvs,
                                        pv_x,
                                        pv_y,
                                        pv_z,
                                        pv_key,
                                        z_min, z_max,
                                        tracks_x, 
                                        tracks_sigma_x, 
                                        tracks_y, 
                                        tracks_sigma_y, 
                                        tracks_z,
                                        tracks_sigma_z,
                                        tracks_pv_key,
                                        "XY",
                                        plot_tag,
                                        sliced_Z
                                       )
            plot_xy_vsz_tracks_polar_inline(iEvt,
                                        l_pvs,
                                        pv_x,
                                        pv_y,
                                        pv_z,
                                        pv_key,
                                        z_min, z_max,
                                        tracks_x, 
                                        tracks_sigma_x, 
                                        tracks_y, 
                                        tracks_sigma_y, 
                                        tracks_z,
                                        tracks_sigma_z,
                                        tracks_pv_key,
                                        "X",
                                        plot_tag,
                                        sliced_Z
                                       )
            plot_xy_vsz_tracks_polar_inline(iEvt,
                                        l_pvs,
                                        pv_x,
                                        pv_y,
                                        pv_z,
                                        pv_key,
                                        z_min, z_max,
                                        tracks_x, 
                                        tracks_sigma_x, 
                                        tracks_y, 
                                        tracks_sigma_y, 
                                        tracks_z,
                                        tracks_sigma_z,
                                        tracks_pv_key,
                                        "Y",
                                        plot_tag,
                                        sliced_Z
                                       )
            '''
            plot_xy_tracks_polar_inline(iEvt,
                                        l_pvs,
                                        pv_x,
                                        pv_y,
                                        pv_z,
                                        pv_key,
                                        z_min, z_max,
                                        tracks_z,
                                        tracks_tx, 
                                        tracks_ty, 
                                        tracks_pv_key,
                                        "tXtY",
                                        plot_tag,
                                        sliced_Z
                                       ) 
            plot_xy_tracks_polar_inline(iEvt,
                                        l_pvs,
                                        pv_x,
                                        pv_y,
                                        pv_z,
                                        pv_key,
                                        z_min, z_max,
                                        tracks_z,
                                        tracks_tx, 
                                        tracks_x, 
                                        tracks_pv_key,
                                        "tXX",
                                        plot_tag,
                                        sliced_Z
                                       )  
            plot_xy_tracks_polar_inline(iEvt,
                                        l_pvs,
                                        pv_x,
                                        pv_y,
                                        pv_z,
                                        pv_key,
                                        z_min, z_max,
                                        tracks_z,
                                        tracks_ty, 
                                        tracks_y, 
                                        tracks_pv_key,
                                        "tYY",
                                        plot_tag,
                                        sliced_Z
                                       )   
            '''
            sliced_Z += 1
        
    #print(plot_tag)
    plt.savefig("%s.png"%plot_tag)
    plt.savefig("%s.pdf"%plot_tag)
        
    plt.show()        