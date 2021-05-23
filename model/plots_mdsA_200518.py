import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import math

def gaussian(x, pos, width):
    height = 1/(width*math.sqrt(2*math.pi))
    return height*np.exp(-(x-pos)**2 / (2*width**2))

def plot_truth_vs_predict(truth, predict, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 2))

    non_zero, = np.nonzero(np.round(truth + predict, 4))

    ax.plot(-truth, label="Truth")
    ax.plot(predict, label="Prediction")
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

RUI_STYLES = {
    "kernel": dict(alpha=0.3, color="g"),
    "target": dict(alpha=0.6, color="b"),
    "predicted": dict(alpha=0.6, color="r"),
    "masked": dict(alpha=0.3, color="k"),
}


def get_color(style):
    color = style.get("color")
    if color is None:
        color = style.get("edgecolor")
    if color is None:
        color = "k"
    return color


def plot_ruiplot(
    zvals, i, inputs, labels, outputs, width=25, ax=None, styles=RUI_STYLES
):
    ## mds    print('zvals = ',zvals)
    ## mds    print(' i    =  ',i)
    ## mds    print('  inputs = ', inputs)
    ## mds    print('  labels = ', labels)
    ## mds    print('  outputs = ', outputs)
    x_bins = np.round(zvals[i - width : i + width] - 0.05, 2)
    ## mds    print('x_bins = ',x_bins)
    y_kernel = inputs.squeeze()[i - width : i + width] * 2500
    ## mds    print('y_kernel = ', y_kernel)
    y_target = labels.squeeze()[i - width : i + width]
    ## mds    print('y_target = ', y_target)
    y_predicted = outputs.squeeze()[i - width : i + width]
    ## mds    print('y_predicted = ', y_predicted)

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

        #ax_prob.set_ylim(0, max(0.8, 1.2 * max(y_predicted)))
        ax_prob.set_ylim(0, max(1.5, 1.2 * max(max(y_predicted),max(y_target))))
        ax_prob.set_ylabel("Probability", color=get_color(styles["predicted"]))
        if np.any(np.isnan(labels)):
            grey_y = np.isnan(y_target) * 0.2
            ax_prob.bar(x_bins, grey_y, width=0.1, **styles["masked"], label="Masked")

        ax_prob.legend(loc="upper right")

    return ax, ax_prob, y_kernel


def dual_train_plots(x=(), train=(), validation=(), eff=(), FP_rate=(), *, axs=None):

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax, lax = axs
    tax = lax.twinx()

    lines = dict()
    lines["train"], = ax.plot(x, train, "o-", label="Train")
    lines["val"], = ax.plot(x, validation, "o-", label="Validation")

    lines["eff"], = lax.plot(x, eff, "o-b", label="Eff")
    lines["fp"], = tax.plot(x, FP_rate, "o-r", label="FP rate")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cost")

    lax.set_xlabel("Epochs")
    lax.set_ylabel("Eff", color="b")
    tax.set_ylabel("FP rate", color="r")

    ax.set_yscale("log")
    ax.legend()
    lax.legend(loc="upper right")
    tax.legend(loc="lower left")
    return ax, tax, lax, lines


def replace_in_ax(ax, lines, x_values, y_values):
    lines.set_data(x_values, y_values)
    if np.max(y_values) > 0:
        ax.set_ylim(np.min(y_values) * 0.9, np.max(y_values) * 1.1)
    ax.set_xlim(-0.5, x_values[-1] + 0.5)

def six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2):
    
## takes ellipsoid axes in Cartesian coordinates and returns'
## six coefficients that describe the surface of the ellipsoid
## as
## (see https://math.stackexchange.com/questions/1865188/how-to-prove-the-parallel-projection-of-an-ellipsoid-is-an-ellipse)
##
##   A x^2 + B y^2 + C z^2 + 2(Dxy + Exz +Fyz) = 1
##
## note that this notation is NOT universal; the wikipedia article at
## https://en.wikipedia.org/wiki/Ellipse usses a similar, but different 
## in detail, notation.

  mag_1 = np.sqrt(np.dot(minorAxis_1,minorAxis_1))
  u1_hat = minorAxis_1/mag_1
  mag_2 = np.sqrt(np.dot(minorAxis_2,minorAxis_2))
  u2_hat = minorAxis_2/mag_2
  mag_3 = np.sqrt(np.dot(majorAxis,majorAxis))
  u3_hat = majorAxis/mag_3

  u1 = u1_hat/mag_1
  u2 = u2_hat/mag_2
  u3 = u3_hat/mag_3

  A = u1[0]*u1[0] + u2[0]*u2[0] + u3[0]*u3[0]
  B = u1[1]*u1[1] + u2[1]*u2[1] + u3[1]*u3[1]
  C = u1[2]*u1[2] + u2[2]*u2[2] + u3[2]*u3[2]
  D = u1[0]*u1[1] + u2[0]*u2[1] + u3[0]*u3[1]
  E = u1[2]*u1[0] + u2[2]*u2[0] + u3[2]*u3[0]
  F = u1[1]*u1[2] + u2[1]*u2[2] + u3[1]*u3[2]
  
  return A, B, C, D, E ,F

def xy_parallel_projection(A, B, C, D, E ,F):
    alpha_xy = C*A - E*E
    beta_xy  = C*B - F*F
    gamma_xy =  - 2*(C*D - E*F)  ## 200901 calculation
    delta_xy = C
    
    return alpha_xy, beta_xy, gamma_xy, delta_xy

def ellipse_parameters_for_plotting(alpha_xy, beta_xy, gamma_xy, delta_xy, A, C):

  sqrt_term = math.sqrt( (alpha_xy-beta_xy)**2 + gamma_xy**2)
  numerator_A = 2*(4*alpha_xy*beta_xy-gamma_xy**2)*delta_xy
  numerator_B_plus  = alpha_xy+beta_xy+sqrt_term
  numerator_B_minus = alpha_xy+beta_xy-sqrt_term
  denominator = (4*alpha_xy*beta_xy-gamma_xy**2)
  a = math.sqrt(numerator_A*numerator_B_plus)/denominator
  b = math.sqrt(numerator_A*numerator_B_minus)/denominator
  if (0 != gamma_xy):
    theta = math.atan( (beta_xy - alpha_xy - sqrt_term)/gamma_xy )
  if (0 == gamma_xy) & (A<C):
    theta = 0
  if (0 == gamma_xy) & (A>C):
    theta = 0.5*math.pi

## the Ellipse() method in matplotlib.patches wants andgles in degrees
  return a, b, 180.*theta/math.pi  