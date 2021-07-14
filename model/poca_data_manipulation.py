import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap as cmap
from matplotlib.cm import ScalarMappable
import numpy as np
import math
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import h5py
try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward
concatenate = awkward.concatenate

def gaussian(x, pos, width):
    height = 1/(width*math.sqrt(2*math.pi))
    return height*np.exp(-(x-pos)**2 / (2*width**2))


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
    
    # project onto x-y plane
    # change order to (C, A, B, E, D, F) for z-x plane (z is horizontal)
    # change order to (C, A, B, E, D, F) for z-y plane (z is horizontal)
    
    
    alpha_xy = C*A - E*E
    beta_xy  = C*B - F*F
    gamma_xy =  - 2*(C*D - E*F)  ## 200901 calculation
    delta_xy = C
    
    return alpha_xy, beta_xy, gamma_xy, delta_xy 
    # to get ellipse parameters for plotting, enter these ^^^ params into next method


def ellipse_parameters_for_plotting(alpha_xy, beta_xy, gamma_xy, delta_xy, A, C):

    # returns height, width, and angle for plotting using matplotlib's Ellipse class
    
    sqrt_term = math.sqrt( (alpha_xy-beta_xy)**2 + gamma_xy**2)
    numerator_A = 2*(4*alpha_xy*beta_xy-gamma_xy**2)*delta_xy
    numerator_B_plus  = alpha_xy+beta_xy+sqrt_term
    numerator_B_minus = alpha_xy+beta_xy-sqrt_term
    denominator = (4*alpha_xy*beta_xy-gamma_xy**2)
    
    if denominator == 0:
        a = 0
        b = 0
        
    if not denominator == 0:
        a = math.sqrt(np.abs(numerator_A*numerator_B_plus))/denominator
        b = math.sqrt(np.abs(numerator_A*numerator_B_minus))/denominator
        
        
    if (0 != gamma_xy):
        theta = math.atan( (beta_xy - alpha_xy - sqrt_term)/gamma_xy )
    if (0 == gamma_xy) & (A<C):
        theta = 0
    if (0 == gamma_xy) & (A>C):
        theta = 0.5*math.pi

    return a, b, 180.*theta/math.pi  


# plots ellipses between min_z and max_z using pocas dictionary (see collect_poca method)
def getEllipses(event, pocas, min_z, max_z, axes, fig):
    
    valid_ellipses = []
    
    trackCount = 0
    
    #initialize lists needed to plot ellipsoids
    ellsXY = []
    ellsZX = []
    ellsZY = []
    zpoca_vals = []
    alphas = []
    
    cm = cmap("cool")
    for j in range(len(pocas["x"]["major_axis"][event])):
        #create three vectors corresponding to major axis and the two minor axes
        major_axis = [pocas["x"]["major_axis"][event][j],
                      pocas["y"]["major_axis"][event][j],
                      pocas["z"]["major_axis"][event][j]]
        minor_axis1 = [pocas["x"]["minor_axis1"][event][j],
                       pocas["y"]["minor_axis1"][event][j],
                       pocas["z"]["minor_axis1"][event][j]]
        minor_axis2 = [pocas["x"]["minor_axis2"][event][j],
                       pocas["y"]["minor_axis2"][event][j],
                       pocas["z"]["minor_axis2"][event][j]]
    
        #calculate magnitude of major axis (used to determine alpha of ellipsoid)
        major_axis_mag = np.sqrt(major_axis[0]**2 + major_axis[1]**2 + major_axis[2]**2)
    
        #determine color of ellipsoid (according to depth in z-axis)
        color_scaling = (pocas["z"]["poca"][event][j] - min_z)/(max_z-min_z)
        color = cm(color_scaling)
        
        #calculate ellipsoid parameters from three-vectors
        A,B,C,D,E,F = six_ellipsoid_parameters(major_axis, minor_axis1, minor_axis2)
                
        #calculate parameters needed for x-y projection of ellipsoid
        alpha_xy, beta_xy, gamma_xy, delta_xy = xy_parallel_projection(A, B, C, D, E ,F)
        #calculate plotting parameters of ellipsoid
        a_xy, b_xy, theta_xy = ellipse_parameters_for_plotting(alpha_xy,beta_xy,gamma_xy,delta_xy,A,C)
                
        #repeat for x,z
        alpha_zx, beta_zx, gamma_zx, delta_zx = xy_parallel_projection(C, A, B, E, D, F)
        a_zx, b_zx, theta_zx = ellipse_parameters_for_plotting(alpha_zx,beta_zx,gamma_zx,delta_zx,C,B)
                
        #repeat for y,z
        alpha_zy, beta_zy, gamma_zy, delta_zy = xy_parallel_projection(C, B, A, F, D, E)
        a_zy, b_zy, theta_zy = ellipse_parameters_for_plotting(alpha_zy,beta_zy,gamma_zy,delta_zy,C,A)
        
        #create Ellipse objects corresponding to track
        thisEllipseXY = Ellipse([pocas["x"]["poca"][event][j], pocas["y"]["poca"][event][j]],
                                a_xy, b_xy, theta_xy, color=color)
        thisEllipseZX = Ellipse([pocas["z"]["poca"][event][j], pocas["x"]["poca"][event][j]],
                                a_zx, b_zx, theta_zx, color=color)
        thisEllipseZY = Ellipse([pocas["z"]["poca"][event][j], pocas["y"]["poca"][event][j]],
                                a_zy, b_zy, theta_zy, color=color)
        
        #add ellipse to list if it falls in the z-range
        if (pocas["z"]["poca"][event][j] >= min_z and pocas["z"]["poca"][event][j] <= max_z):
            
            valid_ellipses.append(j)
            
            ellsXY.append(thisEllipseXY)
            ellsZX.append(thisEllipseZX)
            ellsZY.append(thisEllipseZY)
            zpoca_vals.append(pocas["z"]["poca"][event][j])
                    
            #calculate opacity of ellipse
            alpha = 0.3*major_axis_mag
            alpha = min(alpha,1)
            alpha = 1-alpha
            alpha = 0.7*max(alpha, 0.05)
            alphas.append(alpha)
                    
            trackCount += 1
            
    #sort ellipses according to depth in z-axis (so that we don't plot in a random order, makes visualization easier)
    ellsXY = [e for _,e in sorted(zip(zpoca_vals,ellsXY), reverse = True)]
    ellsZX = [e for _,e in sorted(zip(zpoca_vals,ellsZX), reverse = True)]
    ellsZY = [e for _,e in sorted(zip(zpoca_vals,ellsZY), reverse = True)]
    alphas = [alpha for _,alpha in sorted(zip(zpoca_vals,alphas), reverse = True)]
    
    if trackCount > 30:
        alphas = np.multiply(alphas,0.5)
        
    #plot ellipses
    for j in range(len(alphas)):
        axes[0].add_artist(ellsXY[j])
        ellsXY[j].set_clip_box(axes[0].bbox)
        ellsXY[j].set_alpha(alphas[j])
                
        axes[1].add_artist(ellsZX[j])
        ellsZX[j].set_clip_box(axes[1].bbox)
        ellsZX[j].set_alpha(alphas[j])
                
        axes[2].add_artist(ellsZY[j])
        ellsZY[j].set_clip_box(axes[2].bbox)
        ellsZY[j].set_alpha(alphas[j])
        
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z), cmap=cm),
                 ax=axes[0], label='Position in z')
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z), cmap=cm),
                 ax=axes[1], label='Position in z')
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z), cmap=cm),
                 ax=axes[2], label='Position in z')
    
    
    return valid_ellipses
    
# takes hdf5 file and collects poca information, returning a dictionary with keys corresponding to each direction
def collect_poca(*files):
    
    #initialize lists
    pocax_list = []
    pocay_list = []
    pocaz_list = []

    majoraxisx_list = []
    majoraxisy_list = []
    majoraxisz_list = []

    minoraxis1x_list = []
    minoraxis1y_list = []
    minoraxis1z_list = []
    minoraxis2x_list = []
    minoraxis2y_list = []
    minoraxis2z_list = []

    
    #iterate through all files
    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with h5py.File(XY_file, mode="r") as XY:

            #print keys in current hdf5 file
            print(XY.keys())

            afile = awkward.hdf5(XY)

            #append to appropriate lists
            pocax_list.append(afile["poca_x"])
            pocay_list.append(afile["poca_y"])
            pocaz_list.append(afile["poca_z"])

            majoraxisx_list.append(afile["major_axis_x"])
            majoraxisy_list.append(afile["major_axis_y"])
            majoraxisz_list.append(afile["major_axis_z"])

            minoraxis1x_list.append(afile["minor_axis1_x"])
            minoraxis1y_list.append(afile["minor_axis1_y"])
            minoraxis1z_list.append(afile["minor_axis1_z"])

            minoraxis2x_list.append(afile["minor_axis2_x"])
            minoraxis2y_list.append(afile["minor_axis2_y"])
            minoraxis2z_list.append(afile["minor_axis2_z"])
    
    #construct pocas dictionary
    pocas = {}
    pocas["x"] = {"poca": concatenate(pocax_list),
                  "major_axis": concatenate(majoraxisx_list),
                  "minor_axis1": concatenate(minoraxis1x_list),
                  "minor_axis2": concatenate(minoraxis2x_list)}

    pocas["y"] = {"poca": concatenate(pocay_list),
                  "major_axis": concatenate(majoraxisy_list),
                  "minor_axis1": concatenate(minoraxis1y_list),
                  "minor_axis2": concatenate(minoraxis2y_list)}

    pocas["z"] = {"poca": concatenate(pocaz_list),
                  "major_axis": concatenate(majoraxisz_list),
                  "minor_axis1": concatenate(minoraxis1z_list),
                  "minor_axis2": concatenate(minoraxis2z_list)}

    return pocas


def combine_ellipsoids(centers, major_axes, minor_axes1, minor_axes2):
        
    # combine error ellipsoids to approximate PV
        
    num = len(centers[1,:])
#     Cinv = np.zeros([3,3])
#     mu = np.zeros([3,1])
    
    volumes = np.zeros(num)
    maj_mag = np.zeros(num)
    min1_mag = np.zeros(num)
    min2_mag = np.zeros(num)
    
        
    for i in range(num):
        
        center = centers[:,i].reshape((3,1))
        major_axis = major_axes[:,i].reshape((3,1))
        minor_axis1 = minor_axes1[:,i].reshape((3,1))
        minor_axis2 = minor_axes2[:,i].reshape((3,1))
        
        volume = np.linalg.norm(major_axis) * np.linalg.norm(minor_axis1) * np.linalg.norm(minor_axis2)
        volumes[i] = 1/volume
        maj_mag[i] = 1/np.linalg.norm(major_axis)
        min1_mag[i] = 1/np.linalg.norm(minor_axis1)
        min2_mag[i] = 1/np.linalg.norm(minor_axis2)
            
#         #calculate matrix elements
#         A,B,C,D,E,F = six_ellipsoid_parameters(major_axis.reshape((3,)), 
#                                                minor_axis1.reshape((3,)), 
#                                                minor_axis2.reshape((3,)))

            
        #construct covariance matrix
#         cov = np.matrix([[A,D,E],
#                         [D,B,F],
#                         [E,F,C]], dtype=np.float64)
            
#         cov_inv = np.linalg.inv(cov)
            
#         Cinv += cov_inv
#         mu += np.matmul(cov_inv, center)
        
    #compute center and cov matrix of combined ellipse
#     C = np.linalg.inv(Cinv)
#     center = np.matmul(C, mu)
        
#     #convert to ellipse form
#     evals, evecs = np.linalg.eig(C) # get eigenvalues and eigenvectors
#     major_axis = evals[0] * evecs[0]
#     minor_axis1 = evals[1] * evecs[1]
#     minor_axis2 = evals[2] * evecs[2]
    
    centerx = np.average(centers[0,:], weights = volumes)
    centery = np.average(centers[1,:], weights = volumes)
    centerz = np.average(centers[2,:], weights = volumes)
    center = np.array([centerx, centery, centerz])
    
    major_axisx = np.average(major_axes[0,:], weights = maj_mag)
    major_axisy = np.average(major_axes[1,:], weights = maj_mag)
    major_axisz = np.average(major_axes[2,:], weights = maj_mag)
    major_axis = np.array([major_axisx, major_axisy, major_axisz])
    
    minor_axis1x = np.average(minor_axes1[0,:], weights = min1_mag)
    minor_axis1y = np.average(minor_axes1[1,:], weights = min1_mag)
    minor_axis1z = np.average(minor_axes1[2,:], weights = min1_mag)
    minor_axis1 = np.array([minor_axis1x, minor_axis1y, minor_axis1z])
    
    minor_axis2x = np.average(minor_axes2[0,:], weights = min2_mag)
    minor_axis2y = np.average(minor_axes2[1,:], weights = min2_mag)
    minor_axis2z = np.average(minor_axes2[2,:], weights = min2_mag)
    minor_axis2 = np.array([minor_axis2x, minor_axis2y, minor_axis2z])
    
        
    return center, major_axis, minor_axis1, minor_axis2
            
    
def plot_combined_ellipse(center, major_axis, minor_axis1, minor_axis2, fig, axs):
    
    # this method could also be used to plot any single ellipse, with projections on xy, zx, and zy planes
    
    A,B,C,D,E,F = six_ellipsoid_parameters(major_axis, minor_axis1, minor_axis2)
    
    alpha_xy, beta_xy, gamma_xy, delta_xy = xy_parallel_projection(A, B, C, D, E ,F)
    a_xy, b_xy, theta_xy = ellipse_parameters_for_plotting(alpha_xy, beta_xy, gamma_xy, delta_xy,A,C)
    
    alpha_zx, beta_zx, gamma_zx, delta_zx = xy_parallel_projection(C, A, B, E, D, F)
    a_zx, b_zx, theta_zx = ellipse_parameters_for_plotting(alpha_zx,beta_zx,gamma_zx,delta_zx,C,B)
    
    alpha_zy, beta_zy, gamma_zy, delta_zy = xy_parallel_projection(C, B, A, F, D, E)
    a_zy, b_zy, theta_zy = ellipse_parameters_for_plotting(alpha_zy,beta_zy,gamma_zy,delta_zy,C,A)
    
    thisEllipseXY = Ellipse([center[0], center[1]],
                            a_xy, b_xy, theta_xy, color='g')
    thisEllipseZX = Ellipse([center[2], center[0]],
                            a_zx, b_zx, theta_zx, color='g')
    thisEllipseZY = Ellipse([center[2], center[1]],
                            a_zy, b_zy, theta_zy, color='g')
    
    axs[0].add_artist(thisEllipseXY)
    thisEllipseXY.set_clip_box(axs[0].bbox)
    thisEllipseXY.set_alpha(0.5)
                
    axs[1].add_artist(thisEllipseZX)
    thisEllipseZX.set_clip_box(axs[1].bbox)
    thisEllipseZX.set_alpha(0.5)
                
    axs[2].add_artist(thisEllipseZY)
    thisEllipseZY.set_clip_box(axs[2].bbox)
    thisEllipseZY.set_alpha(0.5)


def plotEllipses(centers, major_axes, minor_axes1, minor_axes2, fig, axes):
    
    # more general method for getEllipses (does not use pocas dictionary)
    
    trackCount = 0
    
    min_z = min(centers[2,:]) - 0.5
    max_z = max(centers[2,:]) + 0.5
    
    #initialize lists needed to plot ellipsoids
    ellsXY = []
    ellsZX = []
    ellsZY = []
    alphas = []
    
    cm = cmap("cool")
    
    num = len(centers[1,:])
    
    for i in range(num):
        
        center = centers[:,i]
        major_axis = major_axes[:,i]
        minor_axis1 = minor_axes1[:,i]
        minor_axis2 = minor_axes2[:,i]
        
        major_axis_mag = np.sqrt(major_axis[0]**2 + major_axis[1]**2 + major_axis[2]**2)
        
        color_scaling = (center[2] - min_z)/(max_z-min_z)
        color = cm(color_scaling)
        
        #calculate 6 ellipsoid parameters
        A,B,C,D,E,F = six_ellipsoid_parameters(major_axis, minor_axis1, minor_axis2)
        
        #calculate plotting parameters for projected ellipsoids
        alpha_xy, beta_xy, gamma_xy, delta_xy = xy_parallel_projection(A, B, C, D, E ,F)
        a_xy, b_xy, theta_xy = ellipse_parameters_for_plotting(alpha_xy,beta_xy,gamma_xy,delta_xy,A,C)
        alpha_zx, beta_zx, gamma_zx, delta_zx = xy_parallel_projection(C, A, B, E, D, F)
        a_zx, b_zx, theta_zx = ellipse_parameters_for_plotting(alpha_zx,beta_zx,gamma_zx,delta_zx,C,B)
        alpha_zy, beta_zy, gamma_zy, delta_zy = xy_parallel_projection(C, B, A, F, D, E)
        a_zy, b_zy, theta_zy = ellipse_parameters_for_plotting(alpha_zy,beta_zy,gamma_zy,delta_zy,C,A)
        
        #create Ellipse objects corresponding to track
        thisEllipseXY = Ellipse([center[0], center[1]],
                                a_xy, b_xy, theta_xy, color=color)
        thisEllipseZX = Ellipse([center[2], center[0]],
                                a_zx, b_zx, theta_zx, color=color)
        thisEllipseZY = Ellipse([center[2], center[1]],
                                a_zy, b_zy, theta_zy, color=color)
        
        ellsXY.append(thisEllipseXY)
        ellsZX.append(thisEllipseZX)
        ellsZY.append(thisEllipseZY)
        
        alpha = 0.3*major_axis_mag
        alpha = min(alpha,1)
        alpha = 1-alpha
        alpha = 0.7*max(alpha, 0.05)
        alphas.append(alpha)
        
        trackCount += 1
        
    #sort ellipses according to depth in z-axis (so that we don't plot in a random order, makes visualization easier)
    ellsXY = [e for _,e in sorted(zip(centers[2,:].tolist(),ellsXY), reverse = True)]
    ellsZX = [e for _,e in sorted(zip(centers[2,:].tolist(),ellsZX), reverse = True)]
    ellsZY = [e for _,e in sorted(zip(centers[2,:].tolist(),ellsZY), reverse = True)]
    alphas = [alpha for _,alpha in sorted(zip(centers[2,:].tolist(),alphas), reverse = True)]
    
    if trackCount > 30:
        alphas = np.multiply(alphas,0.5)
        
    #plot ellipses
    for j in range(len(alphas)):
        axes[0].add_artist(ellsXY[j])
        ellsXY[j].set_clip_box(axes[0].bbox)
        ellsXY[j].set_alpha(alphas[j])
                
        axes[1].add_artist(ellsZX[j])
        ellsZX[j].set_clip_box(axes[1].bbox)
        ellsZX[j].set_alpha(alphas[j])
                
        axes[2].add_artist(ellsZY[j])
        ellsZY[j].set_clip_box(axes[2].bbox)
        ellsZY[j].set_alpha(alphas[j])
        
        
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z), cmap=cm),
                 ax=axes[0], label='Position in z')
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z), cmap=cm),
                 ax=axes[1], label='Position in z')
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z), cmap=cm),
                 ax=axes[2], label='Position in z')
    
    