"""@package docstring
Documentation for this module.

from https://math.stackexchange.com/questions/1865188/how-to-prove-the-parallel-projection-of-an-ellipsoid-is-an-ellipse

Up to translation, a general ellipsoid can be written in the form

  𝐴$𝑥^2$ +𝐵$𝑦^2$+𝐶$𝑧^2$+2(𝐷𝑥𝑦+𝐸𝑥𝑧+𝐹𝑦𝑧)=1
  
for some positive-definite coefficient matrix 

$$
\left(\begin{array}{ccc}
A & D & E \\
D & B & F \\
E & F & C \\
\end{array}\right)
$$

1. For definiteness, project the ellipsoid to the (𝑥,𝑦)-plane along the 𝑧-axis, and call the image the shadow. A point 𝑝=(𝑥,𝑦,𝑧) on the ellipsoid projects to the boundary of the shadow if and only if the tangent plane to the ellipsoid at 𝑝 is parallel to the 𝑧-axis, if and only if 

$$ 
0 = \frac{\partial}{\partial z} \left ( A x^2 + B y^2 + C z^2 + s ( D x y + E x z + F y z ) \right ) \cdot ( p - p_0) \, ,
$$

Our ellipsoids have minor and major axes that (generally) are not parallel to the usual $ x $, $ y $, and $ z $ axes.  Let's label them as the $ u_1 $, $ u_2 $ and $ u_3 $ axes:

The surface of the ellipsoid is now defined by the equation
$$ \left ( \frac{\vec x \cdot \hat u_1}{a} \right )^2 +  
   \left ( \frac{\vec x \cdot \hat u_2}{b} \right )^2 +
   \left ( \frac{\vec x \cdot \hat u_3}{c} \right )^2  = 1
$$
where 
$$ a = | minorAxis_1 | $$
$$ b = | minorAxis_2 | $$
$$ c = | majorAxis | $$

This leads us to define
$$ \vec u_1 = \hat u1 / | minorAxis_1 | $$
$$ \vec u_2 = \hat u2 / | minorAxis_2 | $$
$$ \vec u_3 = \hat u_3 / | majorAxis | $$


With this notation, the equation for the ellipse becomes

$$ (\vec x \cdot \vec u_1)^2 + ( \vec x \cdot u_2)^2 +
   (\vec x \cdot \vec u_3)^2 = 1
   $$
   
  and writing the corrdinates of a point as $ \vec x = (x, y, z) $ it becomes
  
  $$ (x \, u_{1x} + y \, u _{1y} + z \, u_{1z})^2 +
     (x \, u_{2x} + y \, u _{2y} + z \, u_{2z})^2 +
     (x \, u_{3x} + y \, u _{3y} + z \, u_{3z})^2  = 1
  $$
  
  or
  
  $$ \begin{array}{ccc}
    (u_{1x}^2 + u_{2x}^2 + u_{3x}^2) \, x^2 & + & \\
    (u_{1y}^2 + u_{2y}^2 + u_{3y}^2) \, y^2 & + & \\
    (u_{1z}^2 + u_{2z}^2 + u_{3z}^2) \, z^2 & + & \\
    2 \left ( u_{1x}u_{1y} + u_{2x}u_{2y} + u_{3x}u_{3y} \right ) xy & + & \\
    2 \left ( u_{1y}u_{1z} + u_{2y}u_{2z} + u_{3y}u_{3z} \right ) yz & + & \\
    2 \left ( u_{1z}u_{1x} + u_{2z}u_{2x} + u_{3z}u_{3x} \right ) zx & = & 1
     \end{array}
  $$
  
  from which we can extract the forms of $ A $, $ B $, etc.:
  
  $$ \begin{array}{ccc}
       A & = & u_{1x}^2 + u_{2x}^2 + u_{3x}^2 \\
       B & = & u_{1y}^2 + u_{2y}^2 + u_{3y}^2 \\
       C & = & u_{1z}^2 + u_{2z}^2 + u_{3z}^2  \\
       D & = & u_{1x}u_{1y} + u_{2x}u_{2y} + u_{3x}u_{3y} \\
       E & = & u_{1z}u_{1x} + u_{2z}u_{2x} + u_{3z}u_{3x} \\
       F & = & u_{1y}u_{1z} + u_{2y}u_{2z} + u_{3y}u_{3z} \\
     \end{array}
  $$
"""
import numpy as np

def six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2):
    
## takes ellipsoid axes in Cartesian coordinates and returns'
## six coefficients that describe the surface of the ellipsoid as
## (see https://math.stackexchange.com/questions/1865188/how-to-prove-the-parallel-projection-of-an-ellipsoid-is-an-ellipse)
##
##   A x^2 + B y^2 + C z^2 + 2(Dxy + Exz +Fyz) = 1
##
## note that this notation is NOT universal; the wikipedia article at
## https://en.wikipedia.org/wiki/Ellipse uses a similar, but different 
## in detail, notation.

#
  print("have entered six_ellipsoid_parameters")
  print("  ")
  print(" ")
## 
##  majorAxis, minorAxis_1, and minorAxis_2 are jagged arrrays --
##  each event has a variable number of tracks, and each track
##  has three entries corresponding to the lengths of the 
##  x, y, and z components of the axes.  
##  The "usual" numpy methods for manipulating these do not
##  always work as these *assume* fixed array structures
##  It *appears* the hacks used here suffice 

##  first for each track, for each axis, find the length squared
  mag_3_sq = np.multiply(majorAxis[:,0],majorAxis[:,0]) 
  mag_3_sq = mag_3_sq + np.multiply(majorAxis[:,1],majorAxis[:,1]) 
  mag_3_sq = mag_3_sq + np.multiply(majorAxis[:,2],majorAxis[:,2])

  mag_2_sq = np.multiply(minorAxis_2[:,0],minorAxis_2[:,0]) 
  mag_2_sq = mag_2_sq + np.multiply(minorAxis_2[:,1],minorAxis_2[:,1]) 
  mag_2_sq = mag_2_sq + np.multiply(minorAxis_2[:,2],minorAxis_2[:,2])

  mag_1_sq = np.multiply(minorAxis_1[:,0],minorAxis_1[:,0]) 
  mag_1_sq = mag_1_sq + np.multiply(minorAxis_1[:,1],minorAxis_1[:,1]) 
  mag_1_sq = mag_1_sq + np.multiply(minorAxis_1[:,2],minorAxis_1[:,2])

  nEvts = len(majorAxis)
  print("  nEvts = ",nEvts)

## by creating u1, u2, and u3 as copies of the axes,
## they acquire the correct array structure
  u1 = minorAxis_1
  u2 = minorAxis_2 
  u3 = majorAxis

##  this is an ugly, brute force hack, but it
##  seems to work
  for iEvt in range(nEvts):
    nTrks = len(u3[iEvt][0])
    if (iEvt < 10):
      print(" iEvt, nTrks = ", iEvt, nTrks)
    for iTrk in range(nTrks):
      u3[iEvt][0][iTrk] = u3[iEvt][0][iTrk]/mag_3_sq[iEvt][iTrk]
      u3[iEvt][1][iTrk] = u3[iEvt][1][iTrk]/mag_3_sq[iEvt][iTrk]
      u3[iEvt][2][iTrk] = u3[iEvt][2][iTrk]/mag_3_sq[iEvt][iTrk]

      u2[iEvt][0][iTrk] = u2[iEvt][0][iTrk]/mag_2_sq[iEvt][iTrk]
      u2[iEvt][1][iTrk] = u2[iEvt][1][iTrk]/mag_2_sq[iEvt][iTrk]
      u2[iEvt][2][iTrk] = u2[iEvt][2][iTrk]/mag_2_sq[iEvt][iTrk]

      u1[iEvt][0][iTrk] = u1[iEvt][0][iTrk]/mag_1_sq[iEvt][iTrk]
      u1[iEvt][1][iTrk] = u1[iEvt][1][iTrk]/mag_1_sq[iEvt][iTrk]
      u1[iEvt][2][iTrk] = u1[iEvt][2][iTrk]/mag_1_sq[iEvt][iTrk]

##  because u1, u2, and u3 have the original axis structures,
##  it seems we can use the standard numpy method for these
##  calculations
  A = u1[:,0]*u1[:,0] + u2[:,0]*u2[:,0] + u3[:,0]*u3[:,0]
  B = u1[:,1]*u1[:,1] + u2[:,1]*u2[:,1] + u3[:,1]*u3[:,1]
  C = u1[:,2]*u1[:,2] + u2[:,2]*u2[:,2] + u3[:,2]*u3[:,2]

  D = np.multiply(u1[:,0],u1[:,1]) + np.multiply(u2[:,0],u2[:,1]) + np.multiply(u3[:,0],u3[:,1])
  E = np.multiply(u1[:,2],u1[:,0]) + np.multiply(u2[:,2],u2[:,0]) + np.multiply(u3[:,2],u3[:,0])
  F = np.multiply(u1[:,1],u1[:,2]) + np.multiply(u2[:,1],u2[:,2]) + np.multiply(u3[:,1],u3[:,2])

## mds   D = u1[:,0]*u1[:,1] + u2[:,0]*u2[:,1] + u3[:,0]*u3[:,1]
## mds   E = u1[:,2]*u1[:,0] + u2[:,2]*u2[:,0] + u3[:,2]*u3[:,0]
## mds   F = u1[:,1]*u1[:,2] + u2[:,1]*u2[:,2] + u3[:,1]*u3[:,2]

## as a sanity check, let's print out some of the details inputs
## and outputs so we can check them by hand


  
  return A, B, C, D, E ,F
