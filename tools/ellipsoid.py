'''
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
'''
import awkward as ak
import numpy as np
from tools.named_tuples_def import InputData_SV_OV_updated_and_PocaEllip_tuple, InputData_SV_OV_updated_and_PocaEllip_tuple_GNN
from utils.utilities import Timer

def Compute_tracks_ellipsoid(input_tuple, GNN_preprocess=False):
    
    ## ************************************************************    
    ## Get the tracks relevant information from the input tuple
    u1_x = getattr(input_tuple, "minor_axis1_x")
    u1_y = getattr(input_tuple, "minor_axis1_y")
    u1_z = getattr(input_tuple, "minor_axis1_z")
        
    u2_x = getattr(input_tuple, "minor_axis2_x")
    u2_y = getattr(input_tuple, "minor_axis2_y")
    u2_z = getattr(input_tuple, "minor_axis2_z")
    
    u3_x = getattr(input_tuple, "major_axis_x")
    u3_y = getattr(input_tuple, "major_axis_y")
    u3_z = getattr(input_tuple, "major_axis_z")
    
    ## Now get the magnitude of each vector component as |u| = u_x*u_x + u_y*u_y + u_z*u_z
    u1_mag = u1_x*u1_x + u1_y*u1_y + u1_z*u1_z
    u2_mag = u2_x*u2_x + u2_y*u2_y + u2_z*u2_z
    u3_mag = u3_x*u3_x + u3_y*u3_y + u3_z*u3_z
    
    ## And normalize each vector coordinates using the corresponding magnitudes
    u1_x_norm = u1_x / u1_mag
    u1_y_norm = u1_y / u1_mag
    u1_z_norm = u1_z / u1_mag
    
    u2_x_norm = u2_x / u2_mag
    u2_y_norm = u2_y / u2_mag
    u2_z_norm = u2_z / u2_mag
    
    u3_x_norm = u3_x / u3_mag
    u3_y_norm = u3_y / u3_mag
    u3_z_norm = u3_z / u3_mag
    
    ## Finally compute the six ellipsoid parameters derived from the unit vetors
    # First the "diagonal" elements
    A = u1_x_norm*u1_x_norm + u2_x_norm*u2_x_norm + u3_x_norm*u3_x_norm
    B = u1_y_norm*u1_y_norm + u2_y_norm*u2_y_norm + u3_y_norm*u3_y_norm
    C = u1_z_norm*u1_z_norm + u2_z_norm*u2_z_norm + u3_z_norm*u3_z_norm
    
    # then the "off diagonal" elements
    D = u1_x_norm*u1_y_norm + u2_x_norm*u2_y_norm + u3_x_norm*u3_y_norm
    E = u1_x_norm*u1_z_norm + u2_x_norm*u2_z_norm + u3_x_norm*u3_z_norm
    F = u1_y_norm*u1_z_norm + u2_y_norm*u2_z_norm + u3_y_norm*u3_z_norm
    
    if GNN_preprocess:
        return InputData_SV_OV_updated_and_PocaEllip_tuple_GNN(

            getattr(input_tuple, "pv_x"),
            getattr(input_tuple, "pv_y"),
            getattr(input_tuple, "pv_z"),
            getattr(input_tuple, "pv_ntracks"),
            getattr(input_tuple, "pv_cat"),
            getattr(input_tuple, "pv_key"),

            getattr(input_tuple, "sv_x"),
            getattr(input_tuple, "sv_y"),
            getattr(input_tuple, "sv_z"),
            getattr(input_tuple, "sv_ntracks"),

            getattr(input_tuple, "poca_x"),               
            getattr(input_tuple, "poca_y"),              
            getattr(input_tuple, "poca_z"),              

            A,
            B,
            C,
            D,
            E,
            F,

            getattr(input_tuple, "major_axis_x"),        
            getattr(input_tuple, "major_axis_y"),        
            getattr(input_tuple, "major_axis_z"),        
            getattr(input_tuple, "minor_axis1_x"),       
            getattr(input_tuple, "minor_axis1_y"),       
            getattr(input_tuple, "minor_axis1_z"),       
            getattr(input_tuple, "minor_axis2_x"),       
            getattr(input_tuple, "minor_axis2_y"),       
            getattr(input_tuple, "minor_axis2_z"),      

            getattr(input_tuple, "recon_pv_key"), 
            getattr(input_tuple, "recon_cat"), 
            getattr(input_tuple, "recon_ov_x"), 
            getattr(input_tuple, "recon_ov_y"), 
            getattr(input_tuple, "recon_ov_z"), 
            
            getattr(input_tuple, "recon_tx"), 
            getattr(input_tuple, "recon_ty"),    
        ) 
        
    else:
        return InputData_SV_OV_updated_and_PocaEllip_tuple(

            getattr(input_tuple, "pv_x"),
            getattr(input_tuple, "pv_y"),
            getattr(input_tuple, "pv_z"),
            getattr(input_tuple, "pv_ntracks"),
            getattr(input_tuple, "pv_cat"),
            getattr(input_tuple, "pv_key"),

            getattr(input_tuple, "sv_x"),
            getattr(input_tuple, "sv_y"),
            getattr(input_tuple, "sv_z"),
            getattr(input_tuple, "sv_ntracks"),

            getattr(input_tuple, "poca_x"),               
            getattr(input_tuple, "poca_y"),              
            getattr(input_tuple, "poca_z"),              

            A,
            B,
            C,
            D,
            E,
            F,

            getattr(input_tuple, "major_axis_x"),        
            getattr(input_tuple, "major_axis_y"),        
            getattr(input_tuple, "major_axis_z"),        
            getattr(input_tuple, "minor_axis1_x"),       
            getattr(input_tuple, "minor_axis1_y"),       
            getattr(input_tuple, "minor_axis1_z"),       
            getattr(input_tuple, "minor_axis2_x"),       
            getattr(input_tuple, "minor_axis2_y"),       
            getattr(input_tuple, "minor_axis2_z"),      

            getattr(input_tuple, "recon_pv_key"), 
            getattr(input_tuple, "recon_cat"), 
            getattr(input_tuple, "recon_ov_x"), 
            getattr(input_tuple, "recon_ov_y"), 
            getattr(input_tuple, "recon_ov_z"), 
            
            getattr(input_tuple, "recon_tx"), 
            getattr(input_tuple, "recon_ty"), 

            getattr(input_tuple, "IP_KDE"),
            getattr(input_tuple, "IP_KDE_xMax"),
            getattr(input_tuple, "IP_KDE_yMax"),

            getattr(input_tuple, "poca_KDE_A"),
            getattr(input_tuple, "poca_KDE_A_xMax"),
            getattr(input_tuple, "poca_KDE_A_yMax"),
            getattr(input_tuple, "poca_KDE_B"),
            getattr(input_tuple, "poca_KDE_B_xMax"),
            getattr(input_tuple, "poca_KDE_B_yMax"),        
        ) 
    