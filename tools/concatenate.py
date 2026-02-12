import awkward as ak
from tools.named_tuples_def import *

## ========================================================================================
## CONCATENATION TOOLS
## ========================================================================================
def Concatenate_tuples(tuple_type, tuples_list):
    """
    Method to concatenate namedtuple containers from the multiple input files.
    Takes as input a list of tuples and the "type" of namedtuple considered 
    and returns a single namedtuple with the corresponding data concatenated.
    """
    if tuple_type=="InputData_tuple":
        if not len(tuples_list[0]._fields)==len(InputData_tuple._fields):
            print("Argument for 'tuple_type': %s and input list of tuples do not match!"%(tuple_type))
            print("'tuple_type': %s has %s fields, while input tuples have %s"%(len(tuples_list[0]._fields),
                                                                                len(InputData_tuple._fields)))
            exit(0)
                
        return InputData_tuple(
            ak.concatenate([i.pv_x            for i in tuples_list]),
            ak.concatenate([i.pv_y            for i in tuples_list]),
            ak.concatenate([i.pv_z            for i in tuples_list]),
            ak.concatenate([i.pv_ntracks      for i in tuples_list]),
            ak.concatenate([i.pv_cat          for i in tuples_list]),
            ak.concatenate([i.pv_key          for i in tuples_list]),

            ak.concatenate([i.sv_x            for i in tuples_list]),
            ak.concatenate([i.sv_y            for i in tuples_list]),
            ak.concatenate([i.sv_z            for i in tuples_list]),
            ak.concatenate([i.sv_ntracks      for i in tuples_list]),
            ak.concatenate([i.sv_cat          for i in tuples_list]),
            ak.concatenate([i.sv_key          for i in tuples_list]),
            ak.concatenate([i.sv_pv_key       for i in tuples_list]),

            ak.concatenate([i.poca_x          for i in tuples_list]),
            ak.concatenate([i.poca_y          for i in tuples_list]),
            ak.concatenate([i.poca_z          for i in tuples_list]),
            ak.concatenate([i.major_axis_x    for i in tuples_list]),
            ak.concatenate([i.major_axis_y    for i in tuples_list]),
            ak.concatenate([i.major_axis_z    for i in tuples_list]),
            ak.concatenate([i.minor_axis1_x   for i in tuples_list]),
            ak.concatenate([i.minor_axis1_y   for i in tuples_list]),
            ak.concatenate([i.minor_axis1_z   for i in tuples_list]),
            ak.concatenate([i.minor_axis2_x   for i in tuples_list]),
            ak.concatenate([i.minor_axis2_y   for i in tuples_list]),
            ak.concatenate([i.minor_axis2_z   for i in tuples_list]),

            ak.concatenate([i.recon_pv_key    for i in tuples_list]),
            ak.concatenate([i.recon_ov_key    for i in tuples_list]),
            
            ak.concatenate([i.recon_tx        for i in tuples_list]),
            ak.concatenate([i.recon_ty        for i in tuples_list]),
            
            ak.concatenate([i.IP_KDE          for i in tuples_list]),
            ak.concatenate([i.IP_KDE_xMax     for i in tuples_list]),
            ak.concatenate([i.IP_KDE_yMax     for i in tuples_list]),
            
            ak.concatenate([i.poca_KDE_A      for i in tuples_list]),
            ak.concatenate([i.poca_KDE_A_xMax for i in tuples_list]),
            ak.concatenate([i.poca_KDE_A_yMax for i in tuples_list]),
            ak.concatenate([i.poca_KDE_B      for i in tuples_list]),
            ak.concatenate([i.poca_KDE_B_xMax for i in tuples_list]),
            ak.concatenate([i.poca_KDE_B_yMax for i in tuples_list]),            
        )
    elif tuple_type=="InputData_tuple_GNN":
        if not len(tuples_list[0]._fields)==len(InputData_tuple_GNN._fields):
            print("Argument for 'tuple_type': %s and input list of tuples do not match!"%(tuple_type))
            print("'tuple_type': %s has %s fields, while input tuples have %s"%(len(tuples_list[0]._fields),
                                                                                len(InputData_tuple_GNN._fields)))
            exit(0)
                
        return InputData_tuple_GNN(
            ak.concatenate([i.pv_x            for i in tuples_list]),
            ak.concatenate([i.pv_y            for i in tuples_list]),
            ak.concatenate([i.pv_z            for i in tuples_list]),
            ak.concatenate([i.pv_ntracks      for i in tuples_list]),
            ak.concatenate([i.pv_cat          for i in tuples_list]),
            ak.concatenate([i.pv_key          for i in tuples_list]),

            ak.concatenate([i.sv_x            for i in tuples_list]),
            ak.concatenate([i.sv_y            for i in tuples_list]),
            ak.concatenate([i.sv_z            for i in tuples_list]),
            ak.concatenate([i.sv_ntracks      for i in tuples_list]),
            ak.concatenate([i.sv_cat          for i in tuples_list]),
            ak.concatenate([i.sv_key          for i in tuples_list]),
            ak.concatenate([i.sv_pv_key       for i in tuples_list]),

            ak.concatenate([i.poca_x          for i in tuples_list]),
            ak.concatenate([i.poca_y          for i in tuples_list]),
            ak.concatenate([i.poca_z          for i in tuples_list]),
            ak.concatenate([i.major_axis_x    for i in tuples_list]),
            ak.concatenate([i.major_axis_y    for i in tuples_list]),
            ak.concatenate([i.major_axis_z    for i in tuples_list]),
            ak.concatenate([i.minor_axis1_x   for i in tuples_list]),
            ak.concatenate([i.minor_axis1_y   for i in tuples_list]),
            ak.concatenate([i.minor_axis1_z   for i in tuples_list]),
            ak.concatenate([i.minor_axis2_x   for i in tuples_list]),
            ak.concatenate([i.minor_axis2_y   for i in tuples_list]),
            ak.concatenate([i.minor_axis2_z   for i in tuples_list]),

            ak.concatenate([i.recon_pv_key    for i in tuples_list]),
            ak.concatenate([i.recon_ov_key    for i in tuples_list]),
            
            ak.concatenate([i.recon_tx        for i in tuples_list]),
            ak.concatenate([i.recon_ty        for i in tuples_list]),            
        )
    elif tuple_type=="PV_tuple":
        if not len(tuples_list[0]._fields)==len(PV_tuple._fields):
            print("Argument for 'tuple_type': %s and input list of tuples do not match!"%(tuple_type))
            print("'tuple_type': %s has %s fields, while input tuples have %s"%(len(tuples_list[0]._fields),
                                                                                len(PV_tuple._fields)))
            exit(0)
            
        return PV_tuple(
            ak.concatenate([i.pv_x          for i in tuples_list]),
            ak.concatenate([i.pv_y          for i in tuples_list]),
            ak.concatenate([i.pv_z          for i in tuples_list]),
            ak.concatenate([i.pv_ntracks    for i in tuples_list]),
            ak.concatenate([i.pv_cat        for i in tuples_list]),
            ak.concatenate([i.pv_key        for i in tuples_list]),
        )
    elif tuple_type=="SV_tuple":
        if not len(tuples_list[0]._fields)==len(SV_tuple._fields):
            print("Argument for 'tuple_type': %s and input list of tuples do not match!"%(tuple_type))
            print("'tuple_type': %s has %s fields, while input tuples have %s"%(len(tuples_list[0]._fields),
                                                                                len(SV_tuple._fields)))
            exit(0)
            
        return SV_tuple(
            ak.concatenate([i.sv_x          for i in tuples_list]),
            ak.concatenate([i.sv_y          for i in tuples_list]),
            ak.concatenate([i.sv_z          for i in tuples_list]),
            ak.concatenate([i.sv_ntracks    for i in tuples_list]),
            ak.concatenate([i.sv_cat        for i in tuples_list]),
            ak.concatenate([i.sv_key        for i in tuples_list]),
            ak.concatenate([i.sv_pv_key     for i in tuples_list]),
        )
    else: 
        print("Argument for 'tuple_type': %s not implemented yet."%(tuple_type))
        print("Must be either: [InputData_tuple, PV_tuple, SV_tuple]")        
        exit(0)
