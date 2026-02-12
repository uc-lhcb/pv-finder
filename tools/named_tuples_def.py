'''
Below are the main namedtuple containers used to handle the data.
One can add new ones if needed.
WARNING: Several methods in tools depend on the definitions of the namedtuple containers declared here.
'''
from collections import namedtuple

PV_tuple = namedtuple("PV_tuple", 
                      ("x", "y", "z", "n", "cat", "key")
                     )

SV_tuple = namedtuple("SV_tuple", 
                      ("x", "y", "z", "n", "cat", "key", "sv_pv_key")
                     )

InputData_tuple = namedtuple("InputData_tuple",
                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                              "sv_x", "sv_y", "sv_z", "sv_ntracks", "sv_cat", "sv_key", "sv_pv_key", 
                              "poca_x",        "poca_y",        "poca_z",
                              "major_axis_x",  "major_axis_y",  "major_axis_z", 
                              "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                              "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                              "recon_pv_key",  
                              "recon_ov_key",  
                              "recon_tx", "recon_ty",
                              "IP_KDE",     "IP_KDE_xMax",     "IP_KDE_yMax",
                              "poca_KDE_A", "poca_KDE_A_xMax", "poca_KDE_A_yMax",
                              "poca_KDE_B", "poca_KDE_B_xMax", "poca_KDE_B_yMax",
                             )
                            )

InputData_tuple_SV_OV_updated = namedtuple("InputData_tuple_SV_OV_updated",
                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                              "sv_x", "sv_y", "sv_z", "sv_ntracks", 
                              "poca_x",        "poca_y",        "poca_z",
                              "major_axis_x",  "major_axis_y",  "major_axis_z", 
                              "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                              "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                              "recon_pv_key",  
                              "recon_cat",  
                              "recon_ov_x", "recon_ov_y", "recon_ov_z",
                              "recon_tx", "recon_ty",
                              "IP_KDE",     "IP_KDE_xMax",     "IP_KDE_yMax",
                              "poca_KDE_A", "poca_KDE_A_xMax", "poca_KDE_A_yMax",
                              "poca_KDE_B", "poca_KDE_B_xMax", "poca_KDE_B_yMax",
                             )
                            )

InputData_tuple_GNN = namedtuple("InputData_tuple_GNN",
                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                              "sv_x", "sv_y", "sv_z", "sv_ntracks", "sv_cat", "sv_key", "sv_pv_key", 
                              "poca_x",        "poca_y",        "poca_z",
                              "major_axis_x",  "major_axis_y",  "major_axis_z", 
                              "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                              "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                              "recon_pv_key",  
                              "recon_ov_key",  
                              "recon_tx", "recon_ty",
                             )
                            )

InputData_tuple_SV_OV_updated_GNN = namedtuple("InputData_tuple_SV_OV_updated_GNN",
                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                              "sv_x", "sv_y", "sv_z", "sv_ntracks", 
                              "poca_x",        "poca_y",        "poca_z",
                              "major_axis_x",  "major_axis_y",  "major_axis_z", 
                              "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                              "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                              "recon_pv_key",  
                              "recon_cat",  
                              "recon_ov_x", "recon_ov_y", "recon_ov_z",
                              "recon_tx",    "recon_ty",
                             )
                            )


InputData_SV_OV_updated_and_PocaEllip_tuple = namedtuple("InputData_SV_OV_updated_and_PocaEllip_tuple",
                                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                                              "sv_x", "sv_y", "sv_z", "sv_ntracks",
                                              "poca_x",        "poca_y",        "poca_z",
                                              "poca_A",        "poca_B",        "poca_C", 
                                              "poca_D",        "poca_E",        "poca_F", 
                                              "major_axis_x",  "major_axis_y",  "major_axis_z", 
                                              "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                                              "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                                              "recon_pv_key",  
                                              "recon_cat",  
                                              "recon_ov_x", "recon_ov_y", "recon_ov_z",  
                                              "recon_tx", "recon_ty",
                                              "IP_KDE",     "IP_KDE_xMax",     "IP_KDE_yMax",
                                              "poca_KDE_A", "poca_KDE_A_xMax", "poca_KDE_A_yMax",
                                              "poca_KDE_B", "poca_KDE_B_xMax", "poca_KDE_B_yMax",
                                             )
                                            )

InputData_SV_OV_updated_and_PocaEllip_tuple_GNN = namedtuple("InputData_SV_OV_updated_and_PocaEllip_tuple_GNN",
                                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                                              "sv_x", "sv_y", "sv_z", "sv_ntracks",
                                              "poca_x",        "poca_y",        "poca_z",
                                              "poca_A",        "poca_B",        "poca_C", 
                                              "poca_D",        "poca_E",        "poca_F", 
                                              "major_axis_x",  "major_axis_y",  "major_axis_z", 
                                              "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                                              "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                                              "recon_pv_key",  
                                              "recon_cat",  
                                              "recon_ov_x", "recon_ov_y", "recon_ov_z",  
                                              "recon_tx", "recon_ty",
                                             )
                                            )

InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple = namedtuple("InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple",
                                                          ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                                                           "sv_x", "sv_y", "sv_z", "sv_ntracks", 
                                                           "poca_x",        "poca_y",        "poca_z",
                                                           "poca_A",        "poca_B",        "poca_C", 
                                                           "poca_D",        "poca_E",        "poca_F", 
                                                           "major_axis_x",  "major_axis_y",  "major_axis_z", 
                                                           "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                                                           "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                                                           "recon_pv_key",  
                                                           "recon_cat",  
                                                           "recon_ov_x", "recon_ov_y", "recon_ov_z",  
                                                           "recon_tx", "recon_ty",
                                                           "IP_KDE",     "IP_KDE_xMax",     "IP_KDE_yMax",
                                                           "poca_KDE_A", "poca_KDE_A_xMax", "poca_KDE_A_yMax",
                                                           "poca_KDE_B", "poca_KDE_B_xMax", "poca_KDE_B_yMax",
                                                           "targetHists"
                                                          )
                                                         )

InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple_intervals = namedtuple("InputData_SV_OV_updated_and_PocaEllip_and_TargetHist_tuple_intervals",
                                                          ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                                                           "sv_x", "sv_y", "sv_z", "sv_ntracks", 
                                                           "poca_x",        "poca_y",        "poca_z",
                                                           "poca_A",        "poca_B",        "poca_C", 
                                                           "poca_D",        "poca_E",        "poca_F", 
                                                           "poca_x_inter",  "poca_y_inter",  "poca_z_inter",
                                                           "poca_A_inter",  "poca_B_inter",  "poca_C_inter", 
                                                           "poca_D_inter",  "poca_E_inter",  "poca_F_inter", 
                                                           "major_axis_x",  "major_axis_y",  "major_axis_z", 
                                                           "minor_axis1_x", "minor_axis1_y", "minor_axis1_z",
                                                           "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
                                                           "recon_pv_key",  
                                                           "recon_cat",  
                                                           "recon_ov_x", "recon_ov_y", "recon_ov_z",  
                                                           "recon_tx", "recon_ty",
                                                           "IP_KDE",     "IP_KDE_xMax",     "IP_KDE_yMax",
                                                           "poca_KDE_A", "poca_KDE_A_xMax", "poca_KDE_A_yMax",
                                                           "poca_KDE_B", "poca_KDE_B_xMax", "poca_KDE_B_yMax",
                                                           "targetHists",
                                                           "poca_KDE_A_inter", "poca_KDE_A_xMax_inter" ,"poca_KDE_A_yMax_inter",
                                                           "poca_KDE_B_inter", "poca_KDE_B_xMax_inter", "poca_KDE_B_yMax_inter",
                                                           "targetHists_inter"
                                                          )
                                                         )

tracks_tuple = namedtuple("tracks_tuple",
                             ("pv_x", "pv_y", "pv_z", "pv_ntracks", "pv_cat", "pv_key", 
                              "sv_x", "sv_y", "sv_z", "sv_ntracks", "sv_cat", "sv_key", "sv_pv_key", 
                              "poca_x",  "poca_y",  "poca_z",
                              "poca_A",  "poca_B",  "poca_C", 
                              "poca_D",  "poca_E",  "poca_F", 
                              "recon_pv_key",  
                             )
                            )