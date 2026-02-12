## ======= NECESSARY IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

## ======= FIGURE PARAMETERS
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.fontsize"] = 29


fig, ax = plt.subplots(1, 1)
ax.xaxis.set_tick_params(top=True, labeltop=False, direction = "in", length=6, width=1.0)
ax.yaxis.set_tick_params(top=True, labeltop=False, direction = "in", length=6, width=1.0)
#plt.xaxis.set_tick_params(top=True, labeltop=False)


## ======= LIST OF INPUTS

eff = [88.1, 90.4, 92.1, 93.9]
fp =  [0.05, 0.08, 0.13, 0.25]

##  mds added 190804 for FourFeature_CNN6Plus models
eff_190804 = [88.30, 90.885, 92.058, 92.742, 93.87, 94.362, 94.829]
fp_190804 = [0.0454, 0.07403, 0.098, 0.1205, 0.1703, 0.2087, 0.2327]

## mds added starting 190817 for FourFeature_CNN6Plus models with 160K TargetsAA
eff_sse19 = [87.7, 89.4, 91.9, 92.3, 93.4, 94.0, 94.3, 94.7, 95.0]
fp_sse19 = [0.033, 0.043, 0.073, 0.081, 0.110, 0.134, 0.147, 0.188, 0.224]

eff_CNN7 = [92.6, 94.5, 94.9]
fp_CNN7 = [0.086, 0.152, 0.186]

eff_unet_201201 =  [93.3,  94.3,   94.3,  94.9,  95.0,  95.1,  95.3,  95.1]
fp_unet_201201  =  [0.094, 0.127,  0.13,  0.159, 0.178, 0.175, 0.193, 0.209]

## mds added points for FourFeature_CNN6Plus models trained with 80K full LHCb MC
eff_lhcb = [90.18, 93.95, 94.67, 95.85, 96.49, 96.69]
fp_lhcb  = [0.057, 0.0912, 0.104, 0.137, 0.164, 0.178]

## mds add points for Two_KDE_withPcnn
eff_two_kde = [90.48, 94.39,  95.5, 96.55,  97.13, 97.57, 97.77, 97.83]
fp_two_kde  = [0.06,  0.089, 0.102, 0.127,  0.149, 0.176, 0.201,  0.235]

## from Michael Peters, 22 April 2022
# perturbative UNet Full LHCb MC
eff_unet_220422 = [96.7,97.4,97.6,97.7,97.7,97.8,97.9,97.9,98.0]
fp_unet_220422 = [.137,.176,.187,.202,.207,.233,.265,.280,.304]

## mds, 17 September 2022, from HDplusUNet100
## asym_c = 2.5, 5.0, 10.0 (negative asymm. coeff. goes to very low efficiencies below 0.7)
eff_HDplusUNet100 = [97.1,  97.5, 97.8]
fp_HDplusUNet100  =[0.041, 0.060, 0.104]

## sakar, 26 September 2022, from HDplusUNet100
## asym_c = 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20, 25 (negative asymm. coeff. goes to very low efficiencies below 0.7)
# eff_HDplusUNet100_sakar = [93.22, 96.06, 97.05, 97.27, 97.33, 97.50, 97.63] ## c = 0.0 : eff = 81.48
# fp_HDplusUNet100_sakar  = [0.006, 0.015, 0.026, 0.053, 0.058, 0.087, 0.083] ## c = 0.0 : fp  =  0.001

## With resolution parameters: 
# - threshold = 0.07 
# - integral_threshold = 0.7

#
eff_heuristic_TrackBeamLineVertexFinder = [93.80] ## rouding up from [93.73]
fp_heuristic_TrackBeamLineVertexFinder  = [0.015] ## corresponding to a FP rate of ~0.3%

## asym_c =                  1.0,   2.5,   5.0,  10.0,  15.0,  20.0,  25.0, 30.0 
eff_HDplusUNet100_sakar = [93.22, 96.06, 97.05, 97.33, 97.63, 97.55, 97.59]#, 97.50] 
fp_HDplusUNet100_sakar  = [0.006, 0.015, 0.026, 0.058, 0.083, 0.119, 0.132]#, 0.188] 
## c = 0.0 : eff = 81.48; fp  =  0.001
## (negative asymm. coeff. goes to very low efficiencies below 0.7)

##  add new points for HDplusUNet100 with more training; asym = [ 0%, 2.5%, 5%, 10%, 15%, 20%]
eff_HDplusUNet100_moreTraining = [97.5,   97.8,   98.0,  98.2,  98.2]
fp_HDplusUNet100_moreTraining  = [0.040, 0.065, 0.107, 0.146, 0.169]

eff_HDplusUNet100_moreTraining_allPVs = [88.31, 96.04,   96.76,   97.17,  97.39,  97.50]
fp_HDplusUNet100_moreTraining_allPVs  = [0.0054, 0.052, 0.0839, 0.1367, 0.1833, 0.2259]

eff_HDplusUNet100_moreTraining_allPVs_valid4 = [88.31, 95.31,   96.16,   96.66,  96.93,  97.06]
fp_HDplusUNet100_moreTraining_allPVs_valid4  = [0.0054, 0.0289, 0.0537, 0.1013, 0.1450, 0.1867]
#eff_HDplusUNet100_moreTraining_allPVs_valid4 = [95.31]#[88.31, 95.31,   96.16,   96.66,  96.93,  97.06]
#fp_HDplusUNet100_moreTraining_allPVs_valid4  = [0.0289]#[0.0054, 0.0289, 0.0537, 0.1013, 0.1450, 0.1867]


## FP16 results from ML/17April2023_t2hists_HDplusUNet100A_FP16_iter1C_40epochs_4em6_JpsiPhiMagDown_Data_asymm2p5/17April2023_t2hists_HDplusUNet100A_FP16_iter1C_40epochs_4em6_JpsiPhiMagDown_Data_asymm2p5_final.pyt
## With resolution parameters: 
# - threshold = 0.10
# - integral_threshold = 0.15
'''
## asym_c =                                  0.0,   1.0,   2.5,   5.0,   10.0
eff_HDplusUNet100A_16_UNetChannels_FP16 = [92.83, 94.52, 95.07, 95.93, 96.32] 
fp_HDplusUNet100A_16_UNetChannels_FP16  = [0.006, 0.008, 0.015, 0.031, 0.060] 
'''
## FP16 results from ML/21April2023_t2hists_HDplusUNet100A_U32_FP16_iter4A_50epochs_1em8_JpsiPhiMagDown_Data_asymm2p5/21April2023_t2hists_HDplusUNet100A_U32_FP16_iter4A_50epochs_1em8_JpsiPhiMagDown_Data_asymm2p5_final.pyt
## With resolution parameters: 
# - threshold = 0.10
# - integral_threshold = 0.15
'''
## asym_c =                                  0.0,   1.0,   2.5,   5.0,  10.0
eff_HDplusUNet100A_32_UNetChannels_FP16 = [95.24, 95.85, 96.29, 96.38, 96.32] 
fp_HDplusUNet100A_32_UNetChannels_FP16  = [0.014, 0.023, 0.028, 0.043, 0.044] 
'''

## FP16 results from ML/21April2023_t2hists_HDplusUNet100A_U32_FP16_iter4A_50epochs_1em8_JpsiPhiMagDown_Data_asymm2p5/21April2023_t2hists_HDplusUNet100A_U32_FP16_iter4A_50epochs_1em8_JpsiPhiMagDown_Data_asymm2p5_final.pyt
## With resolution parameters: 
# - threshold = 0.15
# - integral_threshold = 0.20
'''
## asym_c =                                  0.0,   1.0,   2.5,   5.0,  10,0,  15.0, 20.0
eff_HDplusUNet100A_64_UNetChannels_FP16 = [94.66, 95.89, 96.42, 96.93, 97.04, 96.96, 96.98] 
fp_HDplusUNet100A_64_UNetChannels_FP16  = [0.011, 0.024, 0.039, 0.051, 0.080, 0.106, 0.133] 
'''

##  add some results with pruning 06November2022
'''
eff_pruning20pc = [97.5]
fp_pruning20pc  = [0.041]

eff_morePruning = [97.3]
fp_morePruning  = [0.037]
'''

## GNN -- WITH GROUPS -- NORM DIST -- ALPHA1 
eff_GNN_groups_loss_norm_alpha1 = [95.1029, 96.4508, 96.7421, 96.8640] 
fp_GNN_groups_loss_norm_alpha1  = [0.01250, 0.01968, 0.02840, 0.04960] 

## GNN -- WITH GROUPS -- NORM DIST -- ALPHA1 -- All_MinBiasUp first 70000 of Dpi and first 70000 of Jpsi samples
eff_GNN_groups_loss_norm_alpha1_moreData1_epoch25 = [95.71, 96.97, 97.22, 97.30] 
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch25  = [0.0117, 0.0166, 0.0246, 0.0394] 
#
eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44 = [95.78, 97.02, 97.26, 97.33] 
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44  = [0.0125, 0.0182, 0.0250, 0.0386] 
#
eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_validPV4 = [95.30, 96.67, 96.98, 97.08] 
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_validPV4  = [0.0125, 0.0182, 0.0254, 0.0390] 



## GNN -- WITH GROUPS -- NORM DIST -- ALPHA1 -- All_MinBiasUp -- GraphTracksMin 2 -- Iso 1p0
## Cands min tracks: 5,4,3
eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0 = [95.52,96.32,96.77] 
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0  = [0.0232,0.0296,0.0427]

eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0_validPV4 = [94.75,95.72,96.33] 
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0_validPV4  = [0.0240,0.0307,0.0444]

## GNN -- WITH GROUPS -- NORM DIST -- ALPHA1 -- All_MinBiasUp -- GraphTracksMin 2 -- Iso 0p7
eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7 = [95.06,95.96,96.52]  
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7  = [0.0183,0.0244,0.0354]

eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4 = [94.31,95.35,96.15]  
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4  = [0.0188,0.0253,0.0366]
#eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4 = [95.35]#[94.31,95.35,96.15]  
#fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4  = [0.0253]#[0.0188,0.0253,0.0366]

## GNN -- WITH GROUPS -- NORM DIST -- ALPHA1 -- All_MinBiasUp -- GraphTracksMin 2 -- Iso 0p5
eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5 = [94.42,95.53,96.15]
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5  = [0.0130,0.0172,0.0269]

eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5_validPV4 = [93.73,94.96,95.73] 
fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5_validPV4  = [0.0134,0.0180,0.0279]

eff_GNN_or_CNN = [96.59]
fp_GNN_or_CNN = [0.0486]

eff_GNN_and_CNN = [94.31]
fp_GNN_and_CNN = [0.0050]

## ======= ADD THE INPUTS DATA TO THE PLOT

plt.plot(eff_heuristic_TrackBeamLineVertexFinder, fp_heuristic_TrackBeamLineVertexFinder, color = 'black', marker = "X", markersize=10, markerfacecolor="blue", label = "LHCb classic", linestyle = "None")



#plt.plot(eff, fp, color = 'b', marker = "o", markersize=8, label = "KDE-to-hist -- ACAT-19 (IP & toy MC)", linestyle = "None")
##plt.plot(eff_190804, fp_190804, color = 'g', marker = "s", markersize=8, label = "CtD-2020: improved architecture",  linestyle = "None")
#plt.plot(eff_sse19, fp_sse19, color = 'green', marker = "s", markersize=8, label = "KDE-to-hist / CNN (IP & toy MC) / CtD-20", linestyle = "None")
#plt.plot(eff_lhcb, fp_lhcb, color = 'red', marker = "^", markersize=8, label = "KDE-to-hist / CNN (IP)", linestyle = "None")
'''
plt.plot(eff_two_kde, fp_two_kde, color = 'blue', marker = "o", markersize=8, markerfacecolor="cyan", label = "KDE-to-hist / CNN (POCA) / CHEP-21", linestyle = "None")
'''
#plt.plot(eff_unet_220422, fp_unet_220422, color = 'r', marker = "s", markersize=8, markerfacecolor="orange", label = "KDE-to-hist / UNet (POCA)", linestyle = "None")
## repeat this data set so it is above the UNet data, but without a legend to maintain the order in the legend box
#plt.plot(eff_two_kde, fp_two_kde, color = 'blue', marker = "o", markersize=8, markerfacecolor="cyan", linestyle = "None")


#plt.plot(eff_HDplusUNet100_sakar, fp_HDplusUNet100_sakar, color = 'black', marker = "D", markersize=8, markerfacecolor="magenta", label = r'tracks-to-hists / FC+UNet (POCA) / CHEP-23', linestyle = "None")

##  230927 add the moreTraining results
plt.plot(eff_HDplusUNet100_moreTraining_allPVs_valid4,fp_HDplusUNet100_moreTraining_allPVs_valid4, color='blue', marker="o", markersize=8, markerfacecolor="magenta", label = "CNN", linestyle="None")
#plt.plot(eff_HDplusUNet100_moreTraining_allPVs,fp_HDplusUNet100_moreTraining_allPVs, color='orange', marker="o", markersize=10, markerfacecolor="magenta", label = "FC+UNet (POCA) / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 5$", linestyle="None")

## GNN
#plt.plot(eff_GNN_groups_loss_norm_alpha1, fp_GNN_groups_loss_norm_alpha1, color = 'black', marker = "*", markersize=12, markerfacecolor="black", label = "GNN / minBiasUp / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 5$", linestyle = "None")
## 
#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch25, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch25, color = 'black', marker = "*", markersize=12, markerfacecolor="orange", label = "GNN (POCA & full LHCb MC) -- mBiasUp + D0pi + JpsiPhi -- Epoch25", linestyle = "None")
## 
#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44, color = 'black', marker = "*", markersize=12, markerfacecolor="limegreen", label = "GNN (POCA) / minBiasUp+D0pi+Jpsiphi / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 5$", linestyle = "None")

#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5, color = 'black', marker = "*", markersize=12, markerfacecolor="limegreen", label = "GNN (POCA) / minBiasUp+D0pi+Jpsiphi / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 5$ / Iso = 0.5 mm", linestyle = "None")

#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7, color = 'black', marker = "*", markersize=12, markerfacecolor="darkgreen", label = "GNN (POCA) / minBiasUp+D0pi+Jpsiphi / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 5$ / Iso = 0.7 mm", linestyle = "None")

#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0, color = 'black', marker = "*", markersize=12, markerfacecolor="black", label = "GNN (POCA) / minBiasUp+D0pi+Jpsiphi / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 5$ / Iso = 1.0 mm", linestyle = "None")


#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5_validPV4, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p5_validPV4, color = 'black', marker = "*", markersize=13, markerfacecolor="gold", label = "GNN (POCA) / Iso = 0.5 mm", linestyle = "None")

#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4, color = 'black', marker = "*", markersize=13, markerfacecolor="darkorange", label = "GNN (POCA) / Iso = 0.7 mm", linestyle = "None")

#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0_validPV4, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso1p0_validPV4, color = 'black', marker = "*", markersize=13, markerfacecolor="firebrick", label = "GNN (POCA) / Iso = 1.0 mm", linestyle = "None")

plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_GraphTracksMin2_Iso0p7_validPV4, color = 'black', marker = "*", markersize=15, markerfacecolor="darkorange", label = "GNN", linestyle = "None")

## 
#plt.plot(eff_GNN_groups_loss_norm_alpha1_moreData1_epoch44_validPV4, fp_GNN_groups_loss_norm_alpha1_moreData1_epoch44_validPV4, color = 'black', marker = "*", markersize=12, markerfacecolor="darkorange", label = "GNN (POCA)  / minBiasUp+D0pi+Jpsiphi / $N^\mathrm{PVs}_{\mathrm{tracks}}\geq 4$", linestyle = "None")


plt.plot(eff_GNN_or_CNN, fp_GNN_or_CNN, color = 'black', marker = "*", markersize=15, markerfacecolor="limegreen", label = "CNN | GNN", linestyle = "None")

plt.plot(eff_GNN_and_CNN, fp_GNN_and_CNN, color = 'black', marker = "*", markersize=15, markerfacecolor="cyan", label = "CNN & GNN", linestyle = "None")


plt.grid(linestyle = '--', linewidth = 0.5)

## ======= ADD THE LEGEND AFTER INCLUDING ALL THE INPUTS DATA
plt.legend(loc="upper left", framealpha=0.01, fontsize=18)

## ======= AXIS RANGES DEFINITION 
#plt.axis([87.0, 98.5, 0.00, 0.15])
plt.axis([93.0, 97.5, 0.00, 0.15])

## ======= X-AXIS 
plt.xlabel("Efficiency (%)", fontsize=20)
plt.xticks(fontsize=18)
plt.xticks(fontname = 'Times New Roman')
#plt.xticks(np.arange(87, 98.5, 1))
plt.xticks(np.arange(93, 97.5, 1))

## ======= Y-AXIS 
plt.ylabel("False Positive Rate (per event) ",fontsize=20)
plt.yticks(fontsize=18)
plt.yticks(fontname = 'Times New Roman')

## ======= OVERALL PLOT TITLE
#plt.title('Performance Evolution', fontsize=16)

## ======= SAVING PLOTS 
plt.savefig("EffVsFP_14March2024_fullModel_zoom_GNN_latest_withComp_simple.png")
plt.savefig("EffVsFP_14March2024_fullModel_zoom_GNN_latest_withComp_simple.pdf")

## ======= SEMI-LOG PLOT SETTINGS and SAVING
## mds plt.axis([87.0, 96.0, 0.03, 0.4])
## mds plt.semilogy(eff, fp, color = 'b', marker = "o", linestyle = "None")
## mds plt.semilogy(eff_190804, fp_190804, color = 'r', marker = "o", linestyle = "None")
## mds plt.semilogy(eff_sse19, fp_sse19, color = 'g', marker = "o", linestyle = "None")
## mds plt.savefig("EffVsFP_semilog_26Apr2023.png")
## mds 
