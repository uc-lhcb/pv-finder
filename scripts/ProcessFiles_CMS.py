# this script was modified for CMS (by EMK) using Rocky's code as a base

import argparse
from pathlib import Path
import numpy as np
import warnings
import uproot
import math
from scipy import special
import numba
from collections import namedtuple
import awkward
import h5py

#Input parameters
zMin = -15 #cm (rocky had this at -240mm for ATLAS)
zMax = 15 #cm (rocky had this at 240mm for ATLAS)
totalNumBins = 8000 # (rocky had this at 12000)
bins_1cm = int(totalNumBins/(zMax - zMin)) #number of bins in 1cm
bins_1cm = totalNumBins/(zMax - zMin)
binWidth = 1/bins_1cm #binsize in cm # 50micrometer = 0.005mm in current case

print()
print("zMin = %s cm" %(zMin))
print("zMax = %s cm" %(zMax))
print("totalNumBins = %s" %(totalNumBins))
# print("bins in 1cm (noninteger) = %s bins" %(totalNumBins/(zMax - zMin)))
print("bins in 1cm = %s bins" %(bins_1cm))
print("binWidth in cm = %s cm" %(binWidth))

#get range of bins for probability computation
bins = np.arange(-10,11) #compute probability in +-10 neighboring bins
edges = np.array([-binWidth/2, binWidth/2]) #[-0.02, 0.02]
ProbRange = binWidth * bins[np.newaxis, :] + edges[:, np.newaxis] + zMin

pv_nCat = 2 # number of categories (first is good PVs, second is bad PVs)

OutputData = namedtuple(
    "OutputData",
    (   "POCAzdata",          # Density in Z (calculated from POCA), totalNumBinsxN
        "POCA_sqzdata",       # Squared Density in Z (calculated from POCA), totalNumBinsxN
        "POCAxmax",           # Position of max density in x for each z-bin (calculated from POCA, same for zdata and sqzdata)
        "POCAymax",           # Position of max density in y for each z-bin (calculated from POCA, same for zdata and sqzdata)
        "oldzdata",           # density in z (old version), totalNumBinsxN
        "oldxmax",            # Position of max density in x for each z-bin (old version)
        "oldymax",            # Position of max density in y for each z-bin (old version)
        "pv",                 # target pv histogram, totalNumBinsxN (has channel for good PVs and channel for bad PVs)
        "pv_loc_x",           # position of PV in x
        "pv_loc_y",           # position of PV in y
        "pv_loc",             # position of PV in z
        "pv_ntrks",           # number of tracks associated with each PV
        #"pv_cat",             # PV category/quality (either 0 or -1)
        "sv_loc_x",           # position of SV in x
        "sv_loc_y",           # position of SV in y
        "sv_loc",             # position of SV in z
        "sv_ntrks",           # number of tracks associated with each SV
        "sv_cat",             # SV category/quality (either 0 or -1)
        "recon_x",            # reconstructed track x position
        "recon_y",            # reconstructed track y position
        "recon_z",            # reconstructed track z position
        "recon_tx",           # reconstructed track x-component of track vector
        "recon_ty",           # reconstructed track y-component of track vector
        "poca_x",             # POCA x-center
        "poca_y",             # POCA y-center
        "poca_z",             # POCA z-center
        "major_axis_x",       # POCA major axis x-component
        "major_axis_y",       # POCA major axis y-component
        "major_axis_z",       # POCA major axis z-component
        "minor_axis1_x",      # POCA minor axis 1 x-component
        "minor_axis1_y",      # POCA minor axis 1 y-component
        "minor_axis1_z",      # POCA minor axis 1 z-component
        "minor_axis2_x",      # POCA minor axis 2 x-component
        "minor_axis2_y",      # POCA minor axis 2 y-component
        "minor_axis2_z"       # POCA minor axis 2 z-component
    )
)
                         
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Getting target histograms")
    parser.add_argument('-i',
                        '--infile',
                        dest='infile',
                        type=Path,
                        required=True,
                        nargs="+",
                        help='Input root files to read in')
    parser.add_argument('-o',
                        '--output',
                        dest='outpath',
                        type=Path,
                        required=True,
                        help='Output .h5 File')      
    return parser

#get bin number, given z value 
def binNumber(mean):
    return int(np.floor((mean - zMin)*bins_1cm))

#get z value, given bin number
def binCenter(zmin, zmax, nbins, ibin):
    return ((ibin + 0.5) / nbins) * (zmax - zmin) + zmin

def findRMax(zval, Xmax, Ymax, zbins = np.linspace(-15,15,8000)):
    idx = (np.abs(zbins - zval)).argmin()
    return np.sqrt(Xmax[idx]**2 + Ymax[idx]**2)

'''
def binValue(zmin, zmax, mean, nbins):
    return int(np.floor((mean - zmin)*(nbins/(zmax - zmin))))
'''

def ComputeSigma(ntrks):

    ##  values found by EMK using extrapolated data from https://arxiv.org/pdf/1405.6569.pdf
    A_res = 62.868879
    B_res = 0.95620653
    C_res = 10.3188071
    
    if ntrks < 4:
        return binWidth
    else:
        return 1e-3 * (A_res * np.power(ntrks, -1 * B_res) + C_res)

@numba.vectorize(nopython=True)
def norm_cdf(mu, sigma, x):
    """
    Cumulative distribution function for the standard normal distribution.
    """
    #TODO
    #add contribution from pv_x and pv_y also 
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))
    #return 0.5 * (1 + special.erf((x - mu) / (sigma * math.sqrt(2.0))))


def main():
    
    options = getArgumentParser().parse_args()
    input_file = options.infile
    output_hd5file = options.outpath
    print(input_file)
    print(options.outpath)

    for f in input_file:
        #opening input file and getting tree
        tree = uproot.open(str(f))["kernel"]
        branches = tree.arrays() # tree.arrays(namedecode='utf-8')
        #print(branches)
        
        #get all the branches 
        kernel_z = branches["POCAzdata"]
        kernel_zsq = branches["POCA_sqzdata"]
        kernel_xmax = branches["POCAxmax"]
        kernel_ymax = branches["POCAymax"]
        
        kernel_z_old = branches["oldzdata"]
        kernel_xmax_old = branches["oldxmax"]
        kernel_ymax_old = branches["oldymax"]
        
        #2D arrays with each row representing an event
        pv_loc_x = branches["pv_loc_x"]
        pv_loc_y = branches["pv_loc_y"]
        pv_loc_z = branches["pv_loc"]
#         pv_cat = branches["pv_cat"]
        pv_ntrks = branches["pv_ntrks"]
        sv_loc_x = branches["sv_loc_x"]
        sv_loc_y = branches["sv_loc_y"]
        sv_loc_z = branches["sv_loc"]
        sv_cat = branches["sv_cat"]
        sv_ntrks = branches["sv_ntrks"]

        recoTrk_x = branches["recon_x"]
        recoTrk_y = branches["recon_y"]
        recoTrk_z = branches["recon_z"]
        recoTrk_tx = branches["recon_tx"]
        recoTrk_ty = branches["recon_ty"]
        
        poca_x = branches["POCA_center_x"]
        poca_y = branches["POCA_center_y"]
        poca_z = branches["POCA_center_z"]
        
        poca_majoraxis_x = branches["POCA_major_axis_x"]
        poca_majoraxis_y = branches["POCA_major_axis_y"]
        poca_majoraxis_z = branches["POCA_major_axis_z"]
        poca_minoraxis1_x = branches["POCA_minor_axis1_x"]
        poca_minoraxis1_y = branches["POCA_minor_axis1_y"]
        poca_minoraxis1_z = branches["POCA_minor_axis1_z"]
        poca_minoraxis2_x = branches["POCA_minor_axis2_x"]
        poca_minoraxis2_y = branches["POCA_minor_axis2_y"]
        poca_minoraxis2_z = branches["POCA_minor_axis2_z"]

        #getting the target histograms
        NumEvts = len(kernel_z)    

        #Output multidim array
        Output_Y = np.zeros([NumEvts, pv_nCat, totalNumBins], dtype=np.float16)

        for ievt in range(NumEvts):
        
            #print("for evt = %s" %(ievt))
            #print("**************")    

            pv_loc_x_curr = pv_loc_x[ievt] 
            pv_loc_y_curr = pv_loc_y[ievt] 
            pv_loc_z_curr = pv_loc_z[ievt]
            
            pv_ntrks_curr = pv_ntrks[ievt]

            #number of pvs in this event
            nPV = len(pv_loc_z_curr)
        
            for ipv in range(nPV):
            
                pv_center = pv_loc_z_curr[ipv]
                ntrks = pv_ntrks_curr[ipv]
                pv_res = ComputeSigma(ntrks)
#                 cat_current = pv_cat[ievt][ipv]
                cat_current=0
                if findRMax(pv_center,kernel_xmax[ievt],kernel_ymax[ievt])==0:
                    cat_current=1
                
                if pv_center >= zMin and pv_center <= zMax:
#                     print()
#                     print("pv_center = ", pv_center)
                    nbin = binNumber(pv_center)
#                     print("nbin = ", nbin)
#                     print("z-position of nbin = ", binCenter(zMin,zMax,totalNumBins,nbin))
                    z_probRange = nbin/bins_1cm + ProbRange
#                     print("z_probRange = ", z_probRange)
                    probValues = norm_cdf(pv_center, pv_res, z_probRange)

                    populate = probValues[1] - probValues[0]

                    #TODO: Check the impact of this step
                    #populate = np.where((0.15 / pv_res) > 1, (0.15 / pv_res) * populate, populate)
                    if cat_current == 1:
                        Output_Y[ievt, 0, bins+nbin] += populate
                    else:
                        Output_Y[ievt, 1, bins+nbin] += populate
                    
    Output = OutputData(
        kernel_z,
        kernel_zsq,
        kernel_xmax,
        kernel_ymax,
        kernel_z_old,
        kernel_xmax_old,
        kernel_ymax_old,
        Output_Y,
        pv_loc_x,
        pv_loc_y,
        pv_loc_z,
        pv_ntrks,
        #pv_cat,
        sv_loc_x,
        sv_loc_y,
        sv_loc_z,
        sv_ntrks,
        sv_cat,
        recoTrk_x,
        recoTrk_y,
        recoTrk_z,
        recoTrk_tx,
        recoTrk_ty,
        poca_x,
        poca_y,
        poca_z,
        poca_majoraxis_x,
        poca_majoraxis_y,
        poca_majoraxis_z,
        poca_minoraxis1_x,
        poca_minoraxis1_y,
        poca_minoraxis1_z,
        poca_minoraxis2_x,
        poca_minoraxis2_y,
        poca_minoraxis2_z
    )

    with h5py.File(str(output_hd5file), "w") as hf:

        grp_POCAzdata = hf.create_group("POCAzdata")
        grp_POCAsqzdata = hf.create_group("POCA_sqzdata")
        grp_POCAxmax = hf.create_group("POCAxmax")
        grp_POCAymax = hf.create_group("POCAymax")
        grp_oldzdata = hf.create_group("oldzdata")
        grp_oldxmax = hf.create_group("oldxmax")
        grp_oldymax = hf.create_group("oldymax")
        grp_pv = hf.create_group("pv")
        grp_pv_loc_x = hf.create_group("pv_loc_x")
        grp_pv_loc_y = hf.create_group("pv_loc_y")
        grp_pv_loc_z = hf.create_group("pv_loc")
        grp_pv_ntrks = hf.create_group("pv_ntrks")
        #grp_pv_cat = hf.create_group("pv_cat")
        grp_sv_loc_x = hf.create_group("sv_loc_x")
        grp_sv_loc_y = hf.create_group("sv_loc_y")
        grp_sv_loc_z = hf.create_group("sv_loc")
        grp_sv_ntrks = hf.create_group("sv_ntrks")
        grp_sv_cat = hf.create_group("sv_cat")
        grp_recon_x = hf.create_group("recon_x")
        grp_recon_y = hf.create_group("recon_y")
        grp_recon_z = hf.create_group("recon_z")
        grp_recon_tx = hf.create_group("recon_tx")
        grp_recon_ty = hf.create_group("recon_ty")
        grp_POCA_center_x = hf.create_group("POCA_center_x")
        grp_POCA_center_y = hf.create_group("POCA_center_y")
        grp_POCA_center_z = hf.create_group("POCA_center_z")
        grp_POCA_major_axis_x = hf.create_group("POCA_major_axis_x")
        grp_POCA_major_axis_y = hf.create_group("POCA_major_axis_y")
        grp_POCA_major_axis_z = hf.create_group("POCA_major_axis_z")
        grp_POCA_minor_axis1_x = hf.create_group("POCA_minor_axis1_x")
        grp_POCA_minor_axis1_y = hf.create_group("POCA_minor_axis1_y")
        grp_POCA_minor_axis1_z = hf.create_group("POCA_minor_axis1_z")
        grp_POCA_minor_axis2_x = hf.create_group("POCA_minor_axis2_x")
        grp_POCA_minor_axis2_y = hf.create_group("POCA_minor_axis2_y")
        grp_POCA_minor_axis2_z = hf.create_group("POCA_minor_axis2_z")
        
        for evt in range(NumEvts):
            datasetName = "Event"+str(evt)
            #print(datasetName)
            grp_POCAzdata.create_dataset(datasetName, data=Output.POCAzdata[evt], compression="lzf")
            grp_POCAsqzdata.create_dataset(datasetName, data=Output.POCA_sqzdata[evt], compression="lzf")
            grp_POCAxmax.create_dataset(datasetName, data=Output.POCAxmax[evt], compression="lzf")
            grp_POCAymax.create_dataset(datasetName, data=Output.POCAymax[evt], compression="lzf")
            grp_oldzdata.create_dataset(datasetName, data=Output.oldzdata[evt], compression="lzf")
            grp_oldxmax.create_dataset(datasetName, data=Output.oldxmax[evt], compression="lzf")
            grp_oldymax.create_dataset(datasetName, data=Output.oldymax[evt], compression="lzf")
            grp_pv.create_dataset(datasetName, data=Output.pv[evt], compression="lzf")
            grp_pv_loc_x.create_dataset(datasetName, data=Output.pv_loc_x[evt], compression="lzf")
            grp_pv_loc_y.create_dataset(datasetName, data=Output.pv_loc_y[evt], compression="lzf")
            grp_pv_loc_z.create_dataset(datasetName, data=Output.pv_loc[evt], compression="lzf")
            grp_pv_ntrks.create_dataset(datasetName, data=Output.pv_ntrks[evt], compression="lzf")
            #grp_pv_cat.create_dataset(datasetName, data=Output.pv_cat[evt])
            grp_sv_loc_x.create_dataset(datasetName, data=Output.sv_loc_x[evt], compression="lzf")
            grp_sv_loc_y.create_dataset(datasetName, data=Output.sv_loc_y[evt], compression="lzf")
            grp_sv_loc_z.create_dataset(datasetName, data=Output.sv_loc[evt], compression="lzf")
            grp_sv_ntrks.create_dataset(datasetName, data=Output.sv_ntrks[evt], compression="lzf")
            grp_sv_cat.create_dataset(datasetName, data=Output.sv_cat[evt])
            grp_recon_x.create_dataset(datasetName, data=Output.recon_x[evt], compression="lzf")
            grp_recon_y.create_dataset(datasetName, data=Output.recon_y[evt], compression="lzf")
            grp_recon_z.create_dataset(datasetName, data=Output.recon_z[evt], compression="lzf")
            grp_recon_tx.create_dataset(datasetName, data=Output.recon_tx[evt], compression="lzf")
            grp_recon_ty.create_dataset(datasetName, data=Output.recon_ty[evt], compression="lzf")
            grp_POCA_center_x.create_dataset(datasetName, data=Output.poca_x[evt], compression="lzf")
            grp_POCA_center_y.create_dataset(datasetName, data=Output.poca_y[evt], compression="lzf")
            grp_POCA_center_z.create_dataset(datasetName, data=Output.poca_z[evt], compression="lzf")
            grp_POCA_major_axis_x.create_dataset(datasetName, data=Output.major_axis_x[evt], compression="lzf")
            grp_POCA_major_axis_y.create_dataset(datasetName, data=Output.major_axis_y[evt], compression="lzf")
            grp_POCA_major_axis_z.create_dataset(datasetName, data=Output.major_axis_z[evt], compression="lzf")
            grp_POCA_minor_axis1_x.create_dataset(datasetName, data=Output.minor_axis1_x[evt], compression="lzf")
            grp_POCA_minor_axis1_y.create_dataset(datasetName, data=Output.minor_axis1_y[evt], compression="lzf")
            grp_POCA_minor_axis1_z.create_dataset(datasetName, data=Output.minor_axis1_z[evt], compression="lzf")
            grp_POCA_minor_axis2_x.create_dataset(datasetName, data=Output.minor_axis2_x[evt], compression="lzf")
            grp_POCA_minor_axis2_y.create_dataset(datasetName, data=Output.minor_axis2_y[evt], compression="lzf")
            grp_POCA_minor_axis2_z.create_dataset(datasetName, data=Output.minor_axis2_z[evt], compression="lzf")
        
    
if __name__ == "__main__":
    main()
    
