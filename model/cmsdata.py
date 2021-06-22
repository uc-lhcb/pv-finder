import os
import awkward as ak
import numpy as np
import pandas as pd
import ROOT
import sys
from writer import  Writer

def process_vertex_info(vfile):
    
    wf = open(vfile, 'r')
    lines = wf.readlines()

    b = ak.ArrayBuilder()

    skipped = []
    for j, line in enumerate(lines):
        try:
            current = line.split(maxsplit = 11)[11]
            current_pts = np.ndarray.tolist(np.fromstring(current, dtype=float, sep=' '))
            b.append(current_pts)

        except Exception:
            skipped.append(j)
            continue

    pts = ak.to_list(b.snapshot())
    
    if len(skipped) == 0:
        skip = None
    else: skip = skipped

    cols = ["name","lumi","evt","num","x","y","z","dx","dy","dz","n_trks"]
    vertex_info = pd.read_csv(vfile, header = None, delim_whitespace=True,
                              usecols = [i for i in range(11)], skiprows=skip)
    vertex_info.columns = cols

    del vertex_info["name"]
    vertex_info.set_index(["evt", "num"], inplace=True)
    vertex_info["trk_pts"] = pts

    wf.close()
    
    return vertex_info


def process_track_info(tfile):

    track_info = pd.read_csv(tfile, header = None, delim_whitespace=True,
                             names =("name","lumi","evt","num","good",
                                     "pt","unknown1","cottheta","phi","d0","z0"),
                             usecols = [i for i in range(11)],
                             dtype={"good": bool})

    del track_info["name"]
    track_info.set_index(["evt", "num"], inplace=True)
    track_info = track_info[track_info.good]
    
    return track_info


def process_truth_vertex_info(tvfile):
    
    wf = open(tvfile, 'r')
    lines = wf.readlines()

    b = ak.ArrayBuilder()

    skipped = []
    for j, line in enumerate(lines):
        try:
            current = line.split(maxsplit = 8)[8]
            nums = np.fromstring(current, dtype=int, sep=' ')
            b.append(nums)
        except Exception: 
            skipped.append(j)
            continue

    # change to fromlist
    nums_ak = ak.to_list(b.snapshot())

    if len(skipped) == 0:
        skip = None
    else: skip = skipped

    cols = ["name","lumi","evt","num","x","y","z","n_trks"]
    truth_vertex_info = pd.read_csv(tvfile,
                                    header = None,
                                    delim_whitespace=True, 
                                    usecols = [i for i in range(8)],
                                    engine = 'python',
                                    skiprows = skip)

    truth_vertex_info.columns = cols

    del truth_vertex_info["name"]
    truth_vertex_info.set_index(["evt", "num"], inplace=True)
    truth_vertex_info["trk_nums"] = nums_ak

    wf.close()
    
    return truth_vertex_info


def process_truth_track_info(ttfile):
    
    truth_track_info = pd.read_csv(ttfile, header = None, delim_whitespace=True)
    truth_track_info.columns = ["name","lumi","evt","num","charge",
                                "px","py","pz","vertexX","vertexY","vertexZ"]

    del truth_track_info["name"]
    truth_track_info.set_index(["evt", "num"], inplace=True)
    
    return ttfile



def create_root_file(output_name, tree_name, truth_vertex_info, track_info, truth_track_info):
    
    tfile = ROOT.TFile(output_name, "RECREATE")
    ttree = ROOT.TTree(tree_name, "")
    
    writer = Writer(ttree) 
    
    #truth vertex info
    writer.add("pvr_x")
    writer.add("pvr_y")
    writer.add("pvr_z")
    writer.add("ntrks_prompt")

    #leave SV stuff empty for now
    writer.add("svr_x") 
    writer.add("svr_y")
    writer.add("svr_z")
    writer.add("svr_pvr")

    #truth info for tracks
    writer.add("prt_pid") #leave empty for now
    writer.add("prt_hits") #leave empty for now

    #fill these below
    writer.add("prt_px")
    writer.add("prt_py")
    writer.add("prt_pz")
    writer.add("prt_e") #can be zeros
    writer.add("prt_pid") #leave empty for now
    writer.add("prt_hits") #can be zeros
    writer.add("prt_x")
    writer.add("prt_y")
    writer.add("prt_z")
    writer.add("prt_pvr")

    #actual track info
    writer.add("recon_x")
    writer.add("recon_y")
    writer.add("recon_z")
    writer.add("recon_tx")
    writer.add("recon_ty")
    writer.add("recon_chi2") # fill with ones for now

    #POCA info (fill below with zeros for now)
    writer.add("recon_pocax")
    writer.add("recon_pocay")
    writer.add("recon_pocaz")
    writer.add("recon_sigmapocaxy")
    
    n_events = len(truth_vertex_info.groupby("evt"))
    counter = 1
    for vrtx, trk, truthtrk in zip(truth_vertex_info.groupby("evt"),
                                   track_info.groupby("evt"), 
                                   truth_track_info.groupby("evt")):

        prt_pvr = []

        trk_nlist = truthtrk[1].index.get_level_values("num").tolist()
        trk_nlists = vrtx[1]["trk_nums"].tolist()
        vrtx_nlist = vrtx[1].index.get_level_values("num").tolist()

        nfound = 0
        truth_dict = {}
        for i in range(len(trk_nlists)):
            for j in trk_nlists[i]:
                truth_dict[trk_nlist[j]] = vrtx_nlist[i]
                nfound += 1
                
        prt_pvr_list = list(truth_dict.values())

        print('Event ', vrtx[0], ' Processed (', counter, '/', n_events, ')')

        xlist = vrtx[1]["x"].tolist()
        ylist = vrtx[1]["y"].tolist()
        zlist = vrtx[1]["z"].tolist()
        nlist = vrtx[1]["n_trks"].tolist()


        for i in range(len(xlist)):

            writer["pvr_x"].append(xlist[i])
            writer["pvr_y"].append(ylist[i])
            writer["pvr_z"].append(zlist[i])
            writer["ntrks_prompt"].append(nlist[i]) # currently not used


        d0 = np.asarray(trk[1]["d0"]).tolist()
        phi = np.asarray(trk[1]["phi"]).tolist()
        z0 = np.asarray(trk[1]["z0"]).tolist()
        cottheta = np.asarray(trk[1]["cottheta"]).tolist()

        arr1 = np.multiply(d0,np.cos(np.add(phi,-np.pi/2)))
        arr2 = np.multiply(d0,np.sin(np.add(phi,-np.pi/2)))

        loc = np.stack((np.multiply(d0,np.cos(np.add(phi,-np.pi/2))),
                        np.multiply(d0,np.sin(np.add(phi,-np.pi/2))), 
                        z0))

        direc = np.stack((np.cos(phi), np.sin(phi)))
        direc = np.divide(direc, np.linalg.norm(direc, axis = 1).reshape(2,1))

        for i in range(len(loc[0])):
            writer["recon_x"].append(loc[0][i])
            writer["recon_y"].append(loc[1][i])
            writer["recon_z"].append(loc[2][i])

            writer["recon_tx"].append(direc[0][i])
            writer["recon_ty"].append(direc[1][i])

            writer["recon_chi2"].append(0)

            writer["recon_pocax"].append(1)
            writer["recon_pocay"].append(1)
            writer["recon_pocaz"].append(1)
            writer["recon_sigmapocaxy"].append(1)

        pxlist = truthtrk[1]["px"].tolist()
        pylist = truthtrk[1]["py"].tolist()
        pzlist = truthtrk[1]["pz"].tolist()
        xlist = truthtrk[1]["vertexX"].tolist()
        ylist = truthtrk[1]["vertexY"].tolist()
        zlist = truthtrk[1]["vertexZ"].tolist()

        for i in range(len(pxlist)):
            writer["prt_px"].append(pxlist[i])
            writer["prt_py"].append(pylist[i])
            writer["prt_pz"].append(pzlist[i])
            writer["prt_e"].append(0)
            writer["prt_x"].append(xlist[i])
            writer["prt_y"].append(ylist[i])
            writer["prt_z"].append(zlist[i])
            writer["prt_hits"].append(0)
            writer["prt_pvr"].append(prt_pvr_list[i])

        ttree.Fill()
        writer.clear()

        counter += 1

    ttree.Print()
    ttree.Write(ttree.GetName(), ROOT.TObject.kOverwrite)
    tfile.Close()
