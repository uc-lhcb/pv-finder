#!/usr/bin/env python

from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser(description="Generate some traks")

parser.add_argument(
    "--events",
    type=int,
    default=10000,
    help="The number of events to produce per thread",
)

parser.add_argument("--threads", type=int, default=1, help="Number of processes to run")

parser.add_argument(
    "--start", type=int, default=1, help="The integer to start counting threads from"
)

parser.add_argument("name", help="The name of the file to produce: .../pv_name.root")

args = parser.parse_args()

import os
import sys
import time
import numpy as np
import ROOT
import pythia8

import velo
from scatter import Scatter
from writer import prtStable, heavyFlavor, Writer, hitSel, Hits


def run(lname, tEvt):
    name = "{}.root".format(lname)
    myhash = abs(hash(lname)) % 900000000
    print("Producing", tEvt, "tracks to", name, "with hash:", myhash)
    # Initialize Pythia.
    random = ROOT.TRandom3(myhash)
    pythia = pythia8.Pythia("", False)
    pythia.readString("Random:seed = {}".format(myhash))
    pythia.readString("Print:quiet = on")
    pythia.readString("SoftQCD:all = on")
    pythia.init()
    module = velo.ModuleMaterial("dat/run3.root")
    rffoil = velo.FoilMaterial("dat/run3.root")
    scatter = Scatter()

    # Create the output TFile and TTree.
    tfile = ROOT.TFile(name, "RECREATE")
    ttree = ROOT.TTree("data", "")

    # New interface
    # Create the writer handler and add branches
    writer = Writer(ttree)
    writer.add("pvr_x")
    writer.add("pvr_y")
    writer.add("pvr_z")
    writer.add("svr_x")
    writer.add("svr_y")
    writer.add("svr_z")
    writer.add("svr_pvr")
    writer.add("hit_x")
    writer.add("hit_y")
    writer.add("hit_z")
    writer.add("hit_prt")
    writer.add("prt_pid")
    writer.add("prt_px")
    writer.add("prt_py")
    writer.add("prt_pz")
    writer.add("prt_e")
    writer.add("prt_x")
    writer.add("prt_y")
    writer.add("prt_z")
    writer.add("prt_hits")
    writer.add("prt_pvr")
    writer.add("ntrks_prompt")


    number_rejected_events = 0

    # Fill the events.

    ttime = 0
    for iEvt in range(tEvt):
        start = time.time()
        ipv = 0
        npv = 0
        target_npv = np.random.poisson(7.6)
        iEvt += 1

        while npv < target_npv:
            if not pythia.next():
                continue

            # All distance measurements are in units of mm
            xPv, yPv, zPv = (
                random.Gaus(0, 0.055),
                random.Gaus(0, 0.055),
                random.Gaus(100, 63),
            )  # normal LHCb operation

            # pvr x and y spead can be found https://arxiv.org/pdf/1410.0149.pdf page 42. z dependent
            ## [-1000,-750, -500, -250] # mm

            writer["pvr_x"].append(xPv)
            writer["pvr_y"].append(yPv)
            writer["pvr_z"].append(zPv)
            number_of_detected_particles = 0
            # find heavy flavor SVs
            for prt in pythia.event:
                if not heavyFlavor(prt.id()):
                    continue
                # TODO: require particles with hits from the SVs
                writer["svr_x"].append(prt.xDec() + xPv)
                writer["svr_y"].append(prt.yDec() + yPv)
                writer["svr_z"].append(prt.zDec() + zPv)
                writer["svr_pvr"].append(ipv)

            for prt in pythia.event:
                if not prt.isFinal or prt.charge() == 0:
                    continue
                if not prtStable(prt.id()):
                    continue
                if abs(prt.zProd()) > 1000:
                    continue
                if (prt.xProd() ** 2 + prt.yProd() ** 2) ** 0.5 > 40:
                    continue
                if prt.pAbs() < 0.1:
                    continue
                prt.xProd(
                    prt.xProd() + xPv
                )  # Need to change the origin of the event before getting the hits
                prt.yProd(prt.yProd() + yPv)
                prt.zProd(prt.zProd() + zPv)
                hits = Hits(module, rffoil, scatter, prt)
                if len(hits) == 0:
                    continue
                if len(hits) > 2 and abs(zPv - prt.zProd()) < 0.001:
                    number_of_detected_particles += 1
                    # if prt.pAbs() < 0.2: print 'slow!', prt.pAbs(), prt.id()
                writer["prt_pid"].append(prt.id())
                writer["prt_px"].append(prt.px())
                writer["prt_py"].append(prt.py())
                writer["prt_pz"].append(prt.pz())
                writer["prt_e"].append(prt.e())
                writer["prt_x"].append(prt.xProd())
                writer["prt_y"].append(prt.yProd())
                writer["prt_z"].append(prt.zProd())
                writer["prt_pvr"].append(ipv)
                writer["prt_hits"].append(len(hits))
                for xHit, yHit, zHit in hits:
                    # xHit_recorded, yHit_recorded, zHit_recorded = np.random.uniform(-0.0275,0.0275)+xHit, np.random.uniform(-0.0275,0.0275)+yHit, zHit # normal
                    xHit_recorded, yHit_recorded, zHit_recorded = (
                        np.random.normal(0, 0.012) + xHit,
                        np.random.normal(0, 0.012) + yHit,
                        zHit,
                    )  # normal
                    writer["hit_x"].append(xHit_recorded)
                    writer["hit_y"].append(yHit_recorded)
                    writer["hit_z"].append(zHit_recorded)
                    writer["hit_prt"].append(len(writer["prt_e"]) - 1)
            # if number_of_detected_particles < 5: iEvt -= 1; number_rejected_events+=1; continue
            writer["ntrks_prompt"].append(number_of_detected_particles)
            ipv += 1
            if number_of_detected_particles > 0:
                npv += 1

        itime = time.time() - start
        ttime += itime
        if iEvt % 100 == 0:
          print(
            "{} Evt {}/{}, {:3} PVs, {:3} tracks in {:.3} s".format(
                name, iEvt, tEvt, npv, len(writer["pvr_z"]), itime
            )
        )
        writer.write()

    # Write and close the TTree and TFile.
    ttree.Print()
    ttree.Write(ttree.GetName(), ROOT.TObject.kOverwrite)
    tfile.Close()
    return ttime


if __name__ == "__main__":
    start = time.time()
    if args.threads == 1:
        stime = run(args.name, args.events)
    else:
        from concurrent.futures import ProcessPoolExecutor as PoolExecutor

        with PoolExecutor(max_workers=args.threads) as pool:
            futures = [
                pool.submit(run, "{}_{}".format(args.name, i), args.events)
                for i in range(args.start, args.start + args.threads)
            ]
            stime = sum(f.result() for f in futures)

    fulltime = time.time() - start
    print(
        "Computed {} events in {:.5} s (Threads time: {:.6} s) ({:.4} event/s)".format(
            args.events * args.threads,
            stime,
            fulltime,
            args.events * args.threads / fulltime,
        )
    )
