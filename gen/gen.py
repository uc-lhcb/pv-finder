#!/usr/bin/env python

from __future__ import print_function
import argparse
import sys

parser = argparse.ArgumentParser(description='Generate some traks')
parser.add_argument('--num', type=int, default=10000,
                            help='an integer for the accumulator')
parser.add_argument('name',
                            help='The name of the file to produce: .../pv_name.root')
args = parser.parse_args()
tEvt = args.num
name = '/data/schreihf/PvFinder/pv_{}.root'.format(args.name)
print('Producing', tEvt, 'tracks to', name)

import os
import pythia8
import ROOT
import numpy as np
import velo

ROOT.gROOT.LoadMacro('scatter.C')
Scatter = ROOT.Scatter

def prtStable(pid):
    return abs(pid) in (211, 321, 11, 13, 2212)

def heavyFlavor(pid):
    return abs(pid) in (411, 421, 431, 4122, 511, 521, 531, 5122)

# Writer class.
class Writer():
    def __init__(self):
        from collections import OrderedDict
        self.vars = OrderedDict()
        self.null = ROOT.vector('double')(1, 0)

    def init(self, tree):
        for key, val in self.vars.iteritems(): tree.Branch(key, val)

    def add(self, var):
        self.vars[var] = ROOT.vector('double')()

    def var(self, var, val = None, idx = -2):
        if not var in self.vars: return self.null.back()
        var = self.vars[var]
        if idx < -1: var.push_back(0 if val == None else val)
        if idx < 0: idx = var.size() - 1
        elif idx >= var.size(): idx = -1
        if idx < 0: return self.null[0]
        if val != None: var[idx] = val
        return var[idx]

    def size(self, var):
        return self.vars[var].size()

    def clear(self):
        for key, val in self.vars.iteritems(): val.clear()

def hitSel(mhit, fhit, pz):
    hit = mhit
    hit_type = -1
    if mhit.T() != 0: hit_type = 0
    if mhit.T() == 0 and fhit.T() != 0:
        hit = fhit
        hit_type = 1
    elif fhit.T() != 0 and (pz/abs(pz))*fhit.Z() < (pz/abs(pz))*mhit.Z():
        hit = fhit
        hit_type = 1
    return [hit.X(), hit.Y(), hit.Z(), hit_type]

def Hits(module, rffoil, scatter, prt):
    hits = []
    p = prt.pAbs()
    if p == 0: return hits
    vx, vy, vz = prt.xProd(), prt.yProd(), prt.zProd(),
    px, py, pz = prt.px()/p, prt.py()/p, prt.pz()/p
    p3 = ROOT.TVector3(prt.px(),prt.py(),prt.pz())
    nrf = 0
    mhit = module.intersect(vx, vy, vz, px, py, pz)
    fhit = rffoil.intersect(vx, vy, vz, px, py, pz)
    hit = hitSel(mhit,fhit,pz)
    while hit[3] >= 0:
        vx, vy, vz = [hit[0],hit[1],hit[2]]
        if hit[3] == 0: hits += [[vx, vy, vz]]
        fx0 = 0.01
        if hit[3] > 0:
            nrf += 1
            fx0 = 0.005
        p3 = scatter.smear(p3,fx0)
        px, py, pz = p3.X()/p3.Mag(), p3.Y()/p3.Mag(), p3.Z()/p3.Mag()
        vx, vy, vz = vx + px*0.1, vy + py*0.1, vz + pz*0.1
        mhit = module.intersect(vx, vy, vz, px, py, pz)
        fhit = rffoil.intersect(vx, vy, vz, px, py, pz)
        hit = hitSel(mhit,fhit,pz)
    return hits

# Initialize Pythia.
random = ROOT.TRandom3()
pythia = pythia8.Pythia('', False)
pythia.readString('Print:quiet = on')
pythia.readString('SoftQCD:all = on')
pythia.init()
module = velo.ModuleMaterial('dat/run3.root')
rffoil = velo.FoilMaterial('dat/run3.root')
scatter = Scatter()

# Create the output TFile and TTree.
tfile = ROOT.TFile(name, 'RECREATE')
ttree = ROOT.TTree('data', '')

# Create the writer handler, add branches, and initialize.
writer = Writer()
writer.add('pvr_x')
writer.add('pvr_y')
writer.add('pvr_z')
writer.add('svr_x')
writer.add('svr_y')
writer.add('svr_z')
writer.add('svr_pvr')
writer.add('hit_x')
writer.add('hit_y')
writer.add('hit_z')
writer.add('hit_prt')
writer.add('prt_pid')
writer.add('prt_px')
writer.add('prt_py')
writer.add('prt_pz')
writer.add('prt_e')
writer.add('prt_x')
writer.add('prt_y')
writer.add('prt_z')
writer.add('prt_hits')
writer.add('prt_pvr')
writer.add('ntrks_prompt')
writer.init(ttree)

number_rejected_events = 0

# Fill the events.

for iEvt in range(tEvt):
    ipv = 0
    npv = 0
    target_npv = np.random.poisson(7.6)
    iEvt += 1

    while npv < target_npv:
        if not pythia.next():
            continue

        # All distance measurements are in units of mm
        xPv, yPv, zPv = random.Gaus(0, 0.055), random.Gaus(0, 0.055), random.Gaus(100, 63) # normal LHCb operation

        #pvr x and y spead can be found https://arxiv.org/pdf/1410.0149.pdf page 42. z dependent
        ## [-1000,-750, -500, -250] # mm

        writer.var('pvr_x', xPv)
        writer.var('pvr_y', yPv)
        writer.var('pvr_z', zPv)
        number_of_detected_particles = 0
        # find heavy flavor SVs
        for prt in pythia.event:
            if not heavyFlavor(prt.id()): continue
            # TODO: require particles with hits from the SVs
            writer.var('svr_x', prt.xDec() + xPv)
            writer.var('svr_y', prt.yDec() + yPv)
            writer.var('svr_z', prt.zDec() + zPv)
            writer.var('svr_pvr', ipv)

        for prt in pythia.event:
            if not prt.isFinal or prt.charge() == 0: continue
            if not prtStable(prt.id()): continue
            if abs(prt.zProd()) > 1000: continue
            if (prt.xProd()**2 + prt.yProd()**2)**0.5 > 40: continue
            if prt.pAbs() < 0.1: continue
            prt.xProd(prt.xProd() + xPv) # Need to change the origin of the event before getting the hits
            prt.yProd(prt.yProd() + yPv)
            prt.zProd(prt.zProd() + zPv)
            hits = Hits(module, rffoil, scatter, prt)
            if len(hits) == 0: continue
            if len(hits) > 2 and abs(zPv-prt.zProd()) < 0.001:
                number_of_detected_particles += 1
                #if prt.pAbs() < 0.2: print 'slow!', prt.pAbs(), prt.id()
            writer.var('prt_pid', prt.id())
            writer.var('prt_px',  prt.px())
            writer.var('prt_py',  prt.py())
            writer.var('prt_pz',  prt.pz())
            writer.var('prt_e',   prt.e())
            writer.var('prt_x',   prt.xProd())
            writer.var('prt_y',   prt.yProd())
            writer.var('prt_z',   prt.zProd())
            writer.var('prt_pvr',   ipv)
            writer.var('prt_hits',  len(hits))
            for xHit, yHit, zHit in hits:
                #xHit_recorded, yHit_recorded, zHit_recorded = np.random.uniform(-0.0275,0.0275)+xHit, np.random.uniform(-0.0275,0.0275)+yHit, zHit # normal
                xHit_recorded, yHit_recorded, zHit_recorded = np.random.normal(0,0.012)+xHit, np.random.normal(0,0.012)+yHit, zHit # normal
                writer.var('hit_x', xHit_recorded)
                writer.var('hit_y', yHit_recorded)
                writer.var('hit_z', zHit_recorded)
                writer.var('hit_prt', writer.size('prt_e') - 1)
        #if number_of_detected_particles < 5: iEvt -= 1; number_rejected_events+=1; continue
        writer.var('ntrks_prompt',number_of_detected_particles)
        ipv += 1
        if number_of_detected_particles > 0:
            npv += 1

    ttree.Fill()
    print("Number of PVs:", npv)
    print("size: ", writer.size('pvr_z'))
    print("Event : ", iEvt, " / ", tEvt, " ")
    writer.clear()

# Write and close the TTree and TFile.
ttree.Print()
ttree.Write(ttree.GetName(), ROOT.TObject.kOverwrite)
tfile.Close()
