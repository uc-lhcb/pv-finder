#!/usr/bin/env python

from __future__ import print_function

from math import *
import numpy
try:
    import uproot3 as uproot
except ModuleNotFoundError:
    import uproot

tree = uproot.open("../dat/test_100pvs.root")["data"]

## the keys() method prints out the names of allthe ROOT NTUPLE branches
print(tree.keys())

##  there is probably a more elegant way to create the jagged arrays
##  corresponding to the branches of the ROOT Tuple, but brute
## force should work here

##  these are the primary vertex (x,y,z) coordinates
pvr_x = tree["pvr_x"].array()
pvr_y = tree["pvr_y"].array()
pvr_z = tree["pvr_z"].array()

##  these are the secondary vertex (x,y,z) coordinates
svr_x = tree["svr_x"].array()
svr_y = tree["svr_y"].array()
svr_z = tree["svr_z"].array()

##  these are the individual hit (x,y,z) coordinates
hit_x = tree["hit_x"].array()
hit_y = tree["hit_y"].array()
hit_z = tree["hit_z"].array()
hit_prt = tree["hit_prt"].array()

##  the following are "particle" (track) quantities
##  "_pid" refers to particle ID (type, using integer values according to the PDG)
##  "_px, _py, _pz are the momenta of the particle in GeV; _e is the energy
##     the momenta can be used to determine the particle's direction
##  "-hits" is the number of hits associated with a particle
##  "_pvr" is the index of the primary vertex (within an event)
prt_pid = tree["prt_pid"].array()
prt_px = tree["prt_px"].array()
prt_py = tree["prt_py"].array()
prt_pz = tree["prt_pz"].array()
prt_e = tree["prt_e"].array()
prt_x = tree["prt_x"].array()
prt_y = tree["prt_y"].array()
prt_z = tree["prt_z"].array()
prt_hits = tree["prt_hits"].array()
prt_pvr = tree["prt_pvr"].array()

## ntrks_prompt is the number of prompt tracks within an event
ntrks_prompt = tree["ntrks_prompt"].array()

## the data structures created above are jagged arrays that do not
## specifically relate to events. Each is indexed from 0 to ...
## the .starts and .stops methods return pointers to the first
## an last indices within a jagged array associated with a particular
## event.  In the code that follows, the .contents method prints
## all the elements of the pvr_x array.  The  .starts and .stops
## methods provide pointers to the first and last elements of the
## pvr_x array associated with an event
print("pvr_x.contents", pvr_x.contents)
print("pvr_x.starts", pvr_x.starts)
print("pvr_x.stop", pvr_x.stops)

## As an example, we can look at the x positions of the primary
## vertices in the first event
first_event_start = pvr_x.starts[0]
first_event_stop = pvr_x.stops[0]
event_array_of_pvr_x = pvr_x.contents[first_event_start:first_event_stop]
for x in event_array_of_pvr_x:
    print("x =", x)

##  and now do the same for the second event
print("now for the second event")
second_event_start = pvr_x.starts[1]
second_event_stop = pvr_x.stops[1]
event_array_of_pvr_x = pvr_x.contents[second_event_start:second_event_stop]
for x in event_array_of_pvr_x:
    print("x =", x)
