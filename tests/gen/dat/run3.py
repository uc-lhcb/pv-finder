#!/bin/env python
import ROOT

# Geometry output file.
tfile = ROOT.TFile("run3.root", "RECREATE")

# Write the function, with parameters and errors, to file.
def write(pre, fnc, par):
    f = ROOT.TObjString(fnc)
    p, e = ROOT.TVectorD(len(par)), ROOT.TVectorD(len(par))
    for i, v in enumerate(par):
        p[i], e[i] = v, 0
    f.Write("%s_fnc" % pre, ROOT.TObject.kOverwrite)
    p.Write("%s_par" % pre, ROOT.TObject.kOverwrite)
    e.Write("%s_err" % pre, ROOT.TObject.kOverwrite)


###############################################################################
# Create the module geometry.
#
# The module edge expressions can take any form which fulfills the
# following requirements:
# (1) The first parameter, [0], is the y-center for global shifts.
# (2) The second parameter, [1], is the x-center for global shifts.
# (3) The inner edge must have more parameters than the outer edge.
# (4) Outside the module, the lower edge > upper edge.
#
# The geometry of the LHCb upgrade VELO TDR, LHCB-TDR-013, is used
# here. Parameter [2] sets the edge outside the module. The remaining
# parameters set the y,x-points.
###############################################################################
# Expression for inner and outer module edges as a function of y.
xexp = (
    "((x-[0])<[3])*[2]"  # Outside tiles at -y.
    "+((x-[0])>=[3]&&(x-[0])<[5])*[4]"  # Begin first tile pair.
    "+((x-[0])>=[5]&&(x-[0])<[7])*[6]"  # Begin second tile pair.
    "+((x-[0])>[7])*[2]"  # Outside tiles at +y.
    "+[1]"
)  # Global x-offset.

# Upper module upper/lower edge points and z-positions.
upu = [0, 0, -100, -33.26, 37.41, -5.1, 33.26, 37.36]
upl = [0, 0, -50, -33.26, -5.1, -5.1, 5.1, 37.36]
uzs = [
    -277,
    -252,
    -227,
    -202,
    -132,
    -62,
    -37,
    -12,
    13,
    38,
    63,
    88,
    113,
    138,
    163,
    188,
    213,
    238,
    263,
    325,
    402,
    497,
    616,
    661,
    706,
    751,
]

# Lower module upper/lower edge points and z-positions.
lpu = [0, 0, -100, -37.36, -5.1, 5.1, 5.1, 33.26]
lpl = [0, 0, -50, -37.36, -33.26, 5.1, -37.41, 33.26]
lzs = [
    -289,
    -264,
    -239,
    -214,
    -144,
    -74,
    -49,
    -24,
    1,
    26,
    51,
    76,
    101,
    126,
    151,
    176,
    201,
    226,
    251,
    313,
    390,
    485,
    604,
    649,
    694,
    739,
]

# Write the modules to file.
zexp = "[0]*TMath::Gaus(x,[1],[2])+[3]*TMath::Gaus(x,[4],[5])"
for i, (lz, uz) in enumerate(zip(lzs, uzs)):

    # Lower module.
    pre = "module_%02i" % (2 * i)
    write(pre + "_a_z", zexp, [0, 0, 0, 0, lz, 0])
    write(pre + "_0_l", xexp, lpl)
    write(pre + "_0_u", xexp + "+[8]", lpu + [0])

    # Upper module.
    pre = "module_%02i" % (2 * i + 1)
    write(pre + "_a_z", zexp, [0, 0, 0, 0, uz, 0])
    write(pre + "_0_l", xexp + "+[8]", upl + [0])
    write(pre + "_0_u", xexp, upu)

###############################################################################
# Create the foil geometry.
#
# The foil expression in the xy-plane must fulfill:
# (1) The first parameter, [0], is the y-center for global shifts.
# (2) The second parameter, [1], is the x-center for global shifts.
# (3) Each parameter must have a z-dependent function.
#
# Here, parameters [2], [3], and [4] give the foil L-shape, while [5]
# gives the z-dependent shift. Consequently, all parameters except [5]
# are constant. For the run 1/2 geometries, the foil is split into
# segments with differing periodicity. The same could be done here,
# but a single segment is simpler.
###############################################################################
# Expression for foil L-shape in the xy-plane as a function of y.
xexp = "(((x-[0])<[2])?[3]:[4])+[1]+[5]"

# Expression for the upper/lower foil as a function of z (parameter 5).
uexp = "10+" + "+".join(  # Flat section.
    [
        "(x>[%i]-3.5&&x<=[%i])*(-6.5/3.5*(x-[%i]+3.5))"  # Slope to tip.
        "+(x>[%i]&&x<=[%i]+3.5)*(6.5/3.5*(x-[%i])-6.5)"  # Slope from tip.
        % tuple([i] * 6)
        for i in range(0, len(uzs))
    ]
)
lexp = "-10+" + "+".join(  # Flat section.
    [
        "(x>[%i]-3.5&&x<=[%i])*(6.5/3.5*(x-[%i]+3.5))"  # Slope to tip.
        "+(x>[%i]&&x<=[%i]+3.5)*(-6.5/3.5*(x-[%i])+6.5)"  # Slope from tip.
        % tuple([i] * 6)
        for i in range(0, len(uzs))
    ]
)

# Write the upper foil to file.
fnc = ROOT.TObjString(xexp)
lim = ROOT.TVectorD(2)
lim[0], lim[1] = -300, 800
fnc.Write("foil_u_c_fnc", ROOT.TObject.kOverwrite)
lim.Write("foil_u_c_lim", ROOT.TObject.kOverwrite)
write("foil_u_c_0", "[0]", [0])
write("foil_u_c_1", "[0]", [0])
write("foil_u_c_2", "[0]", [-3.5])
write("foil_u_c_3", "[0]", [-10.2])
write("foil_u_c_4", "[0]", [0])
write("foil_u_c_5", uexp, uzs)

# Write the lower foil to file.
fnc = ROOT.TObjString(xexp)
lim = ROOT.TVectorD(2)
lim[0], lim[1] = -300, 800
fnc.Write("foil_l_c_fnc", ROOT.TObject.kOverwrite)
lim.Write("foil_l_c_lim", ROOT.TObject.kOverwrite)
write("foil_l_c_0", "[0]", [0])
write("foil_l_c_1", "[0]", [0])
write("foil_l_c_2", "[0]", [3.5])
write("foil_l_c_3", "[0]", [0])
write("foil_l_c_4", "[0]", [10.2])
write("foil_l_c_5", lexp, lzs)
tfile.Close()
