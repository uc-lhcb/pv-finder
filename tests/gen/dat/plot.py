#!/bin/env python
import os, sys

pre = os.path.abspath("../")
cwd = os.getcwd()
sys.path.insert(0, pre)
os.chdir(pre)
import velo, ROOT

os.chdir(cwd)

# Create the material instances.
runs, mms, fms = ["run1", "run2", "run3"], {}, {}
for run in runs:
    mms[run] = velo.ModuleMaterial("%s.root" % run)
    fms[run] = velo.FoilMaterial("%s.root" % run)

# Return the foil for a given plane and variable range.
def foil(run, half, plane, u, vmin, vmax):
    g = ROOT.TGraph(1000)
    g.SetLineWidth(2)
    dv = (vmax - vmin) / 1000.0
    for i in range(0, 1000):
        v = vmin + dv + i * dv
        if plane == "xy":
            g.SetPoint(i, v, fms[run].x(v, u, half))
        else:
            g.SetPoint(i, v, fms[run].x(u, v, half))
    g.SetLineColor(ROOT.kCyan)
    g.SetLineWidth(2)
    return g


# Return the module for a given plane and variable range.
def module(run, half, plane, u, vmin, vmax):
    s = mms[run].sensor(u, half)
    if plane == "xy":
        dv, ups, lps = (vmax - vmin) / 1000.0, [], []
        for i in range(0, 1000):
            uv, lv = vmin + dv + i * dv, vmax - dv - i * dv
            up, lp = (uv, mms[run].x(s, uv, 1)), (lv, mms[run].x(s, lv, -1))
            if up[1] > mms[run].x(s, uv, -1):
                ups += [up]
            if lp[1] < mms[run].x(s, lv, 1):
                lps += [lp]
        ps = ups + lps + [ups[0]]
        g = ROOT.TGraph(len(ps))
        for i, (x, y) in enumerate(ps):
            g.SetPoint(i, x, y)
    else:
        n, g = 0, ROOT.TGraph((mms[run].sensor(800) + 1) * 3 / 2)
        for s in range(0, mms[run].sensor(800) + 1):
            if mms[run].x(s, 0) * half < 0:
                continue
            z = mms[run].z(s)
            g.SetPoint(n, z, half * 1e3)
            g.SetPoint(n + 1, z, mms[run].x(s, u))
            g.SetPoint(n + 2, z, half * 1e3)
            n = n + 3
        g.Set(n)
    g.SetLineColor(ROOT.kMagenta)
    g.SetLineWidth(2)
    return g


# Plot in the xz-plane.
def plotY(run, y, zmin=-300, zmax=800):
    gs = [foil(run, h, "xz", y, zmin, zmax) for h in [-1, 1]]
    gs += [module(run, h, "xz", y, zmin, zmax) for h in [-1, 1]]
    gs[0].Draw()
    cnv = ROOT.gPad.GetCanvas()
    frm = cnv.DrawFrame(zmin, -15, zmax, 15)
    frm.GetXaxis().SetTitle("#it{z} [mm]")
    frm.GetXaxis().CenterTitle()
    frm.GetYaxis().SetTitle("#it{x} [mm]")
    frm.GetYaxis().CenterTitle()
    frm.GetYaxis().SetTitleOffset(0.5)
    frm.SetTitle("#it{y} = %i mm" % y)
    for g in gs:
        g.Draw("SAME L")
    cnv.SetCanvasSize(1200, 600)
    cnv.SetLeftMargin(0.05)
    cnv.SetRightMargin(0.05)
    ROOT.gPad.Print("%sy%i.pdf" % (run, y))


# Plot in the xy-plane.
def plotZ(run, z, ymin=-40, ymax=40, mod=True):
    gs = [foil(run, h, "xy", z, ymin, ymax) for h in [-1, 1]]
    if mod:
        gs += [module(run, 0, "xy", z, ymin, ymax)]
    gs[0].Draw()
    cnv = ROOT.gPad.GetCanvas()
    frm = cnv.DrawFrame(ymin, ymin, ymax, ymax)
    frm.GetXaxis().SetTitle("#it{y} [mm]")
    frm.GetXaxis().CenterTitle()
    frm.GetYaxis().SetTitle("#it{x} [mm]")
    frm.GetYaxis().CenterTitle()
    frm.SetTitle("#it{z} = %i mm" % z)
    for g in gs:
        g.Draw("SAME L")
    cnv.SetCanvasSize(600, 600)
    ROOT.gPad.Print("%sz%i.pdf" % (run, z))


plotZ("run3", 1)
plotZ("run3", 7, mod=False)
plotZ("run3", 13)
plotY("run3", 0, 200, 350)
