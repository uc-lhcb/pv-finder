import ROOT


def prtStable(pid):
    return abs(pid) in (211, 321, 11, 13, 2212)


def heavyFlavor(pid):
    return abs(pid) in (411, 421, 431, 4122, 511, 521, 531, 5122)


# Writer class.
class Writer:
    def __init__(self):
        from collections import OrderedDict

        self.vars = OrderedDict()
        self.null = ROOT.vector("double")(1, 0)

    def init(self, tree):
        for key, val in self.vars.items():
            tree.Branch(key, val)

    def add(self, var):
        self.vars[var] = ROOT.vector("double")()

    def var(self, var, val=None, idx=-2):
        if not var in self.vars:
            return self.null.back()
        var = self.vars[var]
        if idx < -1:
            var.push_back(0 if val == None else val)
        if idx < 0:
            idx = var.size() - 1
        elif idx >= var.size():
            idx = -1
        if idx < 0:
            return self.null[0]
        if val != None:
            var[idx] = val
        return var[idx]

    def size(self, var):
        return self.vars[var].size()

    def clear(self):
        for key, val in self.vars.items():
            val.clear()


def hitSel(mhit, fhit, pz):
    hit = mhit
    hit_type = -1
    if mhit.T() != 0:
        hit_type = 0
    if mhit.T() == 0 and fhit.T() != 0:
        hit = fhit
        hit_type = 1
    elif fhit.T() != 0 and (pz / abs(pz)) * fhit.Z() < (pz / abs(pz)) * mhit.Z():
        hit = fhit
        hit_type = 1
    return [hit.X(), hit.Y(), hit.Z(), hit_type]


def Hits(module, rffoil, scatter, prt):
    hits = []
    p = prt.pAbs()
    if p == 0:
        return hits
    vx, vy, vz = prt.xProd(), prt.yProd(), prt.zProd()
    px, py, pz = prt.px() / p, prt.py() / p, prt.pz() / p
    p3 = ROOT.TVector3(prt.px(), prt.py(), prt.pz())
    nrf = 0
    mhit = module.intersect(vx, vy, vz, px, py, pz)
    fhit = rffoil.intersect(vx, vy, vz, px, py, pz)
    hit = hitSel(mhit, fhit, pz)
    while hit[3] >= 0:
        vx, vy, vz = [hit[0], hit[1], hit[2]]
        if hit[3] == 0:
            hits += [[vx, vy, vz]]
        fx0 = 0.01
        if hit[3] > 0:
            nrf += 1
            fx0 = 0.005
        p3 = scatter.smear(p3, fx0)
        px, py, pz = p3.X() / p3.Mag(), p3.Y() / p3.Mag(), p3.Z() / p3.Mag()
        vx, vy, vz = vx + px * 0.1, vy + py * 0.1, vz + pz * 0.1
        mhit = module.intersect(vx, vy, vz, px, py, pz)
        fhit = rffoil.intersect(vx, vy, vz, px, py, pz)
        hit = hitSel(mhit, fhit, pz)
    return hits
