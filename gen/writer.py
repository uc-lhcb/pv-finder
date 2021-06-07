from __future__ import annotations
import ROOT


def prtStable(pid):
    return abs(pid) in (211, 321, 11, 13, 2212)


def heavyFlavor(pid):
    return abs(pid) in (411, 421, 431, 4122, 511, 521, 531, 5122)


class Vector:
    def __init__(self) -> None:
        self.this = ROOT.vector("double")()

    def append(self, val: float | None) -> None:
        self.this.push_back(0 if val is None else val)

    def __len__(self) -> int:
        return self.this.size()

    def clear(self) -> None:
        self.this.clear()


class Writer:
    def __init__(self, tree: ROOT.TTree) -> None:
        self.tree = tree
        self.vars = {}
        self.null = ROOT.vector("double")(1, 0)

    def __getitem__(self, key: str) -> Vector:
        return self.vars[key]

    def add(self, *vars: str) -> None:
        for var in vars:
            self.vars[var] = Vector()
            self.tree.Branch(var, self.vars[var].this)

    def clear(self) -> None:
        for val in self.vars.values():
            val.clear()

    def write(self) -> None:
        self.tree.Fill()
        self.clear()


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
