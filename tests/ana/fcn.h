#pragma once

#include "data/corenhits.h"
#include "data/corepvs.h"
#include "data/coretruthtracks.h"
#include "tracks.h"
#include "trajectory.h"
#include "triplet.h"

#include <TMinuit.h>

// This is an (ugly) global pointer so that minuit can run a plain function
extern AnyTracks *fcn_global_tracks;

// for TMinuit
// TODO: would be faster with user-supplied derivatives
inline void FCN(Int_t &num_par, Double_t *grad, Double_t &f, Double_t *pars, Int_t iflag) {

    // A 3D point (custom class) for the current location
    Point pv(pars[0], pars[1], pars[2]);

    double sum1 = 0;
    double sum2 = 0;

    // Grab the global instance. This is ugly - can be solved by using Minuit2
    AnyTracks *tracks = fcn_global_tracks;

    for(int i = tracks->tmin(); i <= tracks->tmax(); i++) {
        // Compute the PDF for the current track at the current location (by integer track number)
        double pdf = tracks->at(i).pdf(pv);

        // Sum PDF
        sum1 += pdf;
        sum2 += pdf * pdf;
    }

    // Avoid really small values
    f = (sum1 < 1.e-9) ? 0 : -(sum1 - sum2 / sum1);
}

// kernel value at point pv
inline double kernel(const Point &pv) {
    if(!fcn_global_tracks->run())
        return 0;
    int num_par = 3;
    int iflag = 0;
    double grad[3];
    double f;
    double pars[3];

    pars[0] = pv.x();
    pars[1] = pv.y();
    pars[2] = pv.z();

    FCN(num_par, grad, f, pars, iflag);
    return -f;
}

// gradient decent in x-y to find max kernel value for fixed pv.z()
// pv.x() and pv.y() are start values, then filled with x,y best fit
inline double kernelMax(Point &pv) {

    AnyTracks *tracks = fcn_global_tracks;
    tracks->setRange(pv.z());
    if(!tracks->run())
        return 0;

    double arglist[10], amin, edm, errdef;
    int iflag, nvpar, nparx;

    static TMinuit *min = 0;
    if(min == 0) {
        min = new TMinuit();
        arglist[0] = -1;
        min->mnexcm("SET PRINT", arglist, 1, iflag);
        min->mnexcm("SET NOW", arglist, 0, iflag);
        min->SetFCN(FCN);
    }

    // reset
    min->mnrset(1);

    min->mnparm(0, "PVX", pv.x(), 0.01, -10 * 0.055, 10 * 0.055, iflag);
    min->mnparm(1, "PVY", pv.y(), 0.01, -10 * 0.055, 10 * 0.055, iflag);
    min->mnparm(2, "PVZ", pv.z(), 0, 0, 0, iflag);

    arglist[0] = 1000;
    arglist[1] = 0.1;
    min->mnexcm("MIGRAD", arglist, 2, iflag);

    double x, y, tmp;
    min->GetParameter(0, x, tmp);
    min->GetParameter(1, y, tmp);
    min->mnstat(amin, edm, errdef, nvpar, nparx, iflag);

    pv.set(x, y, pv.z());
    return -amin;
}

inline int
ntrkInAcc(const CoreTruthTracksIn &data_trks, const CorePVsIn &data_pvs, const CoreNHitsIn &data_hits, int i) {
    int ntrk_in_acc = 0;
    int nprt = data_pvs.prt_pvr->size();

    for(int j = 0; j < nprt; j++) {
        if(data_pvs.prt_pvr->at(j) != i)
            continue;
        if(data_hits.prt_hits->at(j) < 3)
            continue;
        if(abs(data_trks.prt_z->at(j) - data_pvs.pvr_z->at(i)) > 0.001)
            continue;
        TVector3 p3(data_trks.prt_px->at(j), data_trks.prt_py->at(j), data_trks.prt_pz->at(j));
        if(p3.Eta() < 2 || p3.Eta() > 5)
            continue;
        if(p3.Mag() < 3)
            continue;

        ntrk_in_acc++;
    }
    return ntrk_in_acc;
}

// -1: < 2 particles made hits
// 0: < 5 long tracks
// 1: LHCb pv
inline int
pvCategory(const CoreTruthTracksIn &data_trks, const CorePVsIn &data_pvs, const CoreNHitsIn &data_hits, int i) {
    if(data_trks.ntrks_prompt->at(i) < 2)
        return -1;

    int ntrk_in_acc = ntrkInAcc(data_trks, data_pvs, data_hits, i);

    if(ntrk_in_acc < 5)
        return 0;
    else
        return 1;
}

inline int nSVPrt(const CoreTruthTracksIn &data_trks, const CorePVsIn &data_pvs, const CoreNHitsIn &data_hits, int i) {
    int nsv_prt = 0, nprt = data_pvs.prt_pvr->size();
    for(int j = 0; j < nprt; j++) {
        if(data_hits.prt_hits->at(j) < 3)
            continue;
        if(abs(data_trks.prt_z->at(j) - data_pvs.svr_z->at(i)) > 0.001)
            continue;
        nsv_prt++;
    }
    return nsv_prt;
}

// -1: no particles made hits
// 0: 1 particle with hits
// 1: 2+ (an actual SV)
inline int
svCategory(const CoreTruthTracksIn &data_trks, const CorePVsIn &data_pvs, const CoreNHitsIn &data_hits, int i) {

    int nsv_prt = nSVPrt(data_trks, data_pvs, data_hits, i);
    if(nsv_prt < 1)
        return -1;
    if(nsv_prt < 2)
        return 0;
    else
        return 1;
}
