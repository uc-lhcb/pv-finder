#ifndef HITS_H
#define HITS_H

#include "data.h"
#include "trajectory.h"

#define SENSORS 52
#define PHI_BINS 60
#define MAX_SCATTER 0.004 // this is default in LHCb tracking
#define PI 3.14159265
#define MAX_HITS_PER_BIN 100

// some helper functions first
double getPhi(double x, double y) {
    double phi = atan2(y, x);
    if(phi < 0)
        phi += 2 * PI;
    return phi;
}

int getPhiBin(double x, double y) {
    double phi_bin_step = 2 * PI / PHI_BINS;
    double phi = atan2(y, x);
    if(phi < 0)
        phi += 2 * PI;
    return phi / phi_bin_step;
}

int getPhiBin(double phi) {
    double phi_bin_step = 2 * PI / PHI_BINS;
    return phi / phi_bin_step;
}

// nearest neighbor phi bin
int getPhiBinNN(double phi) {
    double phi_bin_step = 2 * PI / PHI_BINS;
    int phi_bin = phi / phi_bin_step;
    double dmin = phi - phi_bin * phi_bin_step;
    double dmax = (phi_bin + 1) * phi_bin_step - phi;
    int nn = phi_bin + 1;
    if(dmin < dmax)
        nn = phi_bin - 1;
    if(nn < 0)
        nn = PHI_BINS - 1;
    if(nn > PHI_BINS - 1)
        nn = 0;
    return nn;
}

double deltaPhi(double phi0, double phi1) {
    double dphi = abs(phi0 - phi1);
    if(dphi < PI)
        return dphi;
    return abs(dphi - 2 * PI);
}

// class to store a single hit
class Hit {

  private:
    Point _point;
    int _idx, _true_prt, _sensor, _phi_bin, _phi_bin_nn;
    double _r, _phi; // cache these as they're used a lot

  public:
    Hit() : _point(), _idx(-1), _true_prt(-1), _sensor(-1) {}

    Hit(int idx, int sensor, double x, double y, double z, int true_prt)
        : _point(x, y, z), _idx(idx), _true_prt(true_prt), _sensor(sensor) {
        _r = sqrt(x * x + y * y);
        _phi = getPhi(x, y);
        _phi_bin = getPhiBin(_phi);
        _phi_bin_nn = getPhiBinNN(_phi);
    }

    void set(int idx, int sensor, double x, double y, double z, int true_prt) {
        _point.set(x, y, z);
        _idx = idx;
        _sensor = sensor;
        _true_prt = true_prt;
        _r = sqrt(x * x + y * y);
        _phi = getPhi(x, y);
        _phi_bin = getPhiBin(_phi);
        _phi_bin_nn = getPhiBinNN(_phi);
    }

    const Point &point() const { return _point; }
    int index() const { return _idx; }
    int truePrt() const { return _true_prt; }
    int sensor() const { return _sensor; }
    double r() const { return _r; }
    double phi() const { return _phi; }
    int phiBin() const { return _phi_bin; }
    int phiBinNN() const { return _phi_bin_nn; }
};

// for sorting hits
bool rankHits(const Hit &h0, const Hit &h1) { return h0.r() > h1.r(); }

// singleton that contains all hits in each event
class HitArray {

  private:
    int _n[SENSORS][PHI_BINS];
    Hit *_hits[SENSORS][PHI_BINS];
    int _zo[26] = {-277, -252, -227, -202, -132, -62, -37, -12, 13,  38,  63,  88,  113,
                   138,  163,  188,  213,  238,  263, 325, 402, 497, 616, 661, 706, 751};
    int _ze[26] = {-289, -264, -239, -214, -144, -74, -49, -24, 1,   26,  51,  76,  101,
                   126,  151,  176,  201,  226,  251, 313, 390, 485, 604, 649, 694, 739};

    // make it a singleton
    static HitArray *_instance;
    HitArray() {
        for(int s = 0; s < SENSORS; s++) {
            for(int i = 0; i < PHI_BINS; i++)
                _hits[s][i] = new Hit[MAX_HITS_PER_BIN];
        }
    }
    HitArray(const HitArray &) {}
    HitArray &operator=(const HitArray &) { return *this; }

    void _reset() {
        for(int s = 0; s < SENSORS; s++) {
            for(int i = 0; i < PHI_BINS; i++)
                _n[s][i] = 0;
        }
    }

    void _addHit(Data &data, int h) {
        int s = data.hs->at(h);
        int i = getPhiBin(data.hx->at(h), data.hy->at(h));
        if(_n[s][i] == MAX_HITS_PER_BIN - 1)
            return; // just ignore it (obviously not a great solution, but OK for this test)
        _hits[s][i][_n[s][i]] = Hit(h, s, data.hx->at(h), data.hy->at(h), data.hz->at(h), data.hid->at(h));
        _n[s][i]++;
    }

  public:
    ~HitArray() {
        for(int s = 0; s < SENSORS; s++) {
            for(int i = 0; i < PHI_BINS; i++) {
                if(_hits[s][i])
                    delete[] _hits[s][i];
                _hits[s][i] = 0;
            }
        }
    }

    static HitArray *instance() {
        if(!_instance)
            _instance = new HitArray();
        return _instance;
    }

    int n(int s, int i) const { return _n[s][i]; }

    const Hit &at(int s, int i, int j) const { return _hits[s][i][j]; }

    void newEvent(Data &data) {
        int nhits = data.hz->size();
        _reset();
        for(int h = 0; h < nhits; h++)
            _addHit(data, h);
        // sort hits
        for(int s = 0; s < SENSORS; s++) {
            for(int i = 0; i < PHI_BINS; i++) {
                std::sort(_hits[s][i], _hits[s][i] + _n[s][i], rankHits);
            }
        }
    }

    double zsensor(int s) const {
        if(s % 2 == 0)
            return _ze[s / 2];
        else
            return _zo[s / 2];
    }

    void bestHit(int s,
                 const Hit &hit1,
                 const Trajectory &traj,
                 int &ii,
                 int &jj,
                 int marked[SENSORS][PHI_BINS][MAX_HITS_PER_BIN]) const {
        ii = -1;
        jj = -1;
        if(_n[s][hit1.phiBin()] < 1 && _n[s][hit1.phiBinNN()] < 1)
            return;
        double best_scatter = MAX_SCATTER;
        double z = zsensor(s), x, y, dz = abs(z - hit1.point().z());
        traj.getXY(z, x, y);
        //    if(abs(x)<5.1 && abs(y)<5.1) return;  // slower
        // switch to predicting the phi bin
        // i = getPhiBin(x,y); // this is really slow!
        double r = sqrt(x * x + y * y);
        if(r < 4.9)
            return;
        double tmp = 0.2; // should be configurable
        double rMax = r + tmp, rMin = r - tmp;
        double inv_dz2 = 1 / (dz * dz);
        int bins_to_use[2] = {hit1.phiBin(), hit1.phiBinNN()};

        for(int k = 0; k < 2; k++) {
            int bin = bins_to_use[k];
            if(_n[s][bin] < 1)
                continue;
            if(_hits[s][bin][0].r() < rMin)
                continue;
            if(_hits[s][bin][_n[s][bin] - 1].r() > rMax)
                continue;
            for(int j = 0; j < _n[s][bin]; j++) {
                const Hit &hit = _hits[s][bin][j];
                if(hit.r() > rMax)
                    continue;
                if(hit.r() < rMin)
                    break;
                if(marked[s][bin][j])
                    continue;
                if(deltaPhi(hit1.phi(), hit.phi()) > 0.08)
                    continue;
                double dx = hit.point().x() - x;
                double dy = hit.point().y() - y;
                double scatter = (dx * dx + dy * dy) * inv_dz2;
                if(scatter < best_scatter) {
                    ii = bin;
                    jj = j;
                    best_scatter = scatter;
                }
            }
        }
    }

    double fracMarked(int marked[SENSORS][PHI_BINS][MAX_HITS_PER_BIN]) const {
        int ntot = 0, nmarked = 0;
        for(int s = 0; s < SENSORS; s++) {
            for(int i = 0; i < PHI_BINS; i++) {
                for(int j = 0; j < _n[s][i]; j++) {
                    ntot++;
                    if(marked[s][i][j])
                        nmarked++;
                }
            }
        }
        return nmarked / (double)ntot;
    }
};

HitArray *HitArray::_instance = 0;

#endif /* HITS_H */
