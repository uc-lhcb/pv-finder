#ifndef TRACKLETS_H
#define TRACKLETS_H

#include "hits.h"
#include "trajectory.h"

#define MAX_TRACK_HITS 52
#define FRAC_UNUSED 0.5

class HitIdx {
  public:
    int sensor, phi_bin, idx;
    HitIdx() : sensor(-1), phi_bin(-1), idx(-1) {}
    HitIdx(int s, int phi, int i) : sensor(s), phi_bin(phi), idx(i) {}
};

const Hit &getHit(const HitIdx &h) { return HitArray::instance()->at(h.sensor, h.phi_bin, h.idx); }

class Tracklet {

  private:
    int _n;
    HitIdx _hits[MAX_TRACK_HITS];
    Trajectory _trajectory;
    double _chi2;
    Point _beam_poca;
    int _good, _true_prt;

    void _init() {
        double x, y, z, tx, ty;
        Tracklet::regression(_n, _hits, x, y, z, tx, ty, _chi2);
        _trajectory = Trajectory(x, y, z, tx, ty);
        _beam_poca = _trajectory.beamPOCA();

        _good = 1;
        _true_prt = getHit(_hits[0]).truePrt();
        for(int i = 1; i < _n; i++) {
            if(getHit(_hits[i]).truePrt() != _true_prt) {
                _good = 0;
                _true_prt = -1;
                break;
            }
        }
    }

  public:
    Tracklet() : _n(0), _chi2(-1) {}

    Tracklet(const HitIdx &h0, const HitIdx &h1, const HitIdx &h2) {
        _n = 3;
        _hits[0] = h0;
        _hits[1] = h1;
        _hits[2] = h2;
        _init();
    }

    Tracklet(int n, HitIdx hits[]) {
        _n = n;
        for(int i = 0; i < n; i++)
            _hits[i] = hits[i];
        _init();
    }

    const Hit &hit(int h) const { return getHit(_hits[h]); }
    const HitIdx &hitIdx(int h) const { return _hits[h]; }

    const Point &beamPOCA() const { return _beam_poca; }
    const Trajectory &trajectory() const { return _trajectory; }
    double chi2NDof() const { return _chi2 / _n; }
    bool good() const { return _good; }
    int truePrt() const { return _true_prt; }

    void print() const {
        cout << "tracklet: " << _n << " ";
        for(int i = 0; i < _n; i++) {
            const Hit &hit = getHit(_hits[i]);
            cout << hit.index() << "(" << hit.truePrt() << ") ";
        }
        cout << _chi2 / _n << endl;
    }

    static void
    regression(int nhits, HitIdx hits[], double &x0, double &y0, double &z0, double &tx, double &ty, double &chi2) {
        // analytically solve for track parameters
        double sumZ = 0, sumX = 0, sumY = 0, sumZX = 0, sumZY = 0, sumZ2 = 0;
        z0 = 100; // arbitrary (choose center of collision region for track state)
        for(int i = 0; i < nhits; i++) {
            const Point &hit = getHit(hits[i]).point();
            double x = hit.x(), y = hit.y(), z = hit.z(), dz = z - z0;
            sumZ += dz;
            sumX += x;
            sumY += y;
            sumZX += dz * x;
            sumZY += dz * y;
            sumZ2 += dz * dz;
        }
        double xMean = sumX / nhits;
        double yMean = sumY / nhits;
        double zMean = sumZ / nhits;
        double denominator = sumZ2 - sumZ * zMean;
        tx = (sumZX - sumZ * xMean) / denominator;
        x0 = xMean - tx * zMean;
        ty = (sumZY - sumZ * yMean) / denominator;
        y0 = yMean - ty * zMean;

        chi2 = 0;
        if(nhits == 2)
            return;

        // TODO: assumes hits have constant resolution (could add this to Hit)
        double sig_hit = 0.012;
        for(int i = 0; i < nhits; i++) {
            const Point &hit = getHit(hits[i]).point();
            double xx = tx * (hit.z() - z0) + x0;
            double yy = ty * (hit.z() - z0) + y0;
            chi2 += pow((hit.x() - xx) / sig_hit, 2) + pow((hit.y() - yy) / sig_hit, 2);
        }
    }
};

// this runs (a good approximation of) the LHCb VELO pixel tracking on the
// unmarked hits provided in the HitCollection
void makeTracklets(int marked[SENSORS][PHI_BINS][MAX_HITS_PER_BIN], int &ntracks, Tracklet tracks[]) {

    HitArray *hit_array = HitArray::instance();
    Point poca;
    HitIdx hit_buffer[MAX_TRACK_HITS];
    // cache triplets, use only if involve no shared hits
    Tracklet triplets[1000];
    int ntriplets = 0;

    for(int sens0 = SENSORS - 1; sens0 >= 2; --sens0) {
        const int sens1 = sens0 - 2; // -2 -> same side of VELO
        const float z0 = hit_array->zsensor(sens0);
        const float z1 = hit_array->zsensor(sens1);
        const float dz = z0 - z1;
        const float drMax = 0.4 * abs(dz);

        // loop over hits in sensor 0
        for(int phi0 = 0; phi0 < PHI_BINS; phi0++) {
            int i1min = 0; // should really do this for all 3 possible phi bins
            for(int i0 = 0; i0 < hit_array->n(sens0, phi0); i0++) {
                if(marked[sens0][phi0][i0])
                    continue;
                const Hit &hit0 = hit_array->at(sens0, phi0, i0);
                const float rMin = hit0.r() - drMax;
                const float rMax = hit0.r() + drMax;

                // loop over hits in sensor 1
                int bins_to_use[2] = {phi0, hit0.phiBinNN()};
                for(int k = 0; k < 2; k++) {
                    int phi1 = bins_to_use[k];
                    int min = 0;
                    if(phi1 == phi0)
                        min = i1min;
                    for(int i1 = min; i1 < hit_array->n(sens1, phi1); i1++) {
                        const Hit &hit1 = hit_array->at(sens1, phi1, i1);
                        // if(hit1.truePrt() != hit0.truePrt()) continue; // cheat start, only 10% gain now!
                        if(hit1.r() > rMax) {
                            if(phi1 == phi0)
                                i1min = i1 + 1;
                            continue;
                        }
                        if(hit1.r() < rMin)
                            break;
                        if(marked[sens1][phi1][i1])
                            continue;
                        if(deltaPhi(hit0.phi(), hit1.phi()) > 0.08)
                            continue;
                        Trajectory seed(hit0.point(), hit1.point());
                        if(!seed.goodVeloSlopes())
                            continue; // cuts on tx and ty

                        // good seed, now extend ...
                        hit_buffer[0] = HitIdx(sens0, phi0, i0);
                        hit_buffer[1] = HitIdx(sens1, phi1, i1);

                        int step = 2;
                        int next = sens1 - step;
                        unsigned nbMissed = 0;
                        unsigned foundHits = 2;

                        while(next >= 0) {
                            int phi2, i2;
                            hit_array->bestHit(next, getHit(hit_buffer[foundHits - 1]), seed, phi2, i2, marked);
                            if(i2 >= 0) {
                                hit_buffer[foundHits] = HitIdx(next, phi2, i2);
                                foundHits++;
                                nbMissed = 0; // reset missed hit counter
                                seed = Trajectory(getHit(hit_buffer[foundHits - 2]).point(),
                                                  getHit(hit_buffer[foundHits - 1]).point());
                            } else {            // no hits found
                                if(step == 2) { // look on the other side
                                    hit_array->bestHit(
                                        next + 1, getHit(hit_buffer[foundHits - 1]), seed, phi2, i2, marked);
                                    if(i2 >= 0) {
                                        hit_buffer[foundHits] = HitIdx(next + 1, phi2, i2);
                                        foundHits++;
                                        seed = Trajectory(getHit(hit_buffer[foundHits - 2]).point(),
                                                          getHit(hit_buffer[foundHits - 1]).point());
                                    } else
                                        nbMissed += step;
                                    // switch to scanning every module (left and right)
                                    step = 1;
                                } else
                                    ++nbMissed;
                            }
                            if(1 < nbMissed)
                                break; // default is 2, change to 1
                            next -= step;
                        }
                        if(foundHits < 3)
                            continue;        // no tracklet
                        if(foundHits == 3) { // triplet (10% gain by not treating these special)
                            triplets[ntriplets] = Tracklet(3, hit_buffer);
                            if(triplets[ntriplets].chi2NDof() > 20 / 3.)
                                continue;
                            ntriplets++;
                            continue;
                        }
                        unsigned unUsed = 0;
                        for(int h = 0; h < foundHits; h++) {
                            const HitIdx &hidx = hit_buffer[h];
                            if(marked[hidx.sensor][hidx.phi_bin][hidx.idx] == 0)
                                unUsed++;
                        }
                        if(2 * unUsed < foundHits * FRAC_UNUSED)
                            continue;
                        tracks[ntracks] = Tracklet(foundHits, hit_buffer);
                        ntracks++;
                        for(int h = 0; h < foundHits; h++) {
                            const HitIdx &hidx = hit_buffer[h];
                            marked[hidx.sensor][hidx.phi_bin][hidx.idx] = 1;
                        }
                        break;
                    }
                }
            }
        }
    }
    // now check triplets
    for(int i = 0; i < ntriplets; i++) {
        bool any_marked = false;
        for(int j = 0; j < 3; j++) {
            const HitIdx &hidx = triplets[i].hitIdx(j);
            if(marked[hidx.sensor][hidx.phi_bin][hidx.idx]) {
                any_marked = true;
                break;
            }
        }
        if(any_marked)
            continue;
        tracks[ntracks] = triplets[i];
        ntracks++;
    }
}
#endif /* TRACKLET_H */
