#pragma once

#include "data/corehits.h"
#include "hit.h"
#include "trajectory.h"

#include <iostream>

#define PHI_BINS 314
#define MAX_HITS_PHI_BIN 150 // typically about max of ~50 in a 10-pv event

// for sorting hits
inline bool innerMostHit(const Hit &h0, const Hit &h1) { return h0.point().perp() < h1.point().perp(); }

class Hits {

  private:
    int _n_phi[PHI_BINS], _n;
    Hit _hits_phi[PHI_BINS][MAX_HITS_PHI_BIN];
    TVector3 _phi_centroids[PHI_BINS];

    // set up the overlapping phi bins
    void _init() {
        for(int i = 0; i < PHI_BINS; i++) {
            _phi_centroids[i].SetPtThetaPhi(1, 90 * 3.14159 / 180., i * 2 * 3.14159 / PHI_BINS);
            _n_phi[i] = 0;
        }
    }

  public:
    Hits() { this->_init(); }
    Hits(const Hits &) = default;
    Hits &operator=(const Hits &) = default;

    int n(int i = -1) const {
        if(i < 0)
            return _n;
        else
            return _n_phi[i];
    }

    const Hit &at(int p, int i) const { return _hits_phi[p][i]; }

    const TVector3 &phiCentroid(int i) const { return _phi_centroids[i]; }

    bool goodSeed(int p, int i, int j, Point &poca) const {
        if(sameModule(p, i, j))
            return false;
        Trajectory traj_ij(_hits_phi[p][i].point(), _hits_phi[p][j].point());
        if(!traj_ij.goodVeloSlopes())
            return false; // cut imposed by actual VELO tracking
        poca = traj_ij.beamPOCA();
        return true;
    }

    bool sameModule(int p, int i, int j) const { return Hit::sameModule(_hits_phi[p][i], _hits_phi[p][j]); }

    void newEvent(const CoreHitsIn &data) {
        _n = data.hit_z->size();
        for(int i = 0; i < PHI_BINS; i++)
            _n_phi[i] = 0;

        for(int h = 0; h < _n; h++) {
            TVector3 h3(data.hit_x->at(h), data.hit_y->at(h), data.hit_z->at(h));

            for(int i = 0; i < PHI_BINS; i++) { // obviously this can be done *much* quicker
                if(abs(h3.DeltaPhi(_phi_centroids[i])) < 2 * 3.14159 / PHI_BINS) {
                    if(_n_phi[i] < MAX_HITS_PHI_BIN)
                        _hits_phi[i][_n_phi[i]] = Hit(h, h3.X(), h3.Y(), h3.Z(), data.hit_prt->at(h));
                    else
                        std::cout << "> " << MAX_HITS_PHI_BIN << " hits in phi bin!" << std::endl;
                    _n_phi[i]++;
                }
            }
        }
        for(int i = 0; i < PHI_BINS; i++) {
            std::sort(_hits_phi[i], _hits_phi[i] + _n_phi[i], innerMostHit);
        }
    }

    bool useBin(int i) const {
        if(_n_phi[i] < 3 || _n_phi[i] >= MAX_HITS_PHI_BIN)
            return false;
        else
            return true;
    }
};
