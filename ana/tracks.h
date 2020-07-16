#pragma once

#include "data/corerecontracks.h"
#include "hits.h"
#include "triplet.h"
#include <limits>

// for sorting tracks
inline bool trackBeamPOCAz(const TripletBase &t0, const TripletBase &t1) {
    return t0.beamPOCA().z() < t1.beamPOCA().z();
}

class Tracks {

  private:
    int _ngood, _nbad;
    std::vector<TripletToy> _tracks;

  public:
    Tracks() = default;
    Tracks(const Tracks &) = default;
    Tracks &operator=(const Tracks &) = default;

    const TripletBase &at(int i) const { return _tracks.at(i); }

    int n() const { return _tracks.size(); }
    int ngood() const { return _ngood; }
    int nbad() const { return _nbad; }

    void newEvent(Hits *hits) {

        _ngood = 0;
        _nbad = 0;
        _tracks.clear();

        bool marked[MAX_HITS_PHI_BIN];
        Point poca;
        TripletToy triplet;

        for(int p = 0; p < PHI_BINS; p++) {
            if(!hits->useBin(p))
                continue;
            int n = hits->n(p);
            for(int i = 0; i < n; i++)
                marked[i] = false;

            for(int i = 0; i < n; i++) {
                if(marked[i])
                    continue;
                for(int j = i + 1; j < n; j++) {
                    if(marked[j])
                        continue;
                    if(!hits->goodSeed(p, i, j, poca))
                        continue;
                    if(poca.perp() > 0.5)
                        continue; // really not prompt, could tighten
                    for(int k = j + 1; k < n; k++) {
                        if(marked[k])
                            continue;
                        if(hits->sameModule(p, i, k) || hits->sameModule(p, j, k))
                            continue;
                        TripletToy triplet(hits->at(p, i), hits->at(p, j), hits->at(p, k));
                        if(triplet.chi2NDof() < 10) {
                            // mark hits that "belong" to this particle
                            for(int l = 0; l < n; l++) {
                                if(l == i || l == j || l == k)
                                    marked[l] = true;
                                else if(triplet.projectedHitChi2(hits->at(p, l)) < 9)
                                    marked[l] = true;
                            }
                        }
                        if(triplet.chi2NDof() > 10)
                            continue;
                        if(triplet.deltaPhi(hits->phiCentroid(p)) < 3.14159 / PHI_BINS) {
                            if(triplet.good())
                                _ngood++;
                            else
                                _nbad++;
                            _tracks.push_back(triplet);
                            // triplet.print();
                        }
                        break;
                    }
                    if(marked[j])
                        break;
                }
            }
        }
        std::sort(_tracks.begin(), _tracks.end(), trackBeamPOCAz);
    }
};

class AnyTracks {
    std::vector<TripletBase> _tracks;
    int _tmin = std::numeric_limits<int>::max();
    int _tmax = 0;

  public:
    AnyTracks(const Tracks &tracks) {
        for(int i = 0; i < tracks.n(); i++)
            _tracks.push_back(tracks.at(i));
    }

    AnyTracks(const CoreReconTracksIn &trks) {
        for(int i = 0; i < trks.recon_x->size(); i++) {
            _tracks.emplace_back(trks.recon_x->at(i),
                                 trks.recon_y->at(i),
                                 trks.recon_z->at(i),
                                 trks.recon_tx->at(i),
                                 trks.recon_ty->at(i),
                                 trks.recon_chi2->at(i));
        }
    }

    // defines range of tracks used at this z
    void setRange(double z) {
        int nuse = 0;
        _tmin = 1e9;
        _tmax = 0;
        double x, y;
        for(int i = 0; i < n(); i++) {
            _tracks[i].trajectory().getXY(z, x, y);
            if(abs(x) < 0.5 && abs(y) < 0.5) {
                if(i < _tmin)
                    _tmin = i;
                if(i > _tmax)
                    _tmax = i;
                nuse++;
            } else {
            }
        }
        if(nuse < 2)
            _tmin = -1;
    }

    int tmin() const { return _tmin; }
    int tmax() const { return _tmax; }
    bool run() const { return _tmin >= 0; }

    const TripletBase &at(int i) const { return _tracks[i]; }
    int n() const { return _tracks.size(); }
};

inline std::ostream &operator<<(std::ostream &input, const AnyTracks &self) {
    return input << "AnyTracks: " << self.n();
}

inline void copy_in(CoreReconTracksOut &self, const AnyTracks &tracks) {

    self.clear();
    for(int i = 0; i < tracks.n(); i++) {
        const auto ttraj = tracks.at(i).trajectory();
        const auto trajp = ttraj.point();
        const auto bpoca = ttraj.beamPOCA();
        const auto tchi2 = tracks.at(i).get_chi2();

        self.recon_x->push_back(trajp.x());
        self.recon_y->push_back(trajp.y());
        self.recon_z->push_back(trajp.z());
        self.recon_tx->push_back(ttraj.xslope());
        self.recon_ty->push_back(ttraj.yslope());
        self.recon_chi2->push_back(tchi2);
        self.recon_pocax->push_back(bpoca.x());
        self.recon_pocay->push_back(bpoca.y());
        self.recon_pocaz->push_back(bpoca.z());
        self.recon_sigmapocaxy->push_back(tchi2/3.<=2. ? 0.05 : 0.05+(tchi2-2.)*0.05/4.);
    }
}
