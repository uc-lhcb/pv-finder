#pragma once

#include "data/corerecontracks.h"
#include "hits.h"
#include "triplet.h"
#include <limits>

#include <iostream>
#include <vector>
#include <numeric> 
#include <algorithm> 

using namespace std;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    vector<size_t> idx(v.size());
    
    
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    
  return idx;
}

// for sorting by poca_z
// vector<pair<double, int> > sortArr(vector<double> v) {
//     vector<pair<double, int> > vp; // Vector to store element with respective present index

//     // Inserting element in pair vector to keep track of previous indexes
//     for (int i = 0; i < v.size(); ++i) {
//         vp.push_back(make_pair(v.at(i), i));
//     }
  
//     // Sorting pair vector
//     sort(vp.begin(), vp.end());
  
//     return vp
// }

// below two methods for determining starting index for tracks
inline int getClosest(vector<double> v, int ind1, int ind2, double target) {
    if (target - v.at(ind1) >= v.at(ind2) - target) return ind2;
    return ind1;
}

inline int findClosest(vector<double> v, int n, double target, int start = 0) {
    if (target <= v.at(0)) return 0;
    if (target >= v.at(n - 1)) return n-1;
    
    int i = start, j = n, mid = 0;
    while (i < j) {
        mid = (i + j) / 2;
        if (v.at(mid) == target) return mid;

        // If target is less than array element, then search in left 
        if (target < v.at(mid)) {
            // If target is greater than previous to mid, return closest of two
            if (mid > 0 && target > v.at(mid-1)) {
                return getClosest(v, mid-1, mid, target);
            }
            
            // repeat for left half
            j = mid;
        }

        // If target is greater than mid
        else {
            if (mid < n - 1 && target < v.at(mid+1)) return getClosest(v, mid, mid+1, target);
            i = mid + 1; // update i
        }
    }
    return mid; // Only single element left after search
}

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
    vector<double> _pocaz;

  public:
    // already sorted
    AnyTracks(const Tracks &tracks) {
        for(int i = 0; i < tracks.n(); i++)
            _tracks.push_back(tracks.at(i));
    }

    // sorts when initialized
    AnyTracks(const CoreReconTracksIn &trks) {
        for(int i = 0; i < trks.recon_x->size(); i++) {
            Trajectory current_traj = Trajectory(trks.recon_x->at(i), trks.recon_y->at(i), trks.recon_z->at(i), 
                       trks.recon_tx->at(i), trks.recon_ty->at(i));
            _pocaz.push_back(current_traj.beamPOCA().z());
        }
        
        vector<size_t> ind_sorted = sort_indexes(_pocaz); // get sorted indices of _pocaz
        sort(_pocaz.begin(),_pocaz.end()); // actually sort _pocaz
        
        // append tracks in sorted order
        for(int i = 0; i < ind_sorted.size(); i++) {
            _tracks.emplace_back(trks.recon_x->at(ind_sorted.at(i)),
                                 trks.recon_y->at(ind_sorted.at(i)),
                                 trks.recon_z->at(ind_sorted.at(i)),
                                 trks.recon_tx->at(ind_sorted.at(i)),
                                 trks.recon_ty->at(ind_sorted.at(i)),
                                 trks.recon_chi2->at(ind_sorted.at(i)),
                                 trks.recon_sigmapocaxy->at(ind_sorted.at(i)),
                                 trks.recon_errz0->at(ind_sorted.at(i)));
        }
    }
    

    // defines range of tracks used at this z
    int setRange(double z, int prev = 0) {
        double binwidth = 50./12000.;
        
        //find track starting index (use binary search)
        int start = findClosest(_pocaz,_pocaz.size(),z-50.*binwidth,prev); // does not look at indices under prev
        
        //int nuse = 0;
        _tmin = 1e9;
        _tmax = 0;
        
        double x, y;
        for(int i = start; i < _pocaz.size(); i++) {
            _tracks[i].trajectory().getXY(z, x, y);
            if(abs(x) < 2 && abs(y) < 2 && abs(z-_pocaz.at(i))<1000.*binwidth) {
                if(i < _tmin)  _tmin = i;
                if(i > _tmax) _tmax = i;
                //nuse++;
            }
            if(_pocaz.at(i)-z>=1000.*binwidth){ 
                break;
            }
        }
//         if(nuse < 2)
//             _tmin = -1;        
        return start;
    }

    int tmin() const { return _tmin; }
    int tmax() const { return _tmax; }
    bool run() const { return _tmin >= 0; }
    vector<double> pocaz() const { return _pocaz; }

    const TripletBase &at(int i) const { return _tracks[i]; }
    int n() const { return _tracks.size(); }

    std::vector<Trajectory> trajectories ( ) const {
      std::vector<Trajectory> trjs;
      std::transform(_tracks.begin(), _tracks.end(), std::back_inserter(trjs),[](auto const& triplet){ return triplet.trajectory(); } );
      return trjs;
    }
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
        const auto tsigmapocaxy = tracks.at(i).get_sigmapocaxy(); // EMK
        const auto terrz0 = tracks.at(i).get_errz0(); // EMK

        self.recon_x->push_back(trajp.x());
        self.recon_y->push_back(trajp.y());
        self.recon_z->push_back(trajp.z());
        self.recon_tx->push_back(ttraj.xslope());
        self.recon_ty->push_back(ttraj.yslope());
        self.recon_chi2->push_back(tchi2);
        self.recon_sigmapocaxy->push_back(tsigmapocaxy);
        self.recon_errz0->push_back(terrz0);
        
        
        //self.recon_pocax->push_back(bpoca.x());
        //self.recon_pocay->push_back(bpoca.y());
        //self.recon_pocaz->push_back(bpoca.z());
        //self.recon_sigmapocaxy->push_back(tchi2/3.<=2. ? 0.05 : 0.05+(tchi2-2.)*0.05/4.);
    }
}
