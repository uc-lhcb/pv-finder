#ifndef TRACKS_H
#define TRACKS_H

#include "hits.h"
#include "triplet.h"

#define MAX_TRACKS 2000

// for sorting tracks
bool trackBeamPOCAz(const Triplet &t0, const Triplet &t1){
  return t0.beamPOCA().z() < t1.beamPOCA().z();
}

// avoiding vectors at first so that everything works well in interactive
// ROOT without needing to compile
class Tracks {

private:

  int _n,_tmin,_tmax,_ngood,_nbad;
  bool _use[MAX_TRACKS];
  Triplet _tracks[MAX_TRACKS];

public:
  
  Tracks() = default;
  Tracks(const Tracks &) = default;
  Tracks& operator=(const Tracks &) = default;

  int tmin() const {return _tmin;}
  int tmax() const {return _tmax;}

  const Triplet& at(int i) const {return _tracks[i];}

  int n() const {return _n;}
  int ngood() const {return _ngood;}
  int nbad() const {return _nbad;}

  // defines range of tracks used at this z
  void setRange(double z){
    int nuse=0;
    _tmin=1e9; _tmax=0;
    double x,y;
    for(int i=0; i<_n; i++){
      _tracks[i].trajectory().getXY(z,x,y);
      if(abs(x)<0.5 && abs(y)< 0.5){
        _use[i]=true;
        if(i < _tmin) _tmin=i;
        if(i > _tmax) _tmax=i;
        nuse++;
      }
      else _use[i]=false;
    }
    if(nuse < 2) _tmin=-1;
  }

  bool run() const {return _tmin >= 0;}

  void newEvent(Hits *hits){

    _ngood=0; _nbad=0;
    _n=0;
    bool marked[MAX_HITS_PHI_BIN];
    Point poca;
    Triplet triplet;

    for(int p=0; p<PHI_BINS; p++){
      if(!hits->useBin(p)) continue;
      int n = hits->n(p);
      for(int i=0; i< n; i++) marked[i]=false;

      for(int i=0; i< n; i++){
        if(marked[i]) continue;
        for(int j=i+1; j< n; j++){
          if(marked[j]) continue;
          if(!hits->goodSeed(p,i,j,poca)) continue;
          if(poca.perp() > 0.5) continue; // really not prompt, could tighten
          for(int k=j+1; k < n; k++){
            if(marked[k]) continue;
            if(hits->sameModule(p,i,k) || hits->sameModule(p,j,k)) continue;
            Triplet triplet(hits->at(p,i),hits->at(p,j),hits->at(p,k));
            if(triplet.chi2NDof() < 10){
              // mark hits that "belong" to this particle
              for(int l=0; l<n; l++){
                if(l==i || l==j || l==k) marked[l]=true;
                else if(triplet.projectedHitChi2(hits->at(p,l))<9) marked[l]=true;
              }
            }
            if(triplet.chi2NDof() > 10) continue;
            if(triplet.deltaPhi(hits->phiCentroid(p))<3.14159/PHI_BINS){
              if(triplet.good()) _ngood++;
              else _nbad++;
              _tracks[_n]=triplet;
              _n++;
              //triplet.print();
            }
            break;
          }
          if(marked[j]) break;
        }
      }
    }
    sort(_tracks,_tracks+_n,trackBeamPOCAz);
  }

};

#endif /* TRACKS_H */
