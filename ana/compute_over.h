#pragma once

#include "fcn.h"
#include "tracks.h"
#include "hits.h"
#include "data/raw_hits.h"

inline double bin_center(int nbins, double min, double max, int i) {
    return (i + 0.5) / nbins * (max - min) + min;
}

// Take a function of n, kernel, x, y, and call on each n value from 0 to 4000
inline void compute_over(const DataHits &data, std::function<void(int, float, float, float)> dothis) {
    
    constexpr int nb = 4000;
    constexpr double zmin = -100.;
    constexpr double zmax = 300.;
    
    // gets all hits, bins them in phi
    Hits hits;
    hits.newEvent(data);
    
    // make triplets
    Tracks tracks;
    
    // C style workaround for global FCN tracks inside the fitter
    fcn_global_tracks = &tracks;
    
    tracks.newEvent(&hits);
    cout << " Total tracks: " << tracks.n() << " good tracks: " << tracks.ngood() << " bad tracks: " << tracks.nbad();
    
    Point pv;
    
    // build the kernel vs z profiled in x-y
    // TODO: clearly non-optimal CPU-wise how this search is done
    for(int b=0; b<nb; b++){
        double z = bin_center(nb, zmin, zmax, b);
        double kmax = -1.;
        double xmax = 0.;
        double ymax = 0.;
        
        // 1st do coarse grid search
        tracks.setRange(z);
        if(!tracks.run())
            continue;
        
        for(double x=-0.4; x<=0.41; x+=0.1){
            for(double y=-0.4; y<=0.41; y+=0.1){
                pv.set(x,y,z);
                double val = kernel(pv);
                if(val > kmax){
                    kmax=val;
                    xmax=x;
                    ymax=y;
                }
            }
        }
        
        // now do gradient descent from max found
        pv.set(xmax,ymax,z);
        float kernel = kernelMax(pv);
        dothis(b, kernel, pv.x(), pv.y());
    }
}
