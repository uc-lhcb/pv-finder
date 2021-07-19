#pragma once

#include "data/corehits.h"
#include "ellipsoid.h"
#include "fcn.h"
#include "hits.h"
#include "tracks.h"
#include <functional>

inline double bin_center(int const &nbins, double const &min, double const &max, int const &i) {
    return (i + 0.5) / nbins * (max - min) + min;
}

// Take a function of n, kernel_value, x, y, and call on each n value from 0 to 4000
void compute_over(AnyTracks &any_tracks, std::function<void(int, std::vector<double>, std::vector<double>, std::vector<double>)> dothis) {

    constexpr int nbz = 4000, nbxy = 30; // number of bins in z and x,y for the coarse grid search
    constexpr int ninterxy = 3; // number of interpolating bins in x and y for the fine grid search. (i.e. ninterxy bins
                                // between this and the next bin in x or y)
    constexpr double zmin = -10., zmax = 10., xymin = -0.2, xymax = 0.2; // overall range in z, x and y in mm
    constexpr double interxy = 0.01;                                       // interpolation stepsize in mm

    // C style workaround for global FCN tracks inside the fitter
    fcn_global_tracks = &any_tracks;
    Point pv;

    // make poca error ellipsoids for each track w.r.t. the beamline
    std::vector<Ellipsoid> poca_ellipsoids;
    Trajectory beamline(0., 0., 0., 0., 0.);
    int trkcount = 0;
    for(auto const &trajectory : any_tracks.trajectories()){
        const auto tsigmapocaxy = any_tracks.at(trkcount).get_sigmapocaxy(); // EMK
        const auto terrz0 = any_tracks.at(trkcount).get_errz0(); // EMK
        poca_ellipsoids.emplace_back(beamline, trajectory, tsigmapocaxy, terrz0); //EMK
        trkcount ++;
    }
    
    // sort ellipsoids by zmin so that we can make the iteration a bit faster later
    std::sort(poca_ellipsoids.begin(), poca_ellipsoids.end(), [](Ellipsoid const &a, Ellipsoid const &b) { return a.zmin() < b.zmin(); });

    for(int bz = 0; bz < nbz; bz++) {
        std::vector<double> kernel_value = {-1.,-1.,0.}, bestx = {0.,0.,0.}, besty = {0.,0.,0.};
        Point best(-999.,-999.,-999.);
        double const z = bin_center(nbz, zmin, zmax, bz);
        // remove ellipsoid from containter if it won't contribute to this bin and bins further downstream
        poca_ellipsoids.erase(std::remove_if(poca_ellipsoids.begin(), poca_ellipsoids.end(), [&z](Ellipsoid const &el) { return el.zmax() < z; }), poca_ellipsoids.end());
        // check if there are any ellipsoids in this bin (we have sorted, so it's faster to search for the first
        // ellipsoid that doesn't contribute)
        auto const first_ell_out_of_range = std::find_if_not( poca_ellipsoids.begin(), poca_ellipsoids.end(), [&z](Ellipsoid const &el) { return el.zmin() < z; });
        // go on if there are no ellipsoids in this bin
        if(first_ell_out_of_range == poca_ellipsoids.begin()) continue;
        // function to evaluate the pdf, set kernel_value maximum and it's position
        auto eval_pdf_set_kernel_maximum_and_position = [&kernel_value, &best, &poca_ellipsoids, &first_ell_out_of_range](Point const &p) -> void {
            double this_kernel = 0.,this_kernel_sq = 0.;
            // iterate ellipsoids in range
            for(auto peit = poca_ellipsoids.begin(); peit < first_ell_out_of_range; peit++){
              auto POCA_pdf_for_this_ellipsoid = (*peit).pdf(p);
              this_kernel += POCA_pdf_for_this_ellipsoid;
              this_kernel_sq += POCA_pdf_for_this_ellipsoid*POCA_pdf_for_this_ellipsoid;
            }
            if(this_kernel > kernel_value[0]) {
              kernel_value[0] = this_kernel;
              kernel_value[1] = this_kernel_sq;
              best = p;
            }
        };
        // do coarse grid search in nbxy (20) x and y bins, each 1000*(xymax-xymin)/nbxy (40) micron wide
        for(int bx = 0; bx < nbxy; bx++) {
            for(int by = 0; by < nbxy; by++) {
                Point p(bin_center(nbxy, xymin, xymax, bx), bin_center(nbxy, xymin, xymax, by), z);
                //call the lambda to set kernel_value values and point if p is the best
                eval_pdf_set_kernel_maximum_and_position(p);
            }
        }
        auto const best_first_scan = best;
        // do finer grid search taking ninterxy (3) steps left/up/forward, and ninterxy (3) steps right/down/backward
        // with size interxy (10 microns) from the current best point
        for(auto fbinx = -ninterxy; fbinx <= ninterxy; fbinx++) {
          if(fbinx == 0) continue;// we have this point already
          for(auto fbiny = -ninterxy; fbiny <= ninterxy; fbiny++) {
            if(fbiny == 0) continue; // we have this point already
            Point p(best_first_scan.x() + fbinx * interxy, best_first_scan.y() + fbiny * interxy, z);
            //call the lambda to set kernel_value values and point if p is the best
            eval_pdf_set_kernel_maximum_and_position(p);
          }
        }
        //set x and y of first kernel_value definition
        bestx[0]=best.x();
        besty[0]=best.y();
        bestx[1]=best.x();
        besty[1]=best.y();

        //second kernel_value definition (the original)
        double kmax = -1., xmax = 0., ymax = 0.;
        // 1st do coarse grid search
        any_tracks.setRange(z);
        if(!any_tracks.run()) continue;

        for(double x = -0.4; x <= 0.41; x += 0.1) {
            for(double y = -0.4; y <= 0.41; y += 0.1) {
                pv.set(x, y, z);
                double val = kernel(pv);
                if(val > kmax) {
                    kmax = val;
                    xmax = x;
                    ymax = y;
                }
            }
        }

        // now do gradient descent from max found
        pv.set(xmax, ymax, z);
        kernel_value[2] = kernelMax(pv);
        //set x and y of first kernel_value definition
        bestx[2]=pv.x();
        besty[2]=pv.y();

        dothis(bz, kernel_value, bestx, besty);
    }
}
