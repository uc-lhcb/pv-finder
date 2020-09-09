#pragma once

#include "data/corehits.h"
#include "ellipsoid.h"
//#include "fcn.h"
#include "hits.h"
#include "tracks.h"
#include <functional>

inline double bin_center(int const &nbins, double const &min, double const &max, int const &i) {
    return (i + 0.5) / nbins * (max - min) + min;
}

inline AnyTracks make_tracks(const CoreHitsIn &data) {
    // gets all hits, bins them in phi
    Hits hits;
    hits.newEvent(data);

    // make triplets
    Tracks tracks;

    tracks.newEvent(&hits);
    std::cout << " (" << tracks.ngood() << " good, " << tracks.nbad() << " bad)";

    return AnyTracks(tracks);
}

// Take a function of n, kernel, x, y, and call on each n value from 0 to 4000
void compute_over(AnyTracks &any_tracks, std::function<void(int, float, float, float)> dothis) {

    constexpr int nbz = 4000, nbxy = 20; // number of bins in z and x,y for the coarse grid search
    constexpr int ninterxy = 3; // number of interpolating bins in x and y for the fine grid search. (i.e. ninterxy bins
                                // between this and the next bin in x or y)
    constexpr double zmin = -100., zmax = 300., xymin = -0.4, xymax = 0.4; // overall range in z, x and y in mm
    constexpr double interxy = 0.01;                                       // interpolation stepsize in mm

    // make poca error ellipsoids for each track w.r.t. the beamline
    std::vector<Ellipsoid> poca_ellipsoids;
    Trajectory beamline(0., 0., 0., 0., 0.);
    for(auto const &trajectory : any_tracks.trajectories())
        poca_ellipsoids.emplace_back(beamline, trajectory);
    // sort ellipsoids by zmin so that we can make the iteration a bit faster later
    std::sort(poca_ellipsoids.begin(), poca_ellipsoids.end(), [](Ellipsoid const &a, Ellipsoid const &b) { return a.zmin() < b.zmin(); });

    for(int bz = 0; bz < nbz; bz++) {
        double kernel = 0.;
        Point best;
        double z = bin_center(nbz, zmin, zmax, bz);
        // remove ellipsoid from containter if it won't contribute to this bin and bins further downstream
        poca_ellipsoids.erase(std::remove_if(poca_ellipsoids.begin(), poca_ellipsoids.end(), [&z](Ellipsoid const &el) { return el.zmax() < z; }), poca_ellipsoids.end());
        // check if there are any ellipsoids in this bin (we have sorted, so it's faster to search for the first
        // ellipsoid that doesn't contribute)
        auto const first_ell_out_of_range = std::find_if_not( poca_ellipsoids.begin(), poca_ellipsoids.end(), [&z](Ellipsoid const &el) { return el.zmin() < z; });
        // go on if there are no ellipsoids in this bin
        if(first_ell_out_of_range == poca_ellipsoids.begin())
            continue;
        // function to evaluate the pdf, set kernel maximum and it's position
        auto eval_pdf_set_kernel_maximum_and_position = [&kernel, &best, &poca_ellipsoids, &first_ell_out_of_range](Point const &p) -> void {
            double this_kernel = 0.;
            // iterate ellipsoids in range
            for(auto peit = poca_ellipsoids.begin(); peit < first_ell_out_of_range; peit++)
                this_kernel += (*peit).pdf(p);
            if(this_kernel > kernel) {
                kernel = this_kernel;
                best = p;
            }
        };
        // do coarse grid search in nbxy (20) x and y bins, each 1000*(xymax-xymin)/nbxy (40) micron wide
        for(int bx = 0; bx < nbxy; bx++) {
            for(int by = 0; by < nbxy; by++) {
                Point p(bin_center(nbxy, xymin, xymax, bx), bin_center(nbxy, xymin, xymax, by), z);
                eval_pdf_set_kernel_maximum_and_position(p);
            }
        }
        auto const best_first_scan = best;
        // do finer grid search taking ninterxy (3) steps left/up/forward, and ninterxy (3) steps right/down/backward
        // with size interxy (10 microns) from the current best point
        for(auto fbinx = -ninterxy; fbinx <= ninterxy; fbinx++) {
            if(fbinx == 0)
                continue; // we have this point already
            for(auto fbiny = -ninterxy; fbiny <= ninterxy; fbiny++) {
                if(fbiny == 0)
                    continue; // we have this point already
                Point p(best_first_scan.x() + fbinx * ninterxy, best_first_scan.y() + fbiny * ninterxy, z);
                eval_pdf_set_kernel_maximum_and_position(p);
            }
        }
        dothis(bz, kernel, best.x(), best.y());
    }
}
