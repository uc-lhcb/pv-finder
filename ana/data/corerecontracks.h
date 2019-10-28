#pragma once

#include "ioclass.h"

template <class T> struct CoreReconTracksIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    Vec recon_x{this, "recon_x"};
    Vec recon_y{this, "recon_y"};
    Vec recon_z{this, "recon_z"};
    Vec recon_tx{this, "recon_tx"};
    Vec recon_ty{this, "recon_ty"};
    Vec recon_chi2{this, "recon_chi2"};
};

using CoreReconTracksIn = CoreReconTracksIO<In>;
using CoreReconTracksOut = CoreReconTracksIO<Out>;
