#pragma once

#include "ioclass.h"

template <class T> struct CoreReconTracksIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    /// Reconstructed track information
    Vec recon_x{this, "recon_x"};
    Vec recon_y{this, "recon_y"};
    Vec recon_z{this, "recon_z"};
    Vec recon_tx{this, "recon_tx"};
    Vec recon_ty{this, "recon_ty"};
    Vec recon_chi2{this, "recon_chi2"};
    Vec recon_pocax{this, "recon_pocax"};
    Vec recon_pocay{this, "recon_pocay"};
    Vec recon_pocaz{this, "recon_pocaz"};
    Vec recon_sigmapocaxy{this, "recon_sigmapocaxy"};
};

using CoreReconTracksIn = CoreReconTracksIO<In>;
using CoreReconTracksOut = CoreReconTracksIO<Out>;
