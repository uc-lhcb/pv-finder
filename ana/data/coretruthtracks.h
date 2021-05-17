#pragma once

#include "ioclass.h"

template <class T> struct CoreTruthTracksIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    /// Momentum (direction) of particle
    Vec prt_px{this, "prt_px"};
    Vec prt_py{this, "prt_py"};
    Vec prt_pz{this, "prt_pz"};
    
    /// Location of particle
    Vec prt_x{this, "prt_x"};
    Vec prt_y{this, "prt_y"};
    Vec prt_z{this, "prt_z"};

    /// Number of prompt tracks in event
    Vec ntrks_prompt{this, "ntrks_prompt"};
};

using CoreTruthTracksIn = CoreTruthTracksIO<In>;
using CoreTruthTracksOut = CoreTruthTracksIO<Out>;
