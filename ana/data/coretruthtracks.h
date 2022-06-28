#pragma once

#include "ioclass.h"

template <class T> struct CoreTruthTracksIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    Vec prt_px{this, "prt_px"};
    Vec prt_py{this, "prt_py"};
    Vec prt_pz{this, "prt_pz"};
    Vec prt_x{this, "prt_x"};
    Vec prt_y{this, "prt_y"};
    Vec prt_z{this, "prt_z"};

    Vec ntrks_prompt{this, "ntrks_prompt"};
};

using CoreTruthTracksIn = CoreTruthTracksIO<In>;
using CoreTruthTracksOut = CoreTruthTracksIO<Out>;

template <class T> struct CoreTruthTracks2IO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    Vec prt_px{this, "prt_px"};
    Vec prt_py{this, "prt_py"};
    Vec prt_pz{this, "prt_pz"};
    Vec prt_x{this, "prt_x"};
    Vec prt_y{this, "prt_y"};
    Vec prt_z{this, "prt_z"};
    
    Vec prt_pvr{this, "prt_pvr"};
    Vec prt_e{this, "prt_e"};
    //Vec prt_pid{this, "prt_pid"};
    
    Vec ntrks_prompt{this, "ntrks_prompt"};
};

using CoreTruthTracksIn2 = CoreTruthTracks2IO<In>;
using CoreTruthTracksOut2 = CoreTruthTracks2IO<Out>;
