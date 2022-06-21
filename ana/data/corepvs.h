#pragma once

#include "ioclass.h"

template <class T> struct CorePVsIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    Vec pvr_x{this, "pvr_x"};
    Vec pvr_y{this, "pvr_y"};
    Vec pvr_z{this, "pvr_z"};
    Vec prt_pvr{this, "prt_pvr"};
    Vec ntrks{this,"ntrks"};

    Vec svr_x{this, "svr_x"};
    Vec svr_y{this, "svr_y"};
    Vec svr_z{this, "svr_z"};
    Vec svr_pvr{this, "svr_pvr"};
};

using CorePVsIn = CorePVsIO<In>;
using CorePVsOut = CorePVsIO<Out>;

template <class T> struct CorePVs2IO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    Vec pvr_x{this, "pvr_x"};
    Vec pvr_y{this, "pvr_y"};
    Vec pvr_z{this, "pvr_z"};
    Vec prt_pvr{this, "prt_pvr"};
    Vec pvr_raw_id{this, "pvr_raw_id"};
    Vec pvr_n_source_tracks{this, "pvr_n_source_tracks"};
    Vec pvr_process_type{this, "pvr_process_type"};

    Vec svr_x{this, "svr_x"};
    Vec svr_y{this, "svr_y"};
    Vec svr_z{this, "svr_z"};
    Vec svr_pvr{this, "svr_pvr"};
};

using CorePVsIn2 = CorePVs2IO<In>;
using CorePVsOut2 = CorePVs2IO<Out>;
