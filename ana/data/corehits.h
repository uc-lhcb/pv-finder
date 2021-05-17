#pragma once

#include "ioclass.h"

template <class T> struct CoreHitsIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    /// coordinates of hit
    Vec hit_x{this, "hit_x"};
    Vec hit_y{this, "hit_y"};
    Vec hit_z{this, "hit_z"};
    /// hit ID
    Vec hit_prt{this, "hit_prt"};
};

using CoreHitsIn = CoreHitsIO<In>;
