#pragma once

#include "ioclass.h"

template <class T> struct CoreNHitsIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    /// Number of hits associated with this event, used for size of hits array.
    Vec prt_hits{this, "prt_hits"};
};

using CoreNHitsIn = CoreNHitsIO<In>;
using CoreNHitsOut = CoreNHitsIO<Out>;
