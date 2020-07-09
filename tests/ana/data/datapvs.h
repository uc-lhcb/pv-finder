
#pragma once

#include <TTree.h>

#include "corenhits.h"
#include "corepvs.h"
#include "coretruthtracks.h"

template <class T> struct DataPVsIO : public CoreIO<T, float> {
    using CoreIO<T, float>::CoreIO;
    using Vec = VecIO<T, float>;

    Vec pv_cat{this, "pv_cat"};
    Vec pv_loc{this, "pv_loc"};
    Vec pv_loc_x{this, "pv_loc_x"};
    Vec pv_loc_y{this, "pv_loc_y"};
    Vec pv_ntrks{this, "pv_ntrks"};

    Vec sv_cat{this, "sv_cat"};
    Vec sv_loc{this, "sv_loc"};
    Vec sv_loc_x{this, "sv_loc_x"};
    Vec sv_loc_y{this, "sv_loc_y"};
    Vec sv_ntrks{this, "sv_ntrks"};

    // sv_n and pv_n are no longer present
};

using DataPVsOut = DataPVsIO<Out>;

inline void copy_in_pvs(DataPVsOut &self,
                        const CoreTruthTracksIn &data_trks,
                        const CorePVsIn &data_pvs,
                        const CoreNHitsIn &data_hits) {

    self.clear();

    for(int i = 0; i < data_pvs.pvr_z->size(); ++i) {
        self.pv_cat->push_back(pvCategory(data_trks, data_pvs, data_hits, i));
        self.pv_loc->push_back(data_pvs.pvr_z->at(i));
        self.pv_loc_x->push_back(data_pvs.pvr_x->at(i));
        self.pv_loc_y->push_back(data_pvs.pvr_y->at(i));
        self.pv_ntrks->push_back(ntrkInAcc(data_trks, data_pvs, data_hits, i));
    }

    for(int i = 0; i < data_pvs.svr_z->size(); ++i) {
        self.sv_cat->push_back(svCategory(data_trks, data_pvs, data_hits, i));
        self.sv_loc->push_back(data_pvs.svr_z->at(i));
        self.sv_loc_x->push_back(data_pvs.svr_x->at(i));
        self.sv_loc_y->push_back(data_pvs.svr_y->at(i));
        self.sv_ntrks->push_back(nSVPrt(data_trks, data_pvs, data_hits, i));
    }
}

inline std::ostream &operator<<(std::ostream &stream, const DataPVsOut &self) {
    return stream << TString::Format("PVs: %lu SVs: %lu", self.pv_loc->size(), self.sv_loc->size());
}
