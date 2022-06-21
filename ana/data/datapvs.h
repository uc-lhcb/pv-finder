
#pragma once

#include <TTree.h>

#include "corenhits.h"
#include "corepvs.h"
#include "coretruthtracks.h"

#include <vector>
#include <numeric>

template <class T> struct DataPVs2IO : public CoreIO<T, float> {
    using CoreIO<T, float>::CoreIO;
    using Vec = VecIO<T, float>;

    Vec pvr_z{this, "pvr_z"};
    Vec pvr_x{this, "pvr_x"};
    Vec pvr_y{this, "pvr_y"};
    Vec ntrks_prompt{this, "ntrks_prompt"};
    Vec ntrks{this, "ntrks"};

    Vec sv_cat{this, "sv_cat"};
    Vec svr_z{this, "svr_z"};
    Vec svr_x{this, "svr_x"};
    Vec svr_y{this, "svr_y"};
    Vec svr_pvr{this, "svr_pvr"};
    Vec sv_ntrks{this, "sv_ntrks"};
    
    Vec prt_x{this, "prt_x"};
    Vec prt_y{this, "prt_y"};
    Vec prt_z{this, "prt_z"};
    
    Vec prt_px{this, "prt_px"};
    Vec prt_py{this, "prt_py"};
    Vec prt_pz{this, "prt_pz"};
    
    Vec prt_pvr{this, "prt_pvr"};
    Vec prt_e{this, "prt_e"};
    
    Vec prt_hits{this, "prt_hits"};
};

using DataPVsOut2 = DataPVs2IO<Out>;

inline void copy_in_pvs2(DataPVsOut2 &self,
                         const CoreTruthTracksIn2 &data_trks,
                         const CorePVsIn2 &data_pvs,
                         const CoreNHitsIn &data_hits) {

    self.clear();
    
    std::vector< double > pvr_x_new;
    std::vector< double > pvr_y_new;
    std::vector< double > pvr_z_new;
    std::vector< int > ntrks_prompt_new;
    std::vector< int > ntrks_new;
        
    std::vector< double > pvr_x_current;
    std::vector< double > pvr_y_current;
    std::vector< double > pvr_z_current;
    
    std::vector< int > current_numtracks;
    std::vector< int > current_ntrks_prompt;
    
    std::vector<int> good_pointers;
    std::vector<int> reassigned_pointers;
    
    int trklen = 0;
    int sumofweights = 0;
    double weightedavg_x = 0.0;
    double weightedavg_y = 0.0;
    double weightedavg_z = 0.0;
    
    int counter = 0;
    for(int i = 0; i < data_pvs.pvr_raw_id->back(); ++i) {
        
        pvr_x_current.clear();
        pvr_y_current.clear();
        pvr_z_current.clear();
        current_numtracks.clear();
        current_ntrks_prompt.clear();
        
        double first_pvr_z = data_pvs.pvr_z->at(counter);
        int current_ind = i;
        while (current_ind == i) {
            
            trklen = 0;
            
            good_pointers.push_back(counter);
            reassigned_pointers.push_back(current_ind);
            
            if (data_pvs.pvr_n_source_tracks->at(counter) == 0 && 
                data_pvs.pvr_process_type->at(counter) == 0 &&
                abs(first_pvr_z - data_pvs.pvr_z->at(counter)) <= 0.01) {
                pvr_x_current.push_back(data_pvs.pvr_x->at(counter));
                pvr_y_current.push_back(data_pvs.pvr_y->at(counter));
                pvr_z_current.push_back(data_pvs.pvr_z->at(counter));
                current_ntrks_prompt.push_back(data_trks.ntrks_prompt->at(counter));
                for (int k = 0; k<data_trks.prt_pvr->size(); ++k) {
                    if (data_trks.prt_pvr->at(k)==counter){
                        ++trklen;
                    }
                }
                current_numtracks.push_back(trklen);
                
//                 good_pointers.push_back(counter);
//                 reassigned_pointers.push_back(current_ind);
            }
            
            counter += 1;
            
            if (counter >= data_pvs.pvr_z->size()){
                break;
            }
            
            current_ind = data_pvs.pvr_raw_id->at(counter);
        }
        
        if (!pvr_z_current.empty()) {
            sumofweights = std::accumulate(current_numtracks.begin(), 
                                           current_numtracks.end(), 
                                           decltype(current_numtracks)::value_type(0));
            
            weightedavg_x = 0;
            weightedavg_y = 0;
            weightedavg_z = 0;
            
            for (int k = 0; k < pvr_x_current.size(); ++k) {
                weightedavg_x+=current_numtracks.at(k)*pvr_x_current.at(k)/sumofweights;
                weightedavg_y+=current_numtracks.at(k)*pvr_y_current.at(k)/sumofweights;
                weightedavg_z+=current_numtracks.at(k)*pvr_z_current.at(k)/sumofweights;
            }
            
            ntrks_new.push_back(std::accumulate(current_numtracks.begin(),
                                                current_numtracks.end(),
                                                decltype(current_numtracks)::value_type(0)));
            
            pvr_x_new.push_back(weightedavg_x);
            pvr_y_new.push_back(weightedavg_y);
            pvr_z_new.push_back(weightedavg_z);
            
            ntrks_prompt_new.push_back(std::accumulate(current_ntrks_prompt.begin(),
                                                       current_ntrks_prompt.end(),
                                                       decltype(current_ntrks_prompt)::value_type(0)));
        }
    }
    
     // iterate through truth tracks and construct new prt_pvr
    for(int i = 0; i < data_trks.prt_pvr->size(); ++i) {
        // check if current value is in "good pointer"
        auto it = find(good_pointers.begin(), good_pointers.end(), data_trks.prt_pvr->at(i));
        if (it!=good_pointers.end()) {
            int index = it - good_pointers.begin();
            self.prt_pvr->push_back(reassigned_pointers.at(index));
        }
        else {
            self.prt_pvr->push_back(-1);
        }
    }
    
    for(int i = 0; i < pvr_z_new.size(); ++i) {
        self.pvr_z->push_back(pvr_z_new.at(i));
        self.pvr_x->push_back(pvr_x_new.at(i));
        self.pvr_y->push_back(pvr_y_new.at(i));
        self.ntrks->push_back(ntrks_new.at(i));
    }
    
    for(int i = 0; i < data_hits.prt_hits->size(); ++i) {
        self.prt_hits->push_back(data_hits.prt_hits->at(i));
    }
    
    for(int i = 0; i < ntrks_prompt_new.size(); ++i) {
        self.ntrks_prompt->push_back(ntrks_prompt_new.at(i));
    }

    for(int i = 0; i < data_pvs.svr_z->size(); ++i) {
//         self.sv_cat->push_back(svCategory(data_trks, data_pvs, data_hits, i));
        self.svr_z->push_back(data_pvs.svr_z->at(i));
        self.svr_x->push_back(data_pvs.svr_x->at(i));
        self.svr_y->push_back(data_pvs.svr_y->at(i));
//         self.svr_pvr->push_back(data_pvs.svr_pvr->at(i));
//         self.sv_ntrks->push_back(nSVPrt(data_trks, data_pvs, data_hits, i));
    }
    
    for(int i = 0; i < data_trks.prt_x->size(); ++i) {
        self.prt_x->push_back(data_trks.prt_x->at(i));
        self.prt_y->push_back(data_trks.prt_y->at(i));
        self.prt_z->push_back(data_trks.prt_z->at(i));
        
        self.prt_px->push_back(data_trks.prt_px->at(i));
        self.prt_py->push_back(data_trks.prt_py->at(i));
        self.prt_pz->push_back(data_trks.prt_pz->at(i));
        
        self.prt_e->push_back(data_trks.prt_e->at(i));
    }
    
}


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

inline std::ostream &operator<<(std::ostream &stream, const DataPVsOut2 &self) {
    return stream << TString::Format("PVs: %lu", self.pvr_z->size());
}
