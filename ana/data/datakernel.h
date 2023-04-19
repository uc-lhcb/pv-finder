#pragma once

#include <TTree.h>
#include <array>

class DataKernelOut {
    TTree *t;

  public:
    std::array<float, NBINS> zdata{};
    std::array<float, NBINS> xmax{};
    std::array<float, NBINS> ymax{};

    

    DataKernelOut(TTree *tree) : t(tree) {
        t->Branch("zdata", zdata.data(), TString::Format("zdata[%d]/F", NBINS));
        t->Branch("xmax", xmax.data(), TString::Format("xmax[%d]/F", NBINS));
        t->Branch("ymax", ymax.data(), TString::Format("ymax[%d]/F", NBINS));
    }

    void clear() {
        zdata.fill(0);
        xmax.fill(0);
        ymax.fill(0);
    }
};
