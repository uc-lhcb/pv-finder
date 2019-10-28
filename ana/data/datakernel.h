#pragma once

#include <TTree.h>
#include <array>

class DataKernelOut {
    TTree *t;

  public:
    std::array<float, 4000> zdata{};
    std::array<float, 4000> xmax{};
    std::array<float, 4000> ymax{};

    DataKernelOut(TTree *tree) : t(tree) {
        t->Branch("zdata", zdata.data(), "zdata[4000]/F");
        t->Branch("xmax", xmax.data(), "xmax[4000]/F");
        t->Branch("ymax", ymax.data(), "ymax[4000]/F");
    }

    void clear() {
        zdata.fill(0);
        xmax.fill(0);
        ymax.fill(0);
    }
};
