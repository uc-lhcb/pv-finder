#pragma once

#include <TTree.h>
#include <array>

class DataKernelOut {
    TTree *t;

  public:
    std::array<float, 15000> zdata{};
    std::array<float, 15000> xmax{};
    std::array<float, 15000> ymax{};

    DataKernelOut(TTree *tree, const std::string&& prefix) : t(tree) {
        t->Branch((prefix+"zdata").data(), zdata.data(), (prefix+"zdata[15000]/F").data());
        t->Branch((prefix+"xmax").data(),  xmax.data(),  (prefix+"xmax[15000]/F").data());
        t->Branch((prefix+"ymax").data(),  ymax.data(),  (prefix+"ymax[15000]/F").data());
    }

    void clear() {
        zdata.fill(0);
        xmax.fill(0);
        ymax.fill(0);
    }
};
