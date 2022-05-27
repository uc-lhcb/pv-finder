#pragma once

#include <TTree.h>
#include <array>

class DataKernelOut {
    TTree *t;

  public:
    std::array<float, 12000> zdata{};
    std::array<float, 12000> xmax{};
    std::array<float, 12000> ymax{};

    DataKernelOut(TTree *tree, const std::string&& prefix) : t(tree) {
        t->Branch((prefix+"zdata").data(), zdata.data(), (prefix+"zdata[12000]/F").data());
        t->Branch((prefix+"xmax").data(),  xmax.data(),  (prefix+"xmax[12000]/F").data());
        t->Branch((prefix+"ymax").data(),  ymax.data(),  (prefix+"ymax[12000]/F").data());
    }

    void clear() {
        zdata.fill(0);
        xmax.fill(0);
        ymax.fill(0);
    }
};
