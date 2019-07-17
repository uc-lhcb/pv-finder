#pragma once

#include <TTree.h>

class DataKernel {
    TTree *t;
public:
    
    float zdata[4000] = {0};
    float xmax[4000] = {0};
    float ymax[4000] = {0};
    
    DataKernel(TTree *tree) : t(tree) {
        t->Branch("zdata", zdata,"zdata[4000]/F");
        t->Branch("xmax", xmax, "xmax[4000]/F");
        t->Branch("ymax", ymax, "ymax[4000]/F");
    }
    
    void clear() {
        std::fill(zdata, zdata+4000, 0);
        std::fill(xmax, xmax+4000, 0);
        std::fill(ymax, ymax+4000, 0);
    }
};
