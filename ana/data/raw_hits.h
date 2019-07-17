#pragma once

#include <TTree.h>

class DataHits {
    TTree *t;
public:
    vector<double> *pvx;
    vector<double> *pvy;
    vector<double> *pvz;
    vector<double> *svx;
    vector<double> *svy;
    vector<double> *svz;
    vector<double> *sv_ipv;
    vector<double> *px;
    vector<double> *py;
    vector<double> *pz;
    vector<double> *x;
    vector<double> *y;
    vector<double> *z;
    vector<double> *hx;
    vector<double> *hy;
    vector<double> *hz;
    vector<double> *hid;
    vector<double> *ntrks;
    vector<double> *ipv;
    vector<double> *nhits;
    
    DataHits(TTree *tree) : t(tree) {
        t->SetBranchAddress("pvr_x",&pvx);
        t->SetBranchAddress("pvr_y",&pvy);
        t->SetBranchAddress("pvr_z",&pvz);
        t->SetBranchAddress("svr_x",&svx);
        t->SetBranchAddress("svr_y",&svy);
        t->SetBranchAddress("svr_z",&svz);
        t->SetBranchAddress("svr_pvr",&sv_ipv);
        t->SetBranchAddress("prt_px",&px);
        t->SetBranchAddress("prt_py",&py);
        t->SetBranchAddress("prt_pz",&pz);
        t->SetBranchAddress("prt_x",&x);
        t->SetBranchAddress("prt_y",&y);
        t->SetBranchAddress("prt_z",&z);
        t->SetBranchAddress("prt_pvr",&ipv);
        t->SetBranchAddress("prt_hits",&nhits);
        t->SetBranchAddress("hit_x",&hx);
        t->SetBranchAddress("hit_y",&hy);
        t->SetBranchAddress("hit_z",&hz);
        t->SetBranchAddress("hit_prt",&hid);
        t->SetBranchAddress("ntrks_prompt",&ntrks);
    }
    
    ~DataHits(){
        // Only resetting the addresses we own
        t->ResetBranchAddresses();
    }
};
