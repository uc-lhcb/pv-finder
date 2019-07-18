#pragma once

#include <TTree.h>

class DataHits {
    TTree *t;
public:
    vector<double> *pvx = nullptr;
    vector<double> *pvy = nullptr;
    vector<double> *pvz = nullptr;
    vector<double> *svx = nullptr;
    vector<double> *svy = nullptr;
    vector<double> *svz = nullptr;
    vector<double> *sv_ipv = nullptr;
    vector<double> *px = nullptr;
    vector<double> *py = nullptr;
    vector<double> *pz = nullptr;
    vector<double> *x = nullptr;
    vector<double> *y = nullptr;
    vector<double> *z = nullptr;
    vector<double> *hx = nullptr;
    vector<double> *hy = nullptr;
    vector<double> *hz = nullptr;
    vector<double> *hid = nullptr;
    vector<double> *ntrks = nullptr;
    vector<double> *ipv = nullptr;
    vector<double> *nhits = nullptr;
    
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
