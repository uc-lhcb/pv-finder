
#pragma once

#include <TTree.h>

class DataTracks {
    TTree *t;
public:
    int pv_n, sv_n;
    
    float pv_loc[MAX_TRACKS];
    float pv_loc_x[MAX_TRACKS];
    float pv_loc_y[MAX_TRACKS];
    
    int pv_cat[MAX_TRACKS];
    int pv_ntrks[MAX_TRACKS];
    
    float sv_loc[MAX_TRACKS];
    float sv_loc_x[MAX_TRACKS];
    float sv_loc_y[MAX_TRACKS];
    
    int sv_cat[MAX_TRACKS];
    int sv_ntrks[MAX_TRACKS];
    
    DataTracks(TTree *tree) : t(tree) {
        t->Branch("pv_n", &pv_n, "pv_n/I");
        t->Branch("sv_n", &sv_n, "sv_n/I");
        
        t->Branch("pv_cat", pv_cat, "pv_cat[pv_n]/I");
        t->Branch("pv_loc", pv_loc, "pv_loc[pv_n]/F");
        t->Branch("pv_loc_x", pv_loc_x, "pv_loc_x[pv_n]/F");
        t->Branch("pv_loc_y", pv_loc_y, "pv_loc_y[pv_n]/F");
        t->Branch("pv_ntrks", pv_ntrks, "pv_ntrks[pv_n]/I");
        
        t->Branch("sv_cat", sv_cat, "sv_cat[sv_n]/I");
        t->Branch("sv_loc", sv_loc, "sv_loc[sv_n]/F");
        t->Branch("sv_loc_x", sv_loc_x, "sv_loc_x[sv_n]/F");
        t->Branch("sv_loc_y", sv_loc_y, "sv_loc_y[sv_n]/F");
        
        t->Branch("sv_ntrks", sv_ntrks, "sv_ntrks[sv_n]/I");
    }
};


class DataTracksIn {
    TTree *t;
public:
    int pv_n, sv_n;
    
    float **pv_loc;
    float **pv_loc_x;
    float **pv_loc_y;
    
    int **pv_cat;
    int **pv_ntrks;
    
    float **sv_loc;
    float **sv_loc_x;
    float **sv_loc_y;
    
    int **sv_cat;
    int **sv_ntrks;
    
    DataTracksIn(TTree *tree) : t(tree) {
        t->SetBranchAddress("pv_n", &pv_n);
        t->SetBranchAddress("sv_n", &sv_n);
        
        t->SetBranchAddress("pv_cat", pv_cat);
        t->SetBranchAddress("pv_loc", pv_loc);
        t->SetBranchAddress("pv_loc_x", pv_loc_x);
        t->SetBranchAddress("pv_loc_y", pv_loc_y);
        t->SetBranchAddress("pv_ntrks", pv_ntrks);
        
        t->SetBranchAddress("sv_cat", sv_cat);
        t->SetBranchAddress("sv_loc", sv_loc);
        t->SetBranchAddress("sv_loc_x", sv_loc_x);
        t->SetBranchAddress("sv_loc_y", sv_loc_y);
        
        t->SetBranchAddress("sv_ntrks", sv_ntrks);
    }
    
    ~DataTracksIn(){
        // Only resetting the addresses we own
        t->ResetBranchAddresses();
    }
};
