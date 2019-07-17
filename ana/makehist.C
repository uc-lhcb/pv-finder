#include "fcn.h"
#include "data/raw_hits.h"
#include "data/raw_kernel.h"
#include "data/raw_tracks.h"
#include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <iostream>

void makez(DataHits& data, DataKernel& dk){
    compute_over(data, [&dk](int b, float kernel, float x, float y){
        dk.zdata[b] = kernel;
        dk.xmax[b] = (dk.zdata[b]==0 ? 0.f : x);
        dk.ymax[b] = (dk.zdata[b]==0 ? 0.f : y);
    });
}

void copy_pvs(DataHits& data, DataTracks &dt) {
    
    dt.pv_n = data.pvz->size();
    for(int i=0; i<dt.pv_n; i++){
        dt.pv_cat[i] = pvCategory(data, i);
        dt.pv_loc[i] = data.pvz->at(i);
        dt.pv_loc_x[i] = data.pvx->at(i);
        dt.pv_loc_y[i] = data.pvy->at(i);
        dt.pv_ntrks[i] = ntrkInAcc(data, i);
    }
    
    dt.sv_n = data.svz->size();
    for(int i=0; i<dt.sv_n; i++){
        dt.sv_cat[i] = svCategory(data,i);
        dt.sv_loc[i] = data.svz->at(i);
        dt.sv_loc_x[i] = data.svx->at(i);
        dt.sv_loc_y[i] = data.svy->at(i);
        dt.sv_ntrks[i] = nSVPrt(data, i);
    }
}

void make_output(TString input, TString output) {
    TFile f(input);
    TTree *t = (TTree*) f.Get("data");
    
    TFile out(output, "RECREATE");
    
    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;
    


    TTree tout("kernel", "Output");
    DataTracks dt(&tout);
    
    for(int i=0; i<ntrack; i++) {
        cout << "Entry " << i << "/" << ntrack;
        
        DataHits data(t);
        t->GetEntry(i);
        
        copy_pvs(data, dt);
        
        cout << " PVs: " << dt.pv_n << " SVs: " << dt.sv_n << endl;
        tout.Fill();
    }
}

void read_input(TString input, TString output) {
    
}

/// Run with root -b -q 'makehist.C+("20180814")'
/// Or run runall.sh
void makehist(TString input, TString folder = "/data/schreihf/PvFinder") {

    TFile f(folder + "/pv_"+input+".root");
    TTree *t = (TTree*)f.Get("data");

    TFile out(folder + "/kernel_"+input+".root", "RECREATE");

    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("kernel", "Output");
    DataKernel dk(&tout);
    DataTracks dt(&tout);

    for(int i=0; i<ntrack; i++) {
        dk.clear();
        
        DataHits data(t);
        t->GetEntry(i);
        cout << "Entry " << i << "/" << ntrack;
        
        makez(data, dk);
        
        copy_pvs(data, dt);
        
        cout << " PVs: " << dt.pv_n << " SVs: " << dt.sv_n << endl;
        tout.Fill();
    }

    out.Write();
}
