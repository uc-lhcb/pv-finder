#include "fcn.h"
#include "data.h"
#include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <iostream>

void makez(AnyTracks& tracks, DataKernelOut& dk){
    compute_over(tracks, [&dk](int b, float kernel, float x, float y){
        dk.zdata[b] = kernel;
        dk.xmax[b] = (dk.zdata[b]==0 ? 0.f : x);
        dk.ymax[b] = (dk.zdata[b]==0 ? 0.f : y);
    });
}

// This is an (ugly) global pointer so that minuit can run a plain function
AnyTracks* fcn_global_tracks = nullptr;


/// Run with e.g. root -b -q 'makehist.C+("10pvs","trks","../dat")'
void makehist(TString input, TString tree_name, TString folder, const bool write_track_info, const bool verbose_track_info) {

    TFile f(folder + "/pv_"+input+".root");
    TTree *t = (TTree*)f.Get(tree_name);
    if(t == nullptr)
        throw std::runtime_error("Failed to get hits from file");

    TFile out(folder + "/kernel_"+input+".root", "RECREATE");

    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("kernel", "Output");
    DataKernelOut dk(&tout);
    DataPVsOut dt(&tout);
    CoreReconTracksOut recon_out(&tout,{"recon_x","recon_y","recon_z","recon_tx","recon_ty","recon_chi2"});
    if(verbose_track_info)
        recon_out.extend({"recon_pocax","recon_pocay","recon_pocaz","recon_sigmapocaxy"});

    for(int i=0; i<ntrack; i++) {
        dk.clear();

        CoreHitsIn data_hits(t);
        CoreNHitsIn data_nhits(t);
        CorePVsIn data_pvs(t);
        CoreTruthTracksIn data_trks(t);

        t->GetEntry(i);
        std::cout << "Entry " << i << "/" << ntrack;

        AnyTracks tracks = make_tracks(data_hits);
        std::cout << " " << tracks;

        copy_in_pvs(dt, data_trks, data_pvs, data_nhits);
        if(write_track_info) 
            copy_in(recon_out, tracks, verbose_track_info);

        makez(tracks, dk);
        
        std::cout << " " << dt << std::endl;
        tout.Fill();
    }

    out.Write();
}
