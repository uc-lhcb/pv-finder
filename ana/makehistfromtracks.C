#include "fcn.h"
#include "data.h"
#include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <memory>
#include <iostream>

void makez(AnyTracks& tracks, DataKernelOut& dk);

/// Run with e.g. root -b -q 'makehist.C+("10pvs","trks","../dat")'
/// Or run runall.sh
void makehistfromtracks(TString input, TString tree_name, TString folder, bool include_recon = true) {

    TFile f(folder + "/trks_"+input+".root");
    TTree *t = (TTree*)f.Get(tree_name);
    if(t == nullptr)
        throw std::runtime_error("Failed to get trks from file");

    TFile out(folder + "/kernel_"+input+".root", "RECREATE");

    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("kernel", "Output");
    DataKernelOut dk(&tout);
    DataPVsOut dt(&tout);

    std::unique_ptr<CoreReconTracksOut> recon_out;
    if(include_recon)
        recon_out.reset(new CoreReconTracksOut(&tout));

    for(int i=0; i<ntrack; i++) {
        dk.clear();

        CoreReconTracksIn data_recon(t);
        CoreNHitsIn data_nhits(t);
        CorePVsIn data_pvs(t);
        CoreTruthTracksIn data_trks(t);

        t->GetEntry(i);
        std::cout << "Entry " << i << "/" << ntrack;

        AnyTracks tracks(data_recon);
        std::cout << " " << tracks;

        copy_in_pvs(dt, data_trks, data_pvs, data_nhits);
        if(include_recon)
            copy_in(*recon_out, tracks);

        makez(tracks, dk);

        std::cout << " " << dt << std::endl;
        tout.Fill();
    }

    out.Write();
}
