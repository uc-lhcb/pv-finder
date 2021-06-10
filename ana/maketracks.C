#include "fcn.h"
#include "data.h"
#include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <memory>
#include <iostream>

/// Convert raw data to tracks
void make_tracks(TString input, TString tree_name, TString folder, bool include_recon = true) {
    TFile f(folder + "/pv_"+input+".root");
    TTree *t = (TTree*) f.Get(tree_name);
    if(t == nullptr)
        throw std::runtime_error("Failed to get hits from file");

    TFile out(folder + "/trks_"+input+".root", "RECREATE");

    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("trks", "Tracks");
    DataPVsOut dt(&tout);

    std::unique_ptr<CoreReconTracksOut> recon_out;
    if(include_recon)
        recon_out.reset(new CoreReconTracksOut(&tout));

    CoreNHitsOut nhits_out(&tout);
    CorePVsOut pvs_out(&tout);
    CoreTruthTracksOut truth_out(&tout);

    for(int i=0; i<ntrack; i++) {

        CoreHitsIn data_hits(t);
        CoreNHitsIn data_nhits(t);
        CorePVsIn data_pvs(t);
        CoreTruthTracksIn data_trks(t);

        t->GetEntry(i);
        std::cout << "Entry " << i << "/" << ntrack;

        AnyTracks tracks = make_tracks(data_hits);

        copy_in_pvs(dt, data_trks, data_pvs, data_nhits);

        if(include_recon)
            copy_in(*recon_out, tracks);
        pvs_out.copy_in(data_pvs);
        truth_out.copy_in(data_trks);
        nhits_out.copy_in(data_nhits);

        std::cout << " " << dt << std::endl;

        tout.Fill();
    }
    tout.Write();
}
