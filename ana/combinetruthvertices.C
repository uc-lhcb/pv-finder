#include "fcn.h"
#include "data.h"
// #include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <iostream>

void combinetruthvertices(TString input, TString output, TString tree_name, TString folder, int nevents) {
    //nevents = 1;
    
    TFile f(folder + "/trks_"+input+".root");
    TTree *t = (TTree*)f.Get(tree_name);
    if(t == nullptr)
        throw std::runtime_error("Failed to get trks from file");

    TFile out(folder + "/trks_"+output+".root", "RECREATE");
    
    int ntrack = nevents<1 ? t->GetEntries() : nevents;
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("trks", "Output");
    
    DataPVsOut2 dt(&tout);
    CoreReconTracksOut recon_out(&tout);
    std::unordered_map<std::string,std::vector<double>> dump_data;
    
    Trajectory beamline(0., 0., 0., 0., 0.);
    for(int i=0; i<ntrack; i++) {
        
        CoreReconTracksIn data_recon(t);
        CoreNHitsIn data_nhits(t);
        CorePVsIn2 data_pvs(t);
        CoreTruthTracksIn2 data_trks(t);
        
        t->GetEntry(i);
        
        std::cout << "Entry " << i << "/" << ntrack;

        AnyTracks tracks(data_recon);
        std::cout << " " << tracks;
        
        copy_in_pvs2(dt, data_trks, data_pvs, data_nhits);
        copy_in(recon_out,tracks);

        std::cout << " " << dt << std::endl;
        tout.Fill();

    }

    out.Write();

}
