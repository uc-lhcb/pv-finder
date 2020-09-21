#include "fcn.h"
#include "data.h"
//#include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <iostream>

inline AnyTracks make_tracks(const CoreHitsIn &data) {
    // gets all hits, bins them in phi
    Hits hits;
    hits.newEvent(data);

    // make triplets
    Tracks tracks;

    tracks.newEvent(&hits);
    std::cout << " (" << tracks.ngood() << " good, " << tracks.nbad() << " bad)";

    return AnyTracks(tracks);
}

//void makez(AnyTracks& tracks, std::vector<DataKernelOut>& dks){
//    compute_over(tracks, [&dks](int b, std::vector<double> kernel, std::vector<double> x, std::vector<double> y){
//      //still no zip in c++17, so let's go by index and throw in a sanity check
//      auto const nkernel_definitions = dks.size();
//      if(nkernel_definitions!=kernel.size()) throw std::runtime_error("check how many kernels you want to write and how they are defined...");
//      for(auto i =0u; i < nkernel_definitions; i++){
//        dks[i].zdata[b] = static_cast<float>(kernel[i]);
//        dks[i].xmax [b] = static_cast<float>(dks[i].zdata[b]==0 ? 0.f : x[i]);
//        dks[i].ymax [b] = static_cast<float>(dks[i].zdata[b]==0 ? 0.f : y[i]);
//      }
//    });
//}
//
//// This is an (ugly) global pointer so that minuit can run a plain function
//AnyTracks* fcn_global_tracks = nullptr;

void makez(AnyTracks& tracks,std::vector<DataKernelOut*>& dks);

/// Run with e.g. root -b -q 'makehist.C+("10pvs","trks","../dat")'
void makehist(TString input, TString tree_name, TString folder, int nevents) {

    TFile f(folder + "/pv_"+input+".root");
    TTree *t = (TTree*)f.Get(tree_name);
    if(t == nullptr)
        throw std::runtime_error("Failed to get hits from file");

    TFile out(folder + "/kernel_"+input+".root", "RECREATE");

    int ntrack = nevents<1 ? t->GetEntries() : nevents;
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("kernel", "Output");
    std::vector<DataKernelOut*> dks{new DataKernelOut(&tout,"POCA"),new DataKernelOut(&tout,"old")};
    DataPVsOut dt(&tout);
    CoreReconTracksOut recon_out(&tout);

    for(int i=0; i<ntrack; i++) {
        for(auto& dk : dks) dk->clear();

        CoreHitsIn data_hits(t);
        CoreNHitsIn data_nhits(t);
        CorePVsIn data_pvs(t);
        CoreTruthTracksIn data_trks(t);

        t->GetEntry(i);
        std::cout << "Entry " << i << "/" << ntrack;

        AnyTracks tracks = make_tracks(data_hits);
        std::cout << " " << tracks;

        copy_in_pvs(dt, data_trks, data_pvs, data_nhits);
        copy_in(recon_out, tracks);

        makez(tracks, dks);

        std::cout << " " << dt << std::endl;
        tout.Fill();
    }

    out.Write();
}
