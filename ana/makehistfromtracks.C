#include "fcn.h"
#include "data.h"
#include "compute_over.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <iostream>

void makez(AnyTracks& tracks, std::vector<DataKernelOut*>& dks){
   compute_over(tracks, [&dks](int b, std::vector<double> kernel, std::vector<double> x, std::vector<double> y){
     //still no zip in c++17, so let's go by index and throw in a sanity check
     auto const nkernel_definitions = dks.size();
     if(nkernel_definitions!=kernel.size()) throw std::runtime_error("check how many kernels you want to write and how they are defined...");
     for(auto i =0u; i < nkernel_definitions; i++){
       dks[i]->zdata[b] = kernel[i];
       dks[i]->xmax [b] = (dks[i]->zdata[b]==0 ? 0.f : x[i]);
       dks[i]->ymax [b] = (dks[i]->zdata[b]==0 ? 0.f : y[i]);
     }
   });
}

//void makez(AnyTracks& tracks,std::vector<DataKernelOut>& dks);

AnyTracks* fcn_global_tracks = nullptr;
void makehistfromtracks(TString input, TString tree_name, TString folder, int nevents) {

    //nevents = 1;
    
    TFile f(folder + "/trks_"+input+".root");
    TTree *t = (TTree*)f.Get(tree_name);
    if(t == nullptr)
        throw std::runtime_error("Failed to get trks from file");

    TFile out(folder + "/kernel_"+input+".root", "RECREATE");

    int ntrack = nevents<1 ? t->GetEntries() : nevents;
    std::cout << "Number of entries to read in: " << ntrack << std::endl;

    TTree tout("kernel", "Output");
    std::vector<DataKernelOut*> dks{new DataKernelOut(&tout,"POCA"),new DataKernelOut(&tout,"POCA_sq"),new DataKernelOut(&tout,"old")};
    DataPVsOut dt(&tout);
    CoreReconTracksOut recon_out(&tout);
    std::unordered_map<std::string,std::vector<double>> dump_data;
    std::vector<std::string> branch_names{"POCA_minor_axis1_x", "POCA_minor_axis1_y", "POCA_minor_axis1_z",
                                          "POCA_minor_axis2_x", "POCA_minor_axis2_y", "POCA_minor_axis2_z",
                                          "POCA_major_axis_x", "POCA_major_axis_y", "POCA_major_axis_z",
                                          "POCA_center_x", "POCA_center_y", "POCA_center_z"};
    for(auto const& b : branch_names)
      dump_data.emplace(b,std::vector<double>());
    for(auto& dd : dump_data)
      tout.Branch(dd.first.c_str(),&dd.second);
    Trajectory beamline(0., 0., 0., 0., 0.);

    for(int i=0; i<ntrack; i++) {
        for(auto& dk : dks) dk->clear();

        CoreReconTracksIn data_recon(t);
        CoreNHitsIn data_nhits(t);
        CorePVsIn data_pvs(t);
        CoreTruthTracksIn data_trks(t);

        t->GetEntry(i);
        std::cout << "Entry " << i << "/" << ntrack;

        AnyTracks tracks(data_recon);
        std::cout << " " << tracks;
        
        int trkcount = 0;
        
        // make poca error ellipsoids for each track w.r.t. the beamline (quick and dirty solution...)
        for(auto const &trajectory : tracks.trajectories()){
          const auto tsigmapocaxy = tracks.at(trkcount).get_sigmapocaxy(); // EMK
          const auto terrz0 = tracks.at(trkcount).get_errz0(); // EMK
            
          Ellipsoid ellipsoid(beamline, trajectory, tsigmapocaxy, terrz0, 0.005);
          dump_data["POCA_minor_axis1_x"].emplace_back(ellipsoid.minor_axis1().x());
          dump_data["POCA_minor_axis1_y"].emplace_back(ellipsoid.minor_axis1().y());
          dump_data["POCA_minor_axis1_z"].emplace_back(ellipsoid.minor_axis1().z());
          dump_data["POCA_minor_axis2_x"].emplace_back(ellipsoid.minor_axis2().x());
          dump_data["POCA_minor_axis2_y"].emplace_back(ellipsoid.minor_axis2().y());
          dump_data["POCA_minor_axis2_z"].emplace_back(ellipsoid.minor_axis2().z());
          dump_data["POCA_major_axis_x"].emplace_back(ellipsoid.major_axis().x());
          dump_data["POCA_major_axis_y"].emplace_back(ellipsoid.major_axis().y());
          dump_data["POCA_major_axis_z"].emplace_back(ellipsoid.major_axis().z());
          dump_data["POCA_center_x"].emplace_back(ellipsoid.center().x());
          dump_data["POCA_center_y"].emplace_back(ellipsoid.center().y());
          dump_data["POCA_center_z"].emplace_back(ellipsoid.center().z());
            
          trkcount++;
        }

        copy_in_pvs(dt, data_trks, data_pvs, data_nhits);
        copy_in(recon_out,tracks);

        makez(tracks, dks);

        std::cout << " " << dt << std::endl;
        tout.Fill();
        //clean the map for the next event
        for(auto& kv : dump_data) kv.second = std::vector<double>();

    }

    out.Write();
}
