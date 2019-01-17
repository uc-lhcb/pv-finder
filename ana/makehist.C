#include "utils.h"
#include "data.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

#include <iostream>

using namespace std;

inline double bin_center(int nbins, double min, double max, int i) {
    return (i + 0.5) / nbins * (max - min) + min;
}

void makez(int event, TTree* t, int& pv_n, int& sv_n,
                                int* pv_cat, float* pv_loc, float* pv_loc_x, float* pv_loc_y, int* pv_ntrks,
                                int* sv_cat, float* sv_loc, float* sv_loc_x, float* sv_loc_y, int* sv_ntrks,
                                float* zdata, float* xmaxdata, float* ymaxdata){

  constexpr int nb = 4000;
  constexpr double zmin = -100.;
  constexpr double zmax = 300.;

  Data data;
  data.init(t);
  t->GetEntry(event);

  // gets all hits, bins them in phi
  Hits::instance()->newEvent(data);

  // make triplets
  Tracks *tracks = Tracks::instance();
  tracks->newEvent();
  cout << " Total tracks: " << tracks->n() << " good tracks: " << tracks->ngood() << " bad tracks: " << tracks->nbad();

  Point pv;

  // build the kernel vs z profiled in x-y
  // TODO: clearly non-optimal CPU-wise how this search is done
  for(int b=0; b<nb; b++){
    double z = bin_center(nb, zmin, zmax, b); 
    double kmax = -1.;
    double xmax = 0.;
    double ymax = 0.;

    // 1st do coarse grid search
    tracks->setRange(z);
    if(!tracks->run()) continue;

    for(double x=-0.4; x<=0.41; x+=0.1){
      for(double y=-0.4; y<=0.41; y+=0.1){
        pv.set(x,y,z);
        double val = kernel(pv);
        if(val > kmax){
          kmax=val;
          xmax=x;
          ymax=y;
        }
      }
    }

    // now do gradient descent from max found
    pv.set(xmax,ymax,z);
    zdata[b] = (float) kernelMax(pv);
    xmaxdata[b] = (float) (zdata[b]==0 ? 0 : pv.x());
    ymaxdata[b] = (float) (zdata[b]==0 ? 0 : pv.y());
  }
    
    pv_n = data.pvz->size();
    for(int i=0; i<pv_n; i++){
      pv_cat[i] = pvCategory(data, i);
      pv_loc[i] = data.pvz->at(i);
      pv_loc_x[i] = data.pvx->at(i);
      pv_loc_y[i] = data.pvy->at(i);
      pv_ntrks[i] = ntrkInAcc(data, i);
    }
    
    sv_n = data.svz->size();
    for(int i=0; i<sv_n; i++){
        sv_cat[i] = svCategory(data,i);
        sv_loc[i] = data.svz->at(i);
        sv_loc_x[i] = data.svx->at(i);
        sv_loc_y[i] = data.svy->at(i);
        sv_ntrks[i] = nSVPrt(data, i);
    }
}


/// Run with root -b -q 'makehist.C+("20180814")'
/// Or run runall.sh
void makehist(TString input) {

    TFile f("/data/schreihf/PvFinder/pv_"+input+".root");
    TTree *t = (TTree*)f.Get("data");

    TFile out("/data/schreihf/PvFinder/kernel_"+input+".root", "RECREATE");
    
    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;
    
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
    
    float zdata[4000];
    float xmax[4000];
    float ymax[4000];
    
    int pv_n, sv_n;

    TTree *tout = new TTree("kernel", "Output");
    tout->Branch("pv_n", &pv_n, "pv_n/I");
    tout->Branch("sv_n", &sv_n, "sv_n/I");
    
    tout->Branch("pv_cat", pv_cat, "pv_cat[pv_n]/I");
    tout->Branch("pv_loc", pv_loc, "pv_loc[pv_n]/F");
    tout->Branch("pv_loc_x", pv_loc_x, "pv_loc_x[pv_n]/F");
    tout->Branch("pv_loc_y", pv_loc_y, "pv_loc_y[pv_n]/F");
    tout->Branch("pv_ntrks", pv_ntrks, "pv_ntrks[pv_n]/I");
    
    tout->Branch("sv_cat", sv_cat, "sv_cat[sv_n]/I");
    tout->Branch("sv_loc", sv_loc, "sv_loc[sv_n]/F");
    tout->Branch("sv_loc_x", sv_loc_x, "sv_loc_x[sv_n]/F");
    tout->Branch("sv_loc_y", sv_loc_y, "sv_loc_y[sv_n]/F");
    
    tout->Branch("sv_ntrks", sv_ntrks, "sv_ntrks[sv_n]/I");
    
    tout->Branch("zdata",zdata,"zdata[4000]/F");
    tout->Branch("xmax", xmax, "xmax[4000]/F");
    tout->Branch("ymax", ymax, "ymax[4000]/F");
    
    for(int i=0; i<ntrack; i++) {
        std::fill(zdata, zdata+4000, 0);
        cout << "Entry " << i << "/" << ntrack;
        makez(i, t,
              pv_n, sv_n,
              pv_cat, pv_loc, pv_loc_x, pv_loc_y, pv_ntrks,
              sv_cat, sv_loc, sv_loc_x, sv_loc_y, sv_ntrks,
              zdata, xmax, ymax);
        cout << " PVs: " << pv_n << " SVs: " << sv_n << endl;
        tout->Fill();
    }
    tout->Write();
}
