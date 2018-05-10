#include "lhcbStyle.h"

#include "utils.h"
#include "data.h"

void makez(int event, TTree* t, int& pv_n, int& sv_n,
                                int* pv_cat, float* pv_loc,
                                int* sv_cat, float* sv_loc,
                                float* zdata){
    
  auto hzkernel = new TH1F("hzkernel","",4000,-100,300);

  Data data;
  data.init(t);
  t->GetEntry(event);

  // gets all hits, bins them in phi
  Hits::instance()->newEvent(data);

  // make triplets
  Tracks *tracks = Tracks::instance();
  tracks->newEvent();
  cout << "Total tracks: " << tracks->n() << " good tracks: " << tracks->ngood() << " bad tracks: " << tracks->nbad() << endl;

  int nb=hzkernel->GetNbinsX();
  Point pv;

  // build the kernel vs z profiled in x-y
  // TODO: clearly non-optimal CPU-wise how this search is done
  for(int b=1; b<=nb; b++){
    double z = hzkernel->GetBinCenter(b);
    double kmax=-1,xmax,ymax;

    // 1st do coarse grid search
    tracks->setRange(z);
    if(!tracks->run()) continue;

    for(double x=-0.4; x<=0.4; x+=0.1){
      for(double y=-0.4; y<=0.4; y+=0.1){
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
    double kernel = kernelMax(pv);
    hzkernel->SetBinContent(b,kernel);
    zdata[b-1] = (float) kernel;
  }
    
    pv_n = data.pvz->size();
  for(int i=0; i<pv_n; i++){
    pv_cat[i] = pvCategory(data,i);
    pv_loc[i] = data.pvz->at(i);
  }
    
    sv_n = data.svz->size();
    for(int i=0; i<sv_n; i++){
        sv_cat[i] = svCategory(data,i);
        sv_loc[i] = data.svz->at(i);
    }
    
    delete hzkernel;
}

void makehist() {
    TFile f("/data/schreihf/PvFinder/pv_20180509.root");
    TTree *t = (TTree*)f.Get("data");

    TFile out("/data/schreihf/PvFinder/kernel_20180509.root", "RECREATE");
    
    int ntrack = t->GetEntries();
    std::cout << "Number of entries to read in: " << ntrack << std::endl;
    
    int pv_cat[MAX_TRACKS];
    float pv_loc[MAX_TRACKS];
    int sv_cat[MAX_TRACKS];
    float sv_loc[MAX_TRACKS];
    float zdata[4000];
    int pv_n, sv_n;

    TTree *tout = new TTree("kernel","Output");
    tout->Branch("pv_n",&pv_n,"pv_n/I");
    tout->Branch("sv_n",&sv_n,"sv_n/I");
    tout->Branch("pv_cat",pv_cat,"pv_cat[pv_n]/I");
    tout->Branch("pv_loc",pv_loc,"pv_loc[pv_n]/F");
    tout->Branch("sv_cat",sv_cat,"sv_cat[sv_n]/I");
    tout->Branch("sv_loc",sv_loc,"sv_loc[sv_n]/F");
    tout->Branch("zdata",zdata,"zdata[4000]/F");
    
    for(int i=0; i<ntrack; i++) {
        std::fill(zdata, zdata+4000, 0);
        makez(i, t, pv_n, sv_n, pv_cat, pv_loc, sv_cat, sv_loc, zdata);
        cout << "PVs: " << pv_n << " SVs: " << sv_n << endl;
        tout->Fill();
    }
    tout->Write();
}
