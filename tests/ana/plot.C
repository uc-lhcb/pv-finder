#include "lhcbStyle.h"

#include "fcn.h"
#include "data.h"
#include "compute_over.h"

void plotz(int event){
  lhcbStyle();
  TH1F hzkernel("hzkernel","",4000,-100,300);

  TFile f("../dat/test_10pvs.root");
  TTree *t = (TTree*) f.Get("data");

  Data data;
  data.init(t);
  t->GetEntry(event);

  compute_over(data, [&hzkernel](int b, double kernel, double, double){
        hzkernel.SetBinContent(b,kernel);
  });

  hzkernel.SetMinimum(0);
  hzkernel.DrawCopy();

  // draw lines at true pv and sv locations
  TLine *line = new TLine();
  line->SetLineColor(kRed);
  for(int i=0; i<data.pvz->size(); i++){
    int cat = pvCategory(data,i);
    if(cat < 0) continue;
    if(cat == 0) line->SetLineStyle(2);
    else line->SetLineStyle(1);
    line->DrawLine(data.pvz->at(i),0,data.pvz->at(i),hzkernel.GetMaximum());
  }
  line->SetLineColor(kBlue);
  for(int i=0; i<data.svz->size(); i++){
    int cat = svCategory(data,i);
    if(cat < 0) continue;
    if(cat == 0) line->SetLineStyle(2);
    else line->SetLineStyle(1);
    line->DrawLine(data.svz->at(i),0,data.svz->at(i),hzkernel.GetMaximum());
  }
}

void plotxy(int event, int which_pv, double zshift){
   lhcbStyle();

   const int Number = 2;
   double Red[Number]    = {1,0};
   double Green[Number]  = {1,0};
   double Blue[Number]   = {1,0};
   double Length[Number] = {0,1};
   int nbcolor=512;
   TColor::CreateGradientColorTable(Number,Length,Red,Green,Blue,nbcolor);
   gStyle->SetNumberContours(nbcolor);

   TH1F hbins("hbins","",100,-0.5,0.5);
   TH2F hxykernel("hxykernel","",100,-0.5,0.5,100,-0.5,0.5);

   TFile f("../dat/test_10pvs.root");
   TTree *t = (TTree*)f.Get("data");
   Data data;
   data.init(t);
   t->GetEntry(event);

   // gets all hits, bins them in phi
   Hits hits;
   hits.newEvent(data);

   Tracks tracks;

   // C style workaround for global FCN tracks
   fcn_global_tracks = &tracks;

   // make triplets
   tracks.newEvent(&hits);

   double z = data.pvz->at(which_pv)+zshift;
   tracks.setRange(z);
   Point pv;
   for(int bx=1; bx<=100; bx++){
     double xval = hbins.GetBinCenter(bx);
     for(int by=1; by<=100; by++){
       double yval = hbins.GetBinCenter(by);
       pv.set(xval,yval,z);
       hxykernel.SetBinContent(bx,by,kernel(pv));
     }
   }
   hxykernel.SetMinimum(0);
   //hxykernel.SetMaximum(1000);
   hxykernel.DrawCopy("colz");
   // gPad->SetLogz();

   TMarker *marker = new TMarker();
   marker->SetMarkerStyle(5);
   marker->SetMarkerSize(10);
   marker->SetMarkerColor(kMagenta);
   marker->DrawMarker(data.pvx->at(which_pv),data.pvy->at(which_pv));
   std::cout << "pv: " << data.pvx->at(which_pv) << " " << data.pvy->at(which_pv) << " " << data.pvz->at(which_pv) << std::endl;
   std::cout << "cat: " << pvCategory(data,which_pv) << std::endl;

   marker->SetMarkerColor(kCyan);
   for(int i=0; i<data.svz->size(); i++){
     if(data.sv_ipv->at(i) != which_pv) continue;
     int cat = svCategory(data,i);
     if(cat < 0) continue;
     if(cat == 0) marker->SetMarkerStyle(2);
     else marker->SetMarkerStyle(3);
     std::cout << data.svx->at(i) << " " << data.svy->at(i) << " " << data.svz->at(i) << std::endl;
     marker->DrawMarker(data.svx->at(i),data.svy->at(i));
   }

}
