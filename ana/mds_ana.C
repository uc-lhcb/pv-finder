#define mds_ana_cxx
#include "mds_ana.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TGraphErrors.h>
#include <cstdlib>
#include <math.h>
#include <iostream>


using namespace std;

void mds_ana::Loop()
{
//   In a ROOT session, you can do:
//      root> .L mds_ana.C
//      root> mds_ana t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

  std::cout << "getting started;  sanity check" << std::endl;

  TH1D * PrV_x = new TH1D("PrV x","no selection",100, -0.2, 0.2);
  TH1D * PrV_y = new TH1D("PrV y","no selection",100, -0.2, 0.2);
  TH1D * PrV_z = new TH1D("PrV z","no selection",400, -100.,300.);
  TH1D * PrV_z_4K = new TH1D("PrV v, fine binned","no selection",
           4000, -100., 300.);

  TH1D * deltaPhiHist = new TH1D("delta phi","all hits", 100, -0.02, 0.02);
  TH1D*  deltaThxHist = new TH1D("delta theta_x","all hits", 100, -0.01,0.01);
  TH1D*  deltaThyHist = new TH1D("delta theta_y","all hits", 100, -0.01,0.01);

  TH1D * deltaPhiHist_5GeV = new TH1D("delta phi; ","hits from tracks > 5 GeV", 100, -0.02, 0.02);
  TH1D*  deltaThxHist_5GeV = new TH1D("delta theta_x;","hits from tracks > 5 GeV", 100, -0.01,0.01);
  TH1D*  deltaThyHist_5GeV = new TH1D("delta theta_y;","hits from tracks > 5 GeV", 100, -0.01,0.01);

  TH1D* rmsDeltaPhi = new TH1D("rms delta #phi","from hits on particles in a PV", 100, 0.0, 0.020);

  vector <TH1D *> deltaPhiShiftedInX;
  vector <TH1D *> deltaThxShiftedInX;
  vector <TH1D *> deltaThyShiftedInX;
  vector<double> shiftx_list;
  shiftx_list.push_back(0.025);      //  25 microns = 0.025 mm
  shiftx_list.push_back(0.050);      //  50 microns = 0.050 mm
  shiftx_list.push_back(0.100);
  for (size_t shift_index=0; shift_index<shiftx_list.size(); shift_index++) {
   double shift_value = shiftx_list.at(shift_index)*1000.;  // from mm --> microns
   TString shiftStr;
   shiftStr.Form("%3.0f",shift_value);
// mds    std::cout <<  "in shiftx_list loop with shift_index = " << shift_index
// mds        << "  and shift_value = " << shift_value << " with shiftStr = "
// mds        <<  shiftStr << std::endl;
   deltaThxShiftedInX.push_back(  new TH1D("#D #theta_x shifted in x by"+shiftStr,"all hits",
    100, -0.01, 0.01) );
   deltaThyShiftedInX.push_back(  new TH1D("#D #theta_y shifted in x by"+shiftStr,"all hits",
    100, -0.01, 0.01) );
   deltaPhiShiftedInX.push_back(  new TH1D("#D #phi shifted in x by"+shiftStr,"all hits",
    100, -0.02, 0.02) );
  }  //  end of shiftx for
  std::cout << "exited shiftx loop" << std::endl << std::endl;




   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     Long64_t ientry = LoadTree(jentry);
     if (ientry < 0) break;
     nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;

//  create these vectors of vectors inside the jentry lopp so
//  they go out of scope at the end of each event and get re-created
//  otherwise, they just keep growing

     vector<vector <int>> PrV_ParticleIndices;
     vector<vector <int>> Particle_HitIndices;

     for (size_t i = 0; i<pvr_x->size(); i++) {
       Double_t x = pvr_x->at(i);
       PrV_x->Fill(x);
       PrV_y->Fill(pvr_y->at(i));
       PrV_z->Fill(pvr_z->at(i));
       PrV_z_4K->Fill(pvr_z->at(i));
     }

     for (size_t i=0; i<hit_z->size(); i++) {
       int particle_index = hit_prt->at(i);
       int prv_index = prt_pvr->at(particle_index);
       double x0 = pvr_x->at(prv_index);
       double y0 = pvr_y->at(prv_index);
       double z0 = pvr_z->at(prv_index);
       double hit_phi = atan2(hit_y->at(i)-y0,hit_x->at(i)-x0);
       double track_phi = atan2(prt_py->at(particle_index),prt_px->at(particle_index));
       double hit_thx = atan2(hit_x->at(i)-x0,hit_z->at(i)-z0);
       double track_thx = atan2(prt_px->at(particle_index),prt_pz->at(particle_index));
       double hit_thy = atan2(hit_y->at(i)-y0,hit_z->at(i)-z0);
       double track_thy = atan2(prt_py->at(particle_index),prt_pz->at(particle_index));

       double delta_phi = hit_phi - track_phi;
       double delta_thx = hit_thx - track_thx;
       double delta_thy = hit_thy - track_thy;

       deltaPhiHist->Fill(delta_phi);
       deltaThxHist->Fill(delta_thx);
       deltaThyHist->Fill(delta_thy);

       if (prt_e->at(particle_index) > 5) {   //  Pythia is said to use GeV
         deltaPhiHist_5GeV->Fill(delta_phi);
         deltaThxHist_5GeV->Fill(delta_thx);
         deltaThyHist_5GeV->Fill(delta_thy);
       }


//    not precisely the right question, but easy to ask, and should provide
//    some insight. conflates shift of "road" and dispersion in road.
       for (size_t shift_index=0; shift_index<shiftx_list.size(); shift_index++) {
         x0 = x0+shiftx_list.at(shift_index);
         double hit_phi = atan2(hit_y->at(i)-y0,hit_x->at(i)-x0);
         double hit_thx = atan2(hit_x->at(i)-x0,hit_z->at(i)-z0);
         double hit_thy = atan2(hit_y->at(i)-y0,hit_z->at(i)-z0);
         double delta_phi = hit_phi - track_phi;
         double delta_thx = hit_thx - track_thx;
         double delta_thy = hit_thy - track_thy;
         deltaPhiShiftedInX.at(shift_index)->Fill(delta_phi);
         deltaThxShiftedInX.at(shift_index)->Fill(delta_thx);
         deltaThyShiftedInX.at(shift_index)->Fill(delta_thy);
        }

     }  //  end of loop over hits

     for (size_t prv_index = 0; prv_index<pvr_x->size(); prv_index++) {
       vector<int> prt_from_prv;
       for (size_t prt_index=0; prt_index<prt_x->size(); prt_index++) {
         int prv = prt_pvr->at(prt_index);
         if (prv == (int)prv_index) {
           prt_from_prv.push_back(prt_index);
         }
        }
        PrV_ParticleIndices.push_back(prt_from_prv);
     }

//  some code to check that PrV_ParticleIndices looks right
/*
     if (jentry < 5) {
      std::cout << "  jentry =  " << jentry << std::endl << std::endl;
      for (size_t i=0; i<PrV_ParticleIndices.size(); i++) {
        std::cout << " vertex " << i << " has " <<  PrV_ParticleIndices[i].size()
             << " particles:  "  << std::endl;
        for (size_t j=0; j<PrV_ParticleIndices[i].size(); j++) {
          std::cout << PrV_ParticleIndices[i][j] << " ";
        }
         std::cout << std::endl;
       }
     }
*/

     for (size_t prt_index = 0; prt_index<prt_x->size(); prt_index++) {
       vector<int> hit_from_prt;
       for (size_t hit_index=0; hit_index<hit_x->size(); hit_index++) {
         int prt = hit_prt->at(hit_index);
         if (prt == (int)prt_index) {
           hit_from_prt.push_back(hit_index);
         }
        }
        Particle_HitIndices.push_back(hit_from_prt);
     }

//  some code to check that Particle_HitIndices looks right
/*
     if (jentry < 5) {
      std::cout << "  jentry =  " << jentry << std::endl ;
      for (size_t i=0; i<Particle_HitIndices.size(); i++) {
        std::cout << " particle " << i << " has " <<  Particle_HitIndices[i].size()
             << " hits:  "  << std::endl;
        for (size_t j=0; j<Particle_HitIndices[i].size(); j++) {
          std::cout << Particle_HitIndices[i][j] << " ";
        }
         std::cout << std::endl;
       }
     }
*/
      for (size_t prv_index=0; prv_index<PrV_ParticleIndices.size(); prv_index++) {
//  initialize whatever for this vertex
        double x0 = pvr_x->at(prv_index);
        double y0 = pvr_y->at(prv_index);
        double z0 = pvr_z->at(prv_index);
        double sum_delta_phi_sq = 0;
        int nHits = 0;
/*  for testing & debugging
        if (jentry < 5) {
          std::cout << " event " << jentry << "  prv_index = " << prv_index
               << "  PrV_ParticleIndices[prv_index].size() = "
               <<  PrV_ParticleIndices[prv_index].size() << std::endl;
        }
*/
        if (PrV_ParticleIndices[prv_index].size() >  4) { // at least 5 prts in prv
          for (size_t j=0; j<PrV_ParticleIndices[prv_index].size(); j++) {
            int prt_index = PrV_ParticleIndices[prv_index][j];
/*  for testing & debugging
            if (jentry<5) {
              std::cout << "j = " << j  << "  prt_index = " << prt_index
                   << "  Particle_HitIndices[prt_index].size() = "
                   << Particle_HitIndices[prt_index].size() << std::endl;
            }
*/
            if (Particle_HitIndices[prt_index].size() > 3)  {  // at least 4 hits from prt

              double track_phi = atan2(prt_py->at(prt_index),prt_px->at(prt_index));
/*
              if (jentry<5) {
                std::cout << "track_phi = " << track_phi << std::endl;
              }
*/
              double track_thx = atan2(prt_px->at(prt_index),prt_pz->at(prt_index));
              double track_thy = atan2(prt_py->at(prt_index),prt_pz->at(prt_index));
              for (size_t k=0; k<Particle_HitIndices[prt_index].size(); k++) {
                int hit_index = Particle_HitIndices[prt_index][k];
                double hit_phi = atan2(hit_y->at(hit_index)-y0,hit_x->at(hit_index)-x0);
/*
                if (jentry<5) {
                std::cout << "k = " << k << "  hit_index = " << hit_index
                     << "  hit_phi = " << hit_phi << std::endl;
                }
*/
                double hit_thx = atan2(hit_x->at(hit_index)-x0,hit_z->at(hit_index)-z0);
                double hit_thy = atan2(hit_y->at(hit_index)-y0,hit_z->at(hit_index)-z0);

                double delta_phi = hit_phi - track_phi;
                double delta_thx = hit_thx - track_thx;
                double delta_thy = hit_thy - track_thy;

                if (abs(delta_phi) < 0.010) {
                  sum_delta_phi_sq += delta_phi*delta_phi;
                  nHits++;
                }
/*
                if (jentry < 5) {
                  std::cout << " nHits = " << nHits << "  hit_phi = " << hit_phi
                       << "  delta_phi = " << delta_phi
                       << "  delta_phi_sq = " << delta_phi*delta_phi
                       << "  sum_delta_phi_sq = " << sum_delta_phi_sq << std::endl;
                 }
*/
              }  // end of loop over hits in particle
            }    // end of test for enough hits in particle
          }      // end of loop over particles in PV
          double ave_delta_phi_sq = sum_delta_phi_sq/nHits;
          double delta_phi_rms = sqrt(ave_delta_phi_sq);
/*
          if (jentry < 5) {
            std::cout << "ave_delta_phi_sq = " << ave_delta_phi_sq << "  delta_phi_rms = "
                 << delta_phi_rms << std::endl << std::endl;
          }
*/
          rmsDeltaPhi->Fill(delta_phi_rms);
        }        // end of test for enough particle in PV
       }        //  end of loop over PVs


    } //  end of per event loop (jentry loop)

  TCanvas c1;

  PrV_x->Draw();
  c1.SaveAs("PrV_x.png");

  PrV_y->Draw();
  c1.SaveAs("PrV_y.png");

  PrV_z->Draw();
  c1.SaveAs("PrV_z.png");

  PrV_z_4K->Draw();
  c1.SaveAs("PrV_z_4K.png");

  deltaPhiHist->Draw();
  c1.SaveAs("deltaPhiHist.png");

  deltaThxHist->Draw();
  c1.SaveAs("deltaThxHist.png");

  deltaThyHist->Draw();
  c1.SaveAs("deltaThyHist.png");

  deltaPhiHist_5GeV->Draw();
  c1.SaveAs("deltaPhiHist_5GeV.png");

  deltaThxHist_5GeV->Draw();
  c1.SaveAs("deltaThxHist_5GeV.png");

  deltaThyHist_5GeV->Draw();
  c1.SaveAs("deltaThyHist_5GeV.png");

  for (size_t shift_index=0; shift_index<shiftx_list.size(); shift_index++) {
   double shift_value = shiftx_list.at(shift_index)*1000.;  // from mm --> microns
   TString shiftStr;
   shiftStr.Form("%3.0f",shift_value);
   deltaPhiShiftedInX.at(shift_index)->Draw();
   c1.SaveAs("deltaPhiShiftInX_by_"+shiftStr+"_microns.png");

   deltaThxShiftedInX.at(shift_index)->Draw();
   c1.SaveAs("deltaThxShiftInX_by_"+shiftStr+"_microns.png");

   deltaThyShiftedInX.at(shift_index)->Draw();
   c1.SaveAs("deltaThyShiftInX_by_"+shiftStr+"_microns.png");
  }

   rmsDeltaPhi->Draw();
   c1.SaveAs("rmsDeltaPhi.png");
}
