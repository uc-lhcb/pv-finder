#ifndef UTILS_H
#define UTILS_H

#include "trajectory.h"
#include "triplet.h"
#include "tracks.h"

// for TMinuit
// TODO: would be faster with user-supplied derivatives
void FCN(Int_t &num_par,Double_t *grad,Double_t &f,Double_t *pars,Int_t iflag){
  Point pv(pars[0],pars[1],pars[2]);
  double sum1 = 0, sum2 = 0;
  Tracks* tracks = Tracks::instance();
  for(int i=tracks->tmin(); i<=tracks->tmax(); i++){
    double pdf = tracks->at(i).pdf(pv);
    sum1 += pdf;
    sum2 += pdf*pdf;
  }
  if(sum1 <1e-9) f=0;
  else f = -(sum1 - sum2/sum1);
}

// kernel value at point pv
double kernel(const Point &pv){
  if(!Tracks::instance()->run()) return 0;
  int num_par=3, iflag=0;
  double grad[3],f,pars[3];
  pars[0]=pv.x(); pars[1]=pv.y(); pars[2]=pv.z();
  FCN(num_par,grad,f,pars,iflag);
  return -f;
}

// gradient decent in x-y to find max kernel value for fixed pv.z()
// pv.x() and pv.y() are start values, then filled with x,y best fit
double kernelMax(Point &pv){

  Tracks *tracks = Tracks::instance();
  tracks->setRange(pv.z());
  if(!tracks->run()) return 0;

  double arglist[10],amin,edm,errdef;
  int iflag,nvpar,nparx,icstat;

  static TMinuit *min = 0;
  if(min == 0){
    min = new TMinuit();
    arglist[0] = -1;
    min->mnexcm("SET PRINT",arglist,1,iflag);
    min->mnexcm("SET NOW",arglist,0,iflag);
    min->SetFCN(FCN);
  }

  // reset
  min->mnrset(1);

  min->mnparm(0,"PVX",pv.x(),0.01,-10*0.055,10*0.055,iflag);
  min->mnparm(1,"PVY",pv.y(),0.01,-10*0.055,10*0.055,iflag);
  min->mnparm(2,"PVZ",pv.z(),0,0,0,iflag);

  arglist[0] = 1000; arglist[1] = 0.1;
  min->mnexcm("MIGRAD",arglist,2,iflag);

  double x,y,tmp;
  min->GetParameter(0,x,tmp);
  min->GetParameter(1,y,tmp);
  min->mnstat(amin,edm,errdef,nvpar,nparx,iflag);

  pv.set(x,y,pv.z());
  return -amin;
}

// -1: < 2 particles made hits
// 0: < 5 long tracks
// 1: LHCb pv
int pvCategory(Data &data, int i){
  if(data.ntrks->at(i)<2) return -1;
  int ntrk_in_acc = 0, nprt = data.ipv->size();
  for(int j=0; j<nprt; j++){
    if(data.ipv->at(j) != i) continue;
    if(data.nhits->at(j) < 3) continue;
    if(abs(data.z->at(j)-data.pvz->at(i))>0.001) continue;
    TVector3 p3(data.px->at(j),data.py->at(j),data.pz->at(j));
    if(p3.Eta() < 2 || p3.Eta() > 5) continue;
    if(p3.Mag() < 3) continue;
    ntrk_in_acc++;
  }
  if(ntrk_in_acc < 5) return 0;
  else return 1;
}

// -1: no particles made hits
// 0: 1 particle with hits
// 1: 2+ (an actual SV)
int svCategory(Data &data, int i){
  int nsv_prt = 0, nprt = data.ipv->size();
  for(int j=0; j<nprt; j++){
    if(data.nhits->at(j) < 3) continue;
    if(abs(data.z->at(j)-data.svz->at(i))>0.001) continue;
    nsv_prt++;
  }
  if(nsv_prt < 1) return -1;
  if(nsv_prt < 2) return 0;
  else return 1;
}

#endif /* UTILS_H */
