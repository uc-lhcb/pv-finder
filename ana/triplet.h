#ifndef TRIPLET_H
#define TRIPLET_H

#include "trajectory.h"
#include "hits.h"

#include <TSpline.h>

// more general than needed ATM
void getTrackPars(int nhits, Hit hits[], double &x0, double &y0, double &z0,
  double &tx, double &ty, double &chi2){
  // analytically solve for track parameters
  double sumZ=0, sumX=0, sumY=0, sumZX=0, sumZY=0, sumZ2=0;
  z0=100; // arbitrary (choose center of collision region)
  for(int i=0; i<nhits; i++){
    Hit hit = hits[i];
    double x=hit.point().x(),y=hit.point().y(),z=hit.point().z(),dz=z-z0;
    sumZ += dz;
    sumX += x;
    sumY += y;
    sumZX += dz*x;
    sumZY += dz*y;
    sumZ2 += dz*dz;
  }
  double xMean = sumX / nhits;
  double yMean = sumY / nhits;
  double zMean = sumZ / nhits;
  double denominator = sumZ2 - sumZ * zMean;
  tx = (sumZX - sumZ * xMean) / denominator;
  x0 = xMean - tx * zMean;
  ty = (sumZY - sumZ * yMean) / denominator;
  y0 = yMean - ty * zMean;

  chi2 = 0;
  if(nhits == 2) return;

  // TODO: assumes hits have constant resolution (could add this to Hit)
  double sig_hit = 0.012;
  for(int i=0; i<nhits; i++){
    double xx = tx*(hits[i].point().z()-z0)+x0;
    double yy = ty*(hits[i].point().z()-z0)+y0;
    chi2 += pow((hits[i].point().x()-xx)/sig_hit,2)
    +pow((hits[i].point().y()-yy)/sig_hit,2);
  }
}

// in principle, spline should be faster -- but it's not
// TODO: why is this so slow?
double getPDF(double d, double sigma){
  static TSpline3 *spl = 0;
  if(spl==0){
    double xspl[15]={-0.4,-0.3,-0.2,-0.1,-0.05,-0.02,-0.01,0,0.01,0.02,0.05,0.1,0.2,0.3,0.4};
    double yspl[15];
    yspl[0] = 1.55213;
    yspl[1] = 1.75992;
    yspl[2] = 10.411;
    yspl[3] = 93.1921;
    yspl[4] = 395.427;
    yspl[5] = 1013.88;
    yspl[6] = 1281.06;
    yspl[7] = 1344.67;
    yspl[8] = 1281.06;
    yspl[9] = 1013.88;
    yspl[10] = 395.427;
    yspl[11] = 93.192;
    yspl[12] = 10.4107;
    yspl[13] = 1.7599;
    yspl[14] = 1.55213;
    for(int i=0; i<15; i++) yspl[i] *= 200/23676.;
    spl = new TSpline3("spl_ip_pdf",xspl,yspl,15);
  }
  double val = spl->Eval(d*0.05/sigma)*0.05/sigma;
  if(val < 0) return 0;
  return val;
}

class Triplet {

private:
  Hit _hits[3];
  Trajectory _trajectory;
  double _chi2;
  Point _beam_poca;

public:

  Triplet():_chi2(0){}

  Triplet(const Hit &h0, const Hit &h1, const Hit &h2){
    _hits[0]=h0;
    _hits[1]=h1;
    _hits[2]=h2;
    double x,y,z,tx,ty;
    getTrackPars(3,_hits,x,y,z,tx,ty,_chi2);
    _trajectory = Trajectory(x,y,z,tx,ty);
    _beam_poca = _trajectory.beamPOCA(); // cache it
  }

  const Hit& hit(int h) const {return _hits[h];}

  const Point& beamPOCA() const {return _beam_poca;}

  const Trajectory& trajectory() const {return _trajectory;}

  double chi2NDof() const {return _chi2 / 3;}

  // projects trajectory to hit, checks residual
  double projectedHitChi2(const Hit &h) const {
    double x,y;
    _trajectory.getXY(h.point().z(),x,y);
    // TODO: sigma very approximate below
    double d2h = Point::distance(_hits[2].point(),h.point());
    double d21 = Point::distance(_hits[2].point(),_hits[1].point());
    double sigma = 0.012*(1+d2h/d21);
    double dx = (h.point().x()-x)/sigma, dy = (h.point().y()-y)/sigma;
    return dx*dx + dy*dy;
  }

  bool good() const {
    if(_hits[0].truePrt() != _hits[1].truePrt()) return false;
    if(_hits[0].truePrt() != _hits[2].truePrt()) return false;
    return true;
  }

  int truePrt() const {
    if(good()) return _hits[0].truePrt();
    else return -1;
  }

  double pdf(const Point &pv) const {
    double dx,dy;
    _trajectory.getIPxy(pv,dx,dy);
    if(abs(dx)>0.5 || abs(dy)>0.5) return 0;
    double sigma = 0.05;
    // TODO: improve sigma estimation, e.g. add distance to 1st hit
    if(_chi2/3 > 2) sigma += (_chi2-2)*0.05/4.;
    //if(abs(dx)/sigma > 5 || abs(dy)/sigma > 5) return 0;
    return TMath::Gaus(dx,0,sigma,true)*TMath::Gaus(dy,0,sigma,true);
    //return getPDF(dx,sigma)*getPDF(dy,sigma);
  }

  double deltaPhi(const TVector3 &v) const {
    // TODO: lazy here, using TVector3
    double sign = 1;
    if(_hits[0].point().z() > _hits[1].point().z()) sign=-1;
    TVector3 dir(sign*_trajectory.xslope(),sign*_trajectory.yslope(),sign);
    return abs(dir.DeltaPhi(v));
  }

  void print() const{
    for(int i=0; i<3; i++) cout << _hits[i].index() << "(" << _hits[i].truePrt() << ") ";
    cout << _chi2/3 << endl;
  }

};


#endif /* TRIPLET_H */
