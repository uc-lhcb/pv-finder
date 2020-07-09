#include "scatter.h"

TVector3 Scatter::smear(TVector3 p3, double fx0){
  double p = p3.Mag();
  double theta0=(13.6/(p*1000))*sqrt(fx0)*(1+0.038*log(fx0));

  double asmr = 0;
  if(gRandom->Uniform(0,1) < 0.94) asmr = abs(gRandom->Gaus(0,theta0/sqrt(2)));
  else{
    asmr = theta0 + pow(1 - (1-pow(theta0,-1.5))*gRandom->Uniform(0,1),-1/1.5);
  }
  asmr += abs(gRandom->Gaus(0,theta0/sqrt(2)));

  double azsmr = gRandom->Uniform(-3.14159,3.14159);

  TVector3 v(sin(asmr)*cos(azsmr),sin(asmr)*sin(azsmr),cos(asmr));
  double theta = p3.Theta() + cos(azsmr)*asmr;
  double phi = p3.Phi() + sin(azsmr)*asmr/sin(p3.Theta());
  p3.SetXYZ(p*sin(theta)*cos(phi),p*sin(theta)*sin(phi),p*cos(theta));

  return p3;
}
