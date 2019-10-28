#pragma once

#include <TSpline.h>

#include "hit.h"

// more general than needed ATM
inline void
getTrackPars(int nhits, Hit hits[], double &x0, double &y0, double &z0, double &tx, double &ty, double &chi2) {

    // analytically solve for track parameters
    double sumZ = 0, sumX = 0, sumY = 0, sumZX = 0, sumZY = 0, sumZ2 = 0;
    z0 = 100; // arbitrary (choose center of collision region)
    for(int i = 0; i < nhits; i++) {
        Hit hit = hits[i];
        double x = hit.point().x();
        double y = hit.point().y();
        double z = hit.point().z();
        double dz = z - z0;
        sumZ += dz;
        sumX += x;
        sumY += y;
        sumZX += dz * x;
        sumZY += dz * y;
        sumZ2 += dz * dz;
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
    if(nhits == 2)
        return;

    // TODO: assumes hits have constant resolution (could add this to Hit)
    double sig_hit = 0.012;
    for(int i = 0; i < nhits; i++) {
        double xx = tx * (hits[i].point().z() - z0) + x0;
        double yy = ty * (hits[i].point().z() - z0) + y0;
        chi2 += pow((hits[i].point().x() - xx) / sig_hit, 2) + pow((hits[i].point().y() - yy) / sig_hit, 2);
    }
}

// in principle, spline should be faster -- but it's not
/* TODO: why is this so slow?
double getPDF(double d, double sigma){
  static TSpline3 *spl = nullptr;
  if(spl==nullptr){
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
    for(int i=0; i<15; i++)
        yspl[i] *= 200/23676.;
    spl = new TSpline3("spl_ip_pdf",xspl,yspl,15);
  }
  double val = spl->Eval(d*0.05/sigma)*0.05/sigma;
  if(val < 0)
      return 0;
  return val;
}
*/
