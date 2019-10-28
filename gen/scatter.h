#ifndef SCATTER2_H
#define SCATTER2_H

#include <TF1.h>
#include <TLorentzVector.h>
#include <TRandom3.h>

using namespace std;

double FCN_SCATTER(double *x, double *p);

class Scatter {
    TF1 *fcn;

  public:
    Scatter() {}

    TVector3 smear(TVector3 p3, double fx0);
};

#endif // SCATTER2_H
