#pragma once

#include "hits.h"
#include "trajectory.h"
#include "triplet_utils.h"

#include <TSpline.h>

class TripletBase {
  protected:
    double chi2;
    Point _beam_poca;
    Trajectory _trajectory;

  public:
    TripletBase() { chi2 = 0; };

    TripletBase(double x, double y, double z, double tx, double ty, double chi2)
        : chi2(chi2), _trajectory(x, y, z, tx, ty) {
        _beam_poca = _trajectory.beamPOCA();
    }

    virtual ~TripletBase() = default;

    /// Given a point in space, calculate the PDF value
    double pdf(const Point &pv) const {
        double dx, dy;
        _trajectory.getIPxy(pv, dx, dy);

        if(abs(dx) > 0.5 || abs(dy) > 0.5)
            return 0;

        double sigma = 0.05;

        // TODO: improve sigma estimation, e.g. add distance to 1st hit
        if(chi2 / 3 > 2)
            sigma += (chi2 - 2) * 0.05 / 4.;

        // if(abs(dx)/sigma > 5 || abs(dy)/sigma > 5) return 0;

        return TMath::Gaus(dx, 0, sigma, true) * TMath::Gaus(dy, 0, sigma, true);
        // return getPDF(dx,sigma)*getPDF(dy,sigma);
    }

    const Point &beamPOCA() const { return _beam_poca; }

    const Trajectory &trajectory() const { return _trajectory; }

    double chi2NDof() const { return chi2 / 3; }

    double get_chi2() const { return chi2; }
    void set_chi2(double value) { chi2 = value; }
};

class TripletToy : public TripletBase {

  private:
    Hit _hits[3];

  public:
    TripletToy() : TripletBase() {}
    ~TripletToy() override = default;

    TripletToy(const Hit &h0, const Hit &h1, const Hit &h2) : TripletToy() {
        _hits[0] = h0;
        _hits[1] = h1;
        _hits[2] = h2;

        double x, y, z, tx, ty;
        getTrackPars(3, _hits, x, y, z, tx, ty, chi2);
        _trajectory = Trajectory(x, y, z, tx, ty);
        _beam_poca = _trajectory.beamPOCA();
    }

    const Hit &hit(int h) const { return _hits[h]; }

    // projects trajectory to hit, checks residual
    double projectedHitChi2(const Hit &h) const {
        double x, y;
        _trajectory.getXY(h.point().z(), x, y);
        // TODO: sigma very approximate below
        double d2h = Point::distance(_hits[2].point(), h.point());
        double d21 = Point::distance(_hits[2].point(), _hits[1].point());
        double sigma = 0.012 * (1 + d2h / d21);
        double dx = (h.point().x() - x) / sigma;
        double dy = (h.point().y() - y) / sigma;
        return dx * dx + dy * dy;
    }

    bool good() const {
        if(_hits[0].truePrt() != _hits[1].truePrt())
            return false;
        if(_hits[0].truePrt() != _hits[2].truePrt())
            return false;
        return true;
    }

    int truePrt() const {
        if(good())
            return _hits[0].truePrt();
        else
            return -1;
    }

    double deltaPhi(const TVector3 &v) const {
        // TODO: lazy here, using TVector3
        double sign = 1;
        if(_hits[0].point().z() > _hits[1].point().z())
            sign = -1;
        TVector3 dir(sign * _trajectory.xslope(), sign * _trajectory.yslope(), sign);
        return abs(dir.DeltaPhi(v));
    }

    void print() const {
        for(int i = 0; i < 3; i++)
            std::cout << _hits[i].index() << "(" << _hits[i].truePrt() << ") ";
        std::cout << chi2NDof() << std::endl;
    }
};
