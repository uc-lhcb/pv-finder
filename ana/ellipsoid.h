#pragma once

#include "trajectory.h"

class Ellipsoid {

  private:
    TVector3 _center, _minor_axis1, _minor_axis2, _major_axis;
    double   _zmin, _zmax, _determinant;

  public:
    Ellipsoid() : _center(), _minor_axis1(), _minor_axis2(), _major_axis(), _zmin(0.), _zmax(0.), _determinant(0.) {}

    Ellipsoid(const Trajectory& t1, const Trajectory& t2, const double& xy_error, const double& z_error) { // EMK
      // road error 0.0566
      //##  the error ellipsoid of the point of closest approach of L2
      //##  [the doca vector]
      //##  assuming the uncertainty tranverse to the direction of L2 is
      //##  road_error in each direction
        
      auto const v1      = TVector3(t1.xslope(),t1.yslope(),1.);
      auto const v2      = TVector3(t2.xslope(),t2.yslope(),1.);
      auto const L1_poca = Trajectory::poca(t1,t2).to_vec();
      auto const L2_poca = Trajectory::poca(t2,t1).to_vec();
      _center            = L2_poca;
      auto const v3      = L2_poca - L1_poca;
      auto const doca    = std::sqrt(v3.Dot(v3));
      auto const yhat    = 1/doca*v3; // EMK (I switched around variable names)
      auto const v2_mag  = std::sqrt(v2.Dot(v2));
      auto const zhat    = 1/v2_mag*v2; // EMK
      auto const xhat    = yhat.Cross(zhat); // EMK
      _minor_axis1       = xy_error*yhat; // EMK (before road_error*zhat)
      _minor_axis2       = xy_error*xhat; // EMK (before road_error*yhat)
      _major_axis        = z_error*zhat; // EMK (before (road_error/std::tan(v1.Angle(v2)))*xhat)
      _zmin              = _center.z()-3*_major_axis.z();
      _zmax              = _center.z()+3*_major_axis.z();
      _determinant       = xy_error*xy_error*z_error/std::tan(v1.Angle(v2));
    }

    double pdf(Point const& scan_point) const {
      // ## this function takes the positions p1 and p2  plus the minor axes m1 and m2 and
      // ## the major axis of the error ellipsoid associated with p2 as the inputs. It
      // ## returns the chi-square value  of the distance-of-closest-approach and the
      // ## associated probability value exp( -1/2 chisq) [ignoring (2*pi)^{3/2}]
      auto const xvec = _center-scan_point.to_vec();
      auto const u1 = _minor_axis1.Unit();
      auto const chisq = std::pow(xvec.Dot(_minor_axis1.Unit()),2)/_minor_axis1.Mag2() +
                         std::pow(xvec.Dot(_minor_axis2.Unit()),2)/_minor_axis2.Mag2() +
                         std::pow(xvec.Dot(_major_axis.Unit()),2)/_major_axis.Mag2();

      return std::exp(-0.5*chisq)/std::sqrt(_determinant);
    }
    double zmin() const {return _zmin;}
    double zmax() const {return _zmax;}
    void print() const {
      printf("POCA ellipsoid Info:   center (%+6.3f,%+6.3f,%+6.3f)  zmin %+6.3f  zmax %+6.3f\n",_center.x(),_center.y(),_center.z(),_zmin,_zmax);
    }
    TVector3 minor_axis1() const {return _minor_axis1;}
    TVector3 minor_axis2() const {return _minor_axis2;}
    TVector3 major_axis () const {return _major_axis;}
    TVector3 center     () const {return _center;}
};
