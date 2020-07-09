#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include "TVector3.h"

using namespace std;

class Point {

  protected:
    double _x, _y, _z;

  public:
    Point() : _x(0), _y(0), _z(0) {}
    Point(double x, double y, double z) : _x(x), _y(y), _z(z) {}

    double x() const { return _x; }
    double y() const { return _y; }
    double z() const { return _z; }

    double perp() const { return sqrt(_x * _x + _y * _y); }

    void set(double xx, double yy, double zz) {
        _x = xx;
        _y = yy;
        _z = zz;
    }

    static double distance(const Point &p1, const Point &p2) {
        double dx = p1._x - p2._x;
        double dy = p1._y - p2._y;
        double dz = p1._z - p2._z;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }
};

class Trajectory {

  protected:
    Point _point;
    double _tx, _ty;

  public:
    Trajectory() : _point(), _tx(0), _ty(0) {}

    Trajectory(double x, double y, double z, double tx, double ty) : _point(x, y, z), _tx(tx), _ty(ty) {}

    Trajectory(const Point &p, double tx, double ty) : _point(p), _tx(tx), _ty(ty) {}

    Trajectory(const Point &p1, const Point &p2) {
        _point.set(p1.x(), p1.y(), p1.z());
        double dz = p2.z() - p1.z();
        if(abs(dz) < 1e-3) { // requires hits in same module
            _tx = 666;
            _ty = 666;
        } else {
            _tx = (p2.x() - p1.x()) / dz;
            _ty = (p2.y() - p1.y()) / dz;
        }
    }

    const Point &point() const { return _point; }

    double xslope() const { return _tx; }
    double yslope() const { return _ty; }

    void transport(double zval) {
        double x = _point.x() + (zval - _point.z()) * _tx;
        double y = _point.y() + (zval - _point.z()) * _ty;
        _point.set(x, y, zval);
    }

    void getXY(double z, double &x, double &y) const {
        x = _tx * (z - _point.z()) + _point.x();
        y = _ty * (z - _point.z()) + _point.y();
    }

    void getIPxy(const Point &pv, double &ipx, double &ipy) const {
        ipx = _point.x() + (pv.z() - _point.z()) * _tx - pv.x();
        ipy = _point.y() + (pv.z() - _point.z()) * _ty - pv.y();
    }

    double getIP(const Point &pv, double &ipx, double &ipy) const {
        getIPxy(pv, ipx, ipy);
        return sqrt((ipx * ipx + ipy * ipy) / (1 + _tx * _tx + _ty * _ty));
    }

    bool goodVeloSlopes() const {
        if(abs(_tx) > 0.4 || abs(_ty) > 0.3)
            return false;
        else
            return true;
    }

    // POCA (point of closest approach) is on trajectory t1
    static Point poca(const Trajectory &t1, const Trajectory &t2) {
        // TODO: replace TVector3 usage with something more lightweight
        TVector3 p0(t1._point.x(), t1._point.y(), t1._point.z());
        TVector3 q0(t2._point.x(), t2._point.y(), t2._point.z());
        TVector3 u(t1._tx, t1._ty, 1), v(t2._tx, t2._ty, 1);
        TVector3 w0 = p0 - q0;
        double a = u.Mag2();
        double b = u.Dot(v);
        double c = v.Mag2();
        double d = u.Dot(w0);
        double e = v.Dot(w0);
        double D = a * c - b * b;
        double x = (b * e - c * d) / D;
        u.SetXYZ(x * u.X(), x * u.Y(), x * u.Z());
        TVector3 p = p0 + u;
        return Point(p.X(), p.Y(), p.Z());
    }

    Point beamPOCA() const {
        Trajectory beamline(0, 0, 0, 0, 0);
        return Trajectory::poca(*this, beamline);
    }
};

#endif /* TRAJECTORY_H */
