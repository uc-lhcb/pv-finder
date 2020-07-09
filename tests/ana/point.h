#pragma once

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