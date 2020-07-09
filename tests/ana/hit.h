#pragma once

#include "point.h"

class Hit {

  private:
    Point _point;
    int _idx, _true_prt;

  public:
    Hit() : _point(), _idx(-1), _true_prt(-1) {}

    Hit(int idx, double x, double y, double z, int true_prt) : _point(x, y, z), _idx(idx), _true_prt(true_prt) {}

    const Point &point() const { return _point; }
    int index() const { return _idx; }
    int truePrt() const { return _true_prt; }

    // TODO: add module indices to data file and to this class
    static bool sameModule(const Hit &hit1, const Hit &hit2) { return abs(hit1._point.z() - hit2._point.z()) < 1; }
};
