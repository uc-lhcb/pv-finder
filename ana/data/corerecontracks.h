#pragma once

#include "ioclass.h"
#include <unordered_map>

template <class T> struct CoreReconTracksIO : public CoreIO<T, double> {
    using CoreIO<T, double>::CoreIO;
    using Vec = VecIO<T, double>;

    CoreReconTracksIO<T>(TTree *t, std::vector<std::string> const&& variables) : CoreIO<T, double>::CoreIO(t) {
        for (const auto& var : variables)
            track_variables.emplace(var,Vec{this,var});
    }

    void extend(std::vector<std::string> const&& variables){
        for (const auto& var : variables)
              track_variables.emplace(var,Vec{this,var});
    }

    Vec operator[](const std::string&& var) const {return track_variables[var];}

    private:
        mutable std::unordered_map<std::string,Vec> track_variables;
};

using CoreReconTracksIn = CoreReconTracksIO<In>;
using CoreReconTracksOut = CoreReconTracksIO<Out>;
