#pragma once
// potential.hpp
// Inflaton potential V(phi) and its derivative.
// Change this file to switch between potential models.
// Units: Mpl = 1.

#include <cmath>
#include "constants.hpp"

namespace nmdc {

// V(phi) = lambda * phi^n / n  (large-field power law)
inline double V(double phi) {
    return LAMBDA * std::pow(std::abs(phi), N_POW) / N_POW;
}

// dV/dphi
inline double V_phi(double phi) {
    // d/dphi [ lambda |phi|^n / n ] = lambda * sign(phi) * |phi|^(n-1)
    if (phi > 0.0)
        return LAMBDA * std::pow(phi,  N_POW - 1.0);
    else if (phi < 0.0)
        return LAMBDA * std::pow(-phi, N_POW - 1.0) * (-1.0);
    else
        return 0.0;
}

} // namespace nmdc
