#pragma once
// bg_interp.hpp
// Linear interpolation of background arrays for use in the MS solver.
// Provides interpolated values of all BgState fields at arbitrary N.

#include <vector>
#include <cmath>
#include <stdexcept>
#include "background.hpp"

namespace nmdc {

// ---------------------------------------------------------------
// Interpolate a single field from the background table at e-fold N.
// Uses linear interpolation between stored steps.
// ---------------------------------------------------------------
struct BgInterp {
    const std::vector<BgState>& bg;

    explicit BgInterp(const std::vector<BgState>& bg_) : bg(bg_) {
        if (bg.empty())
            throw std::runtime_error("BgInterp: empty background table");
    }

    // Find bracketing index for given N (binary search)
    int find_index(double N) const {
        int lo = 0;
        int hi = static_cast<int>(bg.size()) - 1;
        if (N <= bg[lo].N) return lo;
        if (N >= bg[hi].N) return hi - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (bg[mid].N <= N) lo = mid;
            else                hi = mid;
        }
        return lo;
    }

    // Linear interpolation weight
    double interp(double N, double (BgState::*field)) const {
        int i = find_index(N);
        int j = i + 1;
        if (j >= static_cast<int>(bg.size())) j = i;
        double dN  = bg[j].N - bg[i].N;
        double t   = (dN > 0.0) ? (N - bg[i].N) / dN : 0.0;
        return (1.0 - t) * (bg[i].*field) + t * (bg[j].*field);
    }

    double phi   (double N) const { return interp(N, &BgState::phi);   }
    double dphi  (double N) const { return interp(N, &BgState::dphi);  }
    double ddphi (double N) const { return interp(N, &BgState::ddphi); }
    double H     (double N) const { return interp(N, &BgState::H);     }
    double Hprime(double N) const { return interp(N, &BgState::Hprime);}
    double eps_H (double N) const { return interp(N, &BgState::eps_H); }
    double FT    (double N) const { return interp(N, &BgState::FT);    }
    double GT    (double N) const { return interp(N, &BgState::GT);    }
    double cT2   (double N) const { return interp(N, &BgState::cT2);   }
    double FS    (double N) const { return interp(N, &BgState::FS);    }
    double GS    (double N) const { return interp(N, &BgState::GS);    }
    double cS2   (double N) const { return interp(N, &BgState::cS2);   }
    double lnzS  (double N) const { return interp(N, &BgState::lnzS);  }
    double lnzT  (double N) const { return interp(N, &BgState::lnzT);  }

    // Numerical derivative of lnz using central differences
    // Returns d(lnz)/dN at N
    double dlnzS_dN(double N) const {
        double dh = DN * 2.0;
        double Nlo = N - dh, Nhi = N + dh;
        // clamp to table range
        if (Nlo < bg.front().N) Nlo = bg.front().N;
        if (Nhi > bg.back().N)  Nhi = bg.back().N;
        return (lnzS(Nhi) - lnzS(Nlo)) / (Nhi - Nlo);
    }
    double dlnzT_dN(double N) const {
        double dh = DN * 2.0;
        double Nlo = N - dh, Nhi = N + dh;
        if (Nlo < bg.front().N) Nlo = bg.front().N;
        if (Nhi > bg.back().N)  Nhi = bg.back().N;
        return (lnzT(Nhi) - lnzT(Nlo)) / (Nhi - Nlo);
    }

    // Second derivative of lnz
    double d2lnzS_dN2(double N) const {
        double dh = DN * 2.0;
        double Nlo = N - dh, Nhi = N + dh;
        if (Nlo < bg.front().N) Nlo = bg.front().N + dh;
        if (Nhi > bg.back().N)  Nhi = bg.back().N  - dh;
        return (lnzS(Nhi) - 2.0*lnzS(N) + lnzS(Nlo)) / (dh * dh);
    }
    double d2lnzT_dN2(double N) const {
        double dh = DN * 2.0;
        double Nlo = N - dh, Nhi = N + dh;
        if (Nlo < bg.front().N) Nlo = bg.front().N + dh;
        if (Nhi > bg.back().N)  Nhi = bg.back().N  - dh;
        return (lnzT(Nhi) - 2.0*lnzT(N) + lnzT(Nlo)) / (dh * dh);
    }

    // Pump field z''/z in conformal time divided by (aH)^2:
    //   (z''/z)|_tau / (aH)^2 = d2lnz/dN2 + (dlnz/dN)^2 + (2-eps)*dlnz/dN
    // This quantity is O(1) and has no overflow.
    // In ms_rhs the pump appears as pump/(aH)^2 so we return this directly.
    double pump_scalar_over_aH2(double N) const {
        double d1  = dlnzS_dN(N);
        double d2  = d2lnzS_dN2(N);
        double eps = eps_H(N);
        return d2 + d1*d1 + (2.0 - eps)*d1;
    }
    double pump_tensor_over_aH2(double N) const {
        double d1  = dlnzT_dN(N);
        double d2  = d2lnzT_dN2(N);
        double eps = eps_H(N);
        return d2 + d1*d1 + (2.0 - eps)*d1;
    }

    // Range helpers
    double N_min() const { return bg.front().N; }
    double N_max() const { return bg.back().N;  }
};

} // namespace nmdc
