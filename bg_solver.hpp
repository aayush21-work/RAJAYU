#pragma once
// bg_solver.hpp
// Integrates the background ODE system in e-fold time N.
//
// State vector: y = {phi, dphi, H}
//
// Equations (exact, Mpl=1):
//   d phi  / dN = dphi
//   d dphi / dN = ddphi   (from KG, computed self-consistently with eps_H)
//   d H    / dN = -eps_H * H
//
// Integration: fixed-step RK4.
// Stop condition: eps_H >= 1  (end of inflation).

#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "background.hpp"
#include "constants.hpp"

namespace nmdc {

using State = std::array<double, 3>;  // {phi, dphi, H}

// ---------------------------------------------------------------
// RHS of the ODE system given current state.
// Returns {dphi, ddphi, Hprime}.
// ---------------------------------------------------------------
inline State rhs(double /*N*/, const State& y) {
    double phi  = y[0];
    double dphi = y[1];
    double H    = y[2];

    // Solve H^2 from Friedmann (quadratic) — use the stored H as check,
    // but trust the passed H as it comes from the integrator
    double eps_H, ddphi;
    solve_eps_ddphi(phi, dphi, H, eps_H, ddphi);

    return { dphi, ddphi, -eps_H * H };
}

// ---------------------------------------------------------------
// Single RK4 step.
// ---------------------------------------------------------------
inline State rk4_step(double N, const State& y, double dN) {
    State k1 = rhs(N,          y);
    State k2, k3, k4;

    // k2
    State y2 = { y[0] + 0.5*dN*k1[0],
                 y[1] + 0.5*dN*k1[1],
                 y[2] + 0.5*dN*k1[2] };
    // Re-solve H from Friedmann at the mid-point to maintain constraint
    y2[2] = std::sqrt(solve_H2(y2[0], y2[1]));
    k2 = rhs(N + 0.5*dN, y2);

    // k3
    State y3 = { y[0] + 0.5*dN*k2[0],
                 y[1] + 0.5*dN*k2[1],
                 y[2] + 0.5*dN*k2[2] };
    y3[2] = std::sqrt(solve_H2(y3[0], y3[1]));
    k3 = rhs(N + 0.5*dN, y3);

    // k4
    State y4 = { y[0] + dN*k3[0],
                 y[1] + dN*k3[1],
                 y[2] + dN*k3[2] };
    y4[2] = std::sqrt(solve_H2(y4[0], y4[1]));
    k4 = rhs(N + dN, y4);

    State out;
    for (int i = 0; i < 3; ++i)
        out[i] = y[i] + (dN/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

    // Project H back onto Friedmann constraint after each step
    out[2] = std::sqrt(solve_H2(out[0], out[1]));

    return out;
}

// ---------------------------------------------------------------
// Run the background solver.
// Returns vector of BgState at each N-step.
// Optionally writes to file.
// ---------------------------------------------------------------
inline std::vector<BgState> run_background(
        double phi0, double dphi0,
        bool write_file = true)
{
    // Initial H from Friedmann
    double H0 = std::sqrt(solve_H2(phi0, dphi0));

    State y = { phi0, dphi0, H0 };

    std::vector<BgState> bg;
    bg.reserve(static_cast<size_t>((N_END - N_INI) / DN) + 10);

    std::ofstream fout;
    if (write_file) {
        fout.open(BG_FILE);
        if (!fout.is_open())
            throw std::runtime_error("Cannot open background output file");
        // Header
        fout << "# N  phi  dphi  ddphi  H  Hprime  eps_H  X  "
             << "FT  GT  cT2  Sigma  Theta  FS  GS  cS2  lnzS  lnzT\n";
        fout.precision(15);
    }

    double N = N_INI;

    while (N <= N_END) {
        // Build full state
        BgState s = make_bg_state(N, y[0], y[1], y[2]);
        bg.push_back(s);

        if (write_file) {
            fout << s.N      << " "
                 << s.phi    << " "
                 << s.dphi   << " "
                 << s.ddphi  << " "
                 << s.H      << " "
                 << s.Hprime << " "
                 << s.eps_H  << " "
                 << s.X      << " "
                 << s.FT     << " "
                 << s.GT     << " "
                 << s.cT2    << " "
                 << s.Sigma  << " "
                 << s.Theta  << " "
                 << s.FS     << " "
                 << s.GS     << " "
                 << s.cS2    << " "
                 << s.lnzS   << " "
                 << s.lnzT   << "\n";
        }

        // Stop conditions
        if (s.eps_H >= 1.0) {
            std::cout << "Inflation ended at N = " << N
                      << "  eps_H = " << s.eps_H << "\n";
            break;
        }

        // Guard: dphi^2/2 approaching 3 means kinetic domination imminent
        // Friedmann quadratic loses positive root when dphi^2 > 6*(1+9kap*H^2)/(1+...)
        // Safe cutoff: stop when dphi^2 > 5.5 (leaves margin before sqrt(6)^2=6)
        if (s.dphi * s.dphi > 5.5) {
            std::cout << "Kinetic domination at N = " << N
                      << "  dphi = " << s.dphi << "  (ending integration)\n";
            break;
        }

        // Stability check — only after transient settles (N > 2)
        if (N > 2.0) {
            if (s.GT <= 0.0)
                throw std::runtime_error("GT <= 0: tensor gradient instability");
            if (s.GS <= 0.0)
                throw std::runtime_error("GS <= 0: scalar gradient instability");
            if (s.cS2 <= 0.0)
                throw std::runtime_error("cS2 <= 0: imaginary scalar sound speed");
        }

        // RK4 step — catch kinetic domination gracefully
        try {
            y = rk4_step(N, y, DN);
        } catch (const std::runtime_error& e) {
            std::cout << "Integration stopped at N = " << N
                      << ": " << e.what() << "\n";
            break;
        }
        N += DN;
    }

    if (bg.empty())
        throw std::runtime_error("Background solver produced no output");

    std::cout << "Background: " << bg.size() << " steps stored.\n";
    return bg;
}

} // namespace nmdc
