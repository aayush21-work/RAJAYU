#pragma once
// ms_solver.hpp
// Solves the Mukhanov-Sasaki equation for a single mode given ln(k).
//
// OVERFLOW SAFETY: All comparisons use ln(k/aH) = ln_k - N - ln(H),
// which is O(1) near horizon crossing regardless of total e-folds.
//
// MS equation in N-time:
//   u'' + (1-eps)*u' + [cs^2*(k/aH)^2 - pump/(aH)^2]*u = 0
//
// where pump = z''/z in conformal time (computed from bg_interp).
//
// Bunch-Davies ICs when ln(k*cs/aH) = ln(BD_FACTOR):
//   u  = exp(-0.5*(ln_k + ln(cs)))  [= 1/sqrt(2*k*cs)]
//   u' = -i * exp(ln_k + ln(cs) - ln_aH) * u
//
// Freeze-out when ln(k*cs/aH) < ln(FO_FACTOR).
// Power spectrum: k^3/(2pi^2) * |u|^2/z^2  using stored lnz.

#include <array>
#include <cmath>
#include <stdexcept>
#include "bg_interp.hpp"
#include "constants.hpp"

namespace nmdc {

using MSState = std::array<double, 4>;  // {Re u, Im u, Re u', Im u'}
enum class Sector { SCALAR, TENSOR };

// ---------------------------------------------------------------
// ln(aH) from interpolated background at e-fold N.
// Safe: just N + ln(H), no exp.
// ---------------------------------------------------------------
inline double ln_aH(double N, const BgInterp& bg) {
    return N + std::log(bg.H(N));
}

// ---------------------------------------------------------------
// RHS of MS ODE.
// Uses ratio (k/aH)^2 = exp(2*(ln_k - ln_aH)) — always finite.
// ---------------------------------------------------------------
inline MSState ms_rhs(double N, const MSState& y,
                      double ln_k, Sector sec,
                      const BgInterp& bg) {
    double eps     = bg.eps_H(N);
    double ln_aH_N = ln_aH(N, bg);

    // (k/aH)^2 = exp(2*(ln_k - ln_aH)) — always finite near crossing
    double ln_ratio = ln_k - ln_aH_N;
    double ratio2   = std::exp(2.0 * ln_ratio);

    double cs2, pump_over_aH2;
    if (sec == Sector::SCALAR) {
        cs2           = bg.cS2(N);
        pump_over_aH2 = bg.pump_scalar_over_aH2(N);
    } else {
        cs2           = bg.cT2(N);
        pump_over_aH2 = bg.pump_tensor_over_aH2(N);
    }

    // MS coefficient: cs^2*(k/aH)^2 - z''/z/(aH)^2
    // Both terms are O(1) — no overflow.
    double coeff_u  = cs2 * ratio2 - pump_over_aH2;
    double coeff_up = -(1.0 - eps);

    MSState d;
    d[0] = y[2];
    d[1] = y[3];
    d[2] = coeff_up * y[2] - coeff_u * y[0];
    d[3] = coeff_up * y[3] - coeff_u * y[1];
    return d;
}

// ---------------------------------------------------------------
// RK4 step for MS equation.
// ---------------------------------------------------------------
inline MSState ms_rk4(double N, const MSState& y, double dN,
                      double ln_k, Sector sec, const BgInterp& bg) {
    auto k1 = ms_rhs(N,           y,  ln_k, sec, bg);
    MSState y2, y3, y4;
    for (int i=0;i<4;++i) y2[i] = y[i]+0.5*dN*k1[i];
    auto k2 = ms_rhs(N+0.5*dN,   y2, ln_k, sec, bg);
    for (int i=0;i<4;++i) y3[i] = y[i]+0.5*dN*k2[i];
    auto k3 = ms_rhs(N+0.5*dN,   y3, ln_k, sec, bg);
    for (int i=0;i<4;++i) y4[i] = y[i]+dN*k3[i];
    auto k4 = ms_rhs(N+dN,        y4, ln_k, sec, bg);
    MSState out;
    for (int i=0;i<4;++i)
        out[i] = y[i] + (dN/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
    return out;
}

// ---------------------------------------------------------------
// Find N_ini: e-fold when ln(k*cs) = ln(BD_FACTOR) + ln(aH).
// i.e. ln_k + ln(cs) - ln(aH) = ln(BD_FACTOR)
// Bisect on f(N) = ln_k + ln(cs(N)) - ln_aH(N) - ln(BD_FACTOR)
// f > 0 early (sub-Hubble), f < 0 late (super-Hubble).
// ---------------------------------------------------------------
inline double find_N_ini(double ln_k, Sector sec, const BgInterp& bg) {
    double ln_bd = std::log(BD_FACTOR);

    auto f = [&](double N) -> double {
        double cs = (sec == Sector::SCALAR)
                    ? std::sqrt(std::abs(bg.cS2(N)))
                    : std::sqrt(std::abs(bg.cT2(N)));
        return ln_k + std::log(cs) - ln_aH(N, bg) - ln_bd;
    };

    double Nlo = bg.N_min();
    double Nhi = bg.N_max();

    if (f(Nlo) < 0.0) return Nlo;  // already super-Hubble at start
    if (f(Nhi) > 0.0) return Nlo;  // never crosses — use earliest point

    for (int iter = 0; iter < 100; ++iter) {
        double Nmid = 0.5*(Nlo+Nhi);
        double fmid = f(Nmid);
        if (std::abs(fmid) < 1.0e-8) return Nmid;
        if (fmid > 0.0) Nlo = Nmid;
        else            Nhi = Nmid;
    }
    return 0.5*(Nlo+Nhi);
}

// ---------------------------------------------------------------
// Power spectrum amplitude |u|^2/z^2 at current N.
// Uses stored lnz to avoid overflow.
// ---------------------------------------------------------------
inline double power_amplitude(const MSState& y, double N,
                               Sector sec, const BgInterp& bg) {
    double u2  = y[0]*y[0] + y[1]*y[1];
    double lnz = (sec == Sector::SCALAR) ? bg.lnzS(N) : bg.lnzT(N);
    // |u/z|^2 = u2 * exp(-2*lnz)
    return u2 * std::exp(-2.0 * lnz);
}

// ---------------------------------------------------------------
// solve_mode_ln: returns ln(P(k)) instead of P(k).
// Safe for arbitrarily large ln_k — no overflow.
// ln P = 3*ln_k - ln(2*pi^2) + ln(|u/z|^2)
//      = 3*ln_k - ln(2*pi^2) + 2*ln_amp + ln(u_norm^2) - 2*lnz
// All terms individually may be large but the physical combination
// is always O(-20) for inflation.
// ---------------------------------------------------------------
inline double solve_mode_ln(double ln_k, Sector sec, const BgInterp& bg) {
    double N_start = find_N_ini(ln_k, sec, bg);
    double N_stop  = bg.N_max();

    double cs_ini = (sec == Sector::SCALAR)
                    ? std::sqrt(std::abs(bg.cS2(N_start)))
                    : std::sqrt(std::abs(bg.cT2(N_start)));
    double ln_aH_ini = ln_aH(N_start, bg);

    // BD: u_phys = exp(ln_amp) * u_norm, u_norm starts at (1,0)
    double ln_amp   = -0.5 * (ln_k + std::log(cs_ini));
    double kcs_aH   = std::exp(ln_k + std::log(cs_ini) - ln_aH_ini);

    MSState y = { 1.0, 0.0, 0.0, -kcs_aH };

    double dN = DN;
    double N  = N_start;
    double ln_fo = std::log(FO_FACTOR);

    // Track convergence in log space
    double lnP_prev = -1e300, lnP_curr = -1e300;
    int frozen_steps = 0;
    int step = 0;
    constexpr int RENORM = 10;

    while (N < N_stop) {
        // Renormalise u_norm periodically
        if (step % RENORM == 0 && step > 0) {
            double norm = std::sqrt(y[0]*y[0] + y[1]*y[1]);
            if (norm > 0.0 && std::isfinite(norm)) {
                ln_amp += std::log(norm);
                y[0] /= norm; y[1] /= norm;
                y[2] /= norm; y[3] /= norm;
            }
        }
        ++step;

        // Freeze-out check
        double cs_now = (sec == Sector::SCALAR)
                        ? std::sqrt(std::abs(bg.cS2(N)))
                        : std::sqrt(std::abs(bg.cT2(N)));
        double ln_kcs_aH = ln_k + std::log(cs_now) - ln_aH(N, bg);

        if (ln_kcs_aH < ln_fo) {
            double lnz = (sec==Sector::SCALAR) ? bg.lnzS(N) : bg.lnzT(N);
            double u2  = y[0]*y[0] + y[1]*y[1];
            lnP_curr = 2.0*ln_amp + std::log(u2) - 2.0*lnz;

            double rel = std::abs(lnP_curr - lnP_prev)
                         / (std::abs(lnP_prev) + 1e-10);
            if (lnP_prev > -1e299 && rel < PS_CONV)
                { if (++frozen_steps >= 5) break; }
            else frozen_steps = 0;
            lnP_prev = lnP_curr;
        }

        y  = ms_rk4(N, y, dN, ln_k, sec, bg);
        N += dN;
    }

    // Final read-off
    {
        double lnz = (sec==Sector::SCALAR) ? bg.lnzS(N-dN) : bg.lnzT(N-dN);
        double u2  = y[0]*y[0] + y[1]*y[1];
        lnP_curr = 2.0*ln_amp + std::log(u2) - 2.0*lnz;
    }

    // ln P = 3*ln_k - ln(2pi^2) + ln(|u/z|^2)
    double ln_k3_over_2pi2 = 3.0*ln_k - std::log(2.0*M_PI*M_PI);
    double lnP = ln_k3_over_2pi2 + lnP_curr;

    // Tensor: extra factor 2
    if (sec == Sector::TENSOR) lnP += std::log(2.0);

    return lnP;
}
// We instead normalise u so |u| = 1 at IC, track log-amplitude separately,
// and only combine at the end:
//   u_phys = u_norm * exp(ln_amp)
//   |u_phys/z|^2 = exp(2*ln_amp + ln(u_norm^2) - 2*lnz)
// ---------------------------------------------------------------
inline double solve_mode(double ln_k, Sector sec, const BgInterp& bg) {
    double N_start = find_N_ini(ln_k, sec, bg);
    double N_stop  = bg.N_max();

    double cs_ini = (sec == Sector::SCALAR)
                    ? std::sqrt(std::abs(bg.cS2(N_start)))
                    : std::sqrt(std::abs(bg.cT2(N_start)));
    double ln_aH_ini = ln_aH(N_start, bg);

    // Physical BD ICs:
    //   u_phys  = exp(-0.5*(ln_k + ln(cs)))
    //   u'_phys = -i * exp(ln_k + ln(cs) - ln_aH) * u_phys
    // Normalise: u_norm = u_phys / |u_phys| = 1 (real), track ln_amp
    double ln_amp   = -0.5 * (ln_k + std::log(cs_ini));  // ln|u_phys|
    double kcs_aH   = std::exp(ln_k + std::log(cs_ini) - ln_aH_ini);

    // u_norm starts at (1, 0), u'_norm = (0, -kcs_aH) * exp(0) = (0, -kcs_aH)
    // because u'_phys = -i*kcs_aH * u_phys, so u'_norm = -i*kcs_aH
    MSState y = { 1.0, 0.0, 0.0, -kcs_aH };

    double dN     = DN;
    double N      = N_start;

    // Renormalise every ~10 steps to keep |y| ~ 1
    constexpr int RENORM_STEPS = 10;

    double ln_fo = std::log(FO_FACTOR);
    double P_prev = -1.0, P_curr = -1.0;
    int frozen_steps = 0;

    int step = 0;
    while (N < N_stop) {
        // Renormalise to prevent underflow/overflow of u_norm
        if (step % RENORM_STEPS == 0 && step > 0) {
            double norm = std::sqrt(y[0]*y[0] + y[1]*y[1]);
            if (norm > 0.0 && std::isfinite(norm)) {
                ln_amp += std::log(norm);
                y[0] /= norm; y[1] /= norm;
                y[2] /= norm; y[3] /= norm;
            }
        }
        ++step;

        // Freeze-out check
        double cs_now = (sec == Sector::SCALAR)
                        ? std::sqrt(std::abs(bg.cS2(N)))
                        : std::sqrt(std::abs(bg.cT2(N)));
        double ln_kcs_aH = ln_k + std::log(cs_now) - ln_aH(N, bg);

        if (ln_kcs_aH < ln_fo) {
            // |u_phys/z|^2 = exp(2*ln_amp) * (y[0]^2+y[1]^2) * exp(-2*lnz)
            double lnz   = (sec==Sector::SCALAR) ? bg.lnzS(N) : bg.lnzT(N);
            double ln_P_raw = 2.0*ln_amp
                              + std::log(y[0]*y[0]+y[1]*y[1])
                              - 2.0*lnz;
            P_curr = std::exp(ln_P_raw);

            if (P_prev > 0.0) {
                double rel = std::abs(P_curr-P_prev)/(P_prev+1e-100);
                if (rel < PS_CONV) {
                    if (++frozen_steps >= 5) break;
                } else {
                    frozen_steps = 0;
                }
            }
            P_prev = P_curr;
        }

        y  = ms_rk4(N, y, dN, ln_k, sec, bg);
        N += dN;
    }

    // Final read-off
    {
        double lnz   = (sec==Sector::SCALAR) ? bg.lnzS(N-dN) : bg.lnzT(N-dN);
        double ln_P_raw = 2.0*ln_amp
                          + std::log(y[0]*y[0]+y[1]*y[1])
                          - 2.0*lnz;
        P_curr = std::exp(ln_P_raw);
    }

    // Dimensionless power spectrum: k^3/(2pi^2) * |u/z|^2
    double k3_over_2pi2 = std::exp(3.0*ln_k) / (2.0*M_PI*M_PI);
    double result = k3_over_2pi2 * P_curr;
    return (sec == Sector::SCALAR) ? result : 2.0 * result;
}

} // namespace nmdc
