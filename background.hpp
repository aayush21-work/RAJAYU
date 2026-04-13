#pragma once
// background.hpp
// Computes all exact background quantities from state vector (phi, dphi, H)
// where dphi = d phi / dN  and  N = ln a is e-fold time.
//
// Equations derived from KYY (2011) with NMDC mapping:
//   G2 = X - V,  G3 = 0,  G4 = 1/2,  G5 = -(kappa/2)*phi
//   G5_phi = -kappa/2,  all other G_i derivatives = 0
//
// State vector: y = {phi, dphi, H}
// All quantities in units Mpl = 1.

#include <cmath>
#include <stdexcept>
#include "constants.hpp"
#include "potential.hpp"

namespace nmdc {

// ---------------------------------------------------------------
// Background state: everything computed from (phi, dphi, H)
// ---------------------------------------------------------------
struct BgState {
    // --- primary state ---
    double N;       // e-fold time
    double phi;     // inflaton field
    double dphi;    // phi' = d phi / dN
    double H;       // Hubble parameter

    // --- derived ---
    double X;       // kinetic variable = H^2 * dphi^2 / 2
    double eps_H;   // slow-roll epsilon = -H'/H  (=  -Hdot/H^2 )
    double ddphi;   // phi'' = d^2 phi / dN^2
    double Hprime;  // H' = dH/dN = -eps_H * H

    // --- perturbation coefficients ---
    double FT;      // tensor kinetic coeff
    double GT;      // tensor gradient coeff
    double cT2;     // tensor sound speed squared
    double Sigma;   // scalar coefficient Sigma
    double Theta;   // scalar coefficient Theta
    double FS;      // scalar kinetic coeff  (requires Theta')
    double GS;      // scalar gradient coeff
    double cS2;     // scalar sound speed squared

    // --- MS pump field helpers (stored for numerical differentiation) ---
    double zS;      // sqrt(2) * a * (FS*GS)^{1/4}   (a stored separately)
    double zT;      // (a/2) * (FT*GT)^{1/4}
    double lnzS;    // ln(zS)  -- differentiated numerically
    double lnzT;    // ln(zT)
};

// ---------------------------------------------------------------
// Solve Friedmann equation for H^2 given (phi, dphi).
// Exact quadratic:
//   (9*kappa*dphi^2/2) * H^4 - (3 - dphi^2/2) * H^2 + V = 0
// Returns positive root. Throws if discriminant < 0.
// ---------------------------------------------------------------
inline double solve_H2(double phi, double dphi) {
    const double kap  = KAPPA;
    const double Vphi = V(phi);
    const double dp2  = dphi * dphi;

    const double a_coef = 4.5 * kap * dp2;          // 9*kappa/2 * dphi^2
    const double b_coef = -(3.0 - 0.5 * dp2);       // -(3 - dphi^2/2)
    const double c_coef = Vphi;                       // V(phi)

    // Standard inflation: no NMDC limit
    if (std::abs(a_coef) < 1.0e-30) {
        // kappa -> 0: H^2 = V / (3 - dphi^2/2)
        double denom = 3.0 - 0.5 * dp2;
        if (denom <= 0.0)
            throw std::runtime_error("solve_H2: kinetic domination (dphi^2 >= 6), inflation ended");
        return c_coef / denom;
    }

    // Check b_coef sign: if dp2 > 6, b_coef > 0 and both roots likely negative
    // This signals kinetic domination — inflation has ended
    if (dp2 >= 6.0)
        throw std::runtime_error("solve_H2: kinetic domination (dphi^2 >= 6), inflation ended");

    double disc = b_coef * b_coef - 4.0 * a_coef * c_coef;
    if (disc < 0.0)
        throw std::runtime_error("solve_H2: negative discriminant in Friedmann quadratic");

    // Two roots: take the one that reduces to standard GR for kappa->0
    // For kappa->0, positive root -> V/(3-dp2/2), negative root -> unphysical
    double sqrtD = std::sqrt(disc);
    double H2_plus  = (-b_coef + sqrtD) / (2.0 * a_coef);
    double H2_minus = (-b_coef - sqrtD) / (2.0 * a_coef);

    // Pick the physical root: must be positive and reduce to GR limit
    // In GR limit b_coef = -(3-dp2/2) < 0, so -b > 0, plus root is larger
    // We want the root that is positive and consistent with inflation
    if (H2_plus > 0.0 && H2_minus > 0.0) {
        // Both positive: take the smaller one (standard branch)
        return std::min(H2_plus, H2_minus);
    } else if (H2_plus > 0.0) {
        return H2_plus;
    } else if (H2_minus > 0.0) {
        return H2_minus;
    } else {
        throw std::runtime_error("solve_H2: no positive root found");
    }
}

// ---------------------------------------------------------------
// Compute phi'' from exact KG equation (N-time):
//
//   phi'' = -phi' * [(3 - eps_H) - 6*kappa*eps_H*H^2 / (1 + 3*kappa*H^2)]
//           - V_phi / [H^2 * (1 + 3*kappa*H^2)]
//
// This is called with a trial eps_H (from previous step or iteration).
// ---------------------------------------------------------------
inline double compute_ddphi(double phi, double dphi, double H,
                             double eps_H) {
    const double kap  = KAPPA;
    const double H2   = H * H;
    const double kH2  = kap * H2;
    const double denom = 1.0 + 3.0 * kH2;

    double term1 = dphi * ((3.0 - eps_H) - 6.0 * kH2 * eps_H / denom);
    double term2 = V_phi(phi) / (H2 * denom);

    return -(term1 + term2);
}

// ---------------------------------------------------------------
// Compute epsilon_H from exact Raychaudhuri (N-time):
//
// From sum P_i = 0, using Friedmann to eliminate V:
//   2*Hdot*(1 - kappa*X) = -dphi^2*(1+3*kappa*H^2) + 2*kappa*H^2*dphi*ddphi
// With Hdot = -eps_H*H^2 and X = H^2*dphi^2/2:
//   eps_H = [dphi^2*(1+3*kappa*H^2) - 2*kappa*H^2*dphi*ddphi]
//           / [2*H^2*(1 - kappa*H^2*dphi^2/2) * (1/H^2)]
// In N-time (dphi = phi', ddphi -> ddphi terms already in N-derivatives):
//   eps_H = [phi'^2*(1+3*kappa*H^2) + 2*kappa*H^2*phi'*phi'']  <- numerator
//           / [2*(1 - kappa*H^2*phi'^2/2)]                       <- denominator
// Note: sign of ddphi term: Hdot = -eps*H^2 so
//   -2*eps*H^2*(1-kX) = -dp2*(1+3kH2) + 2kH2*dp*ddp  [cosmic time]
// Converting ddphi_cosmic = H^2*(phi'' - eps*phi') and collecting eps terms:
//   eps_H * [2*(1-kX) + 2*kH2*dp2 - ... ] = ...
// The self-consistent solution derived carefully:
//   eps_H = [dp2*(1+3kH2) + 2kH2*dp*ddp_N] / [2*(1 - kX + kH2*dp2)]
// where kX = kappa*H^2*dp^2/2 and ddp_N = phi'' in N-time
// Simplifying denominator: 1 - kX + kH2*dp2 = 1 - kH2*dp2/2 + kH2*dp2 = 1 + kH2*dp2/2 = 1 + kX
// Wait -- let me redo carefully.
//
// EXACT derivation (cosmic time, then convert):
// 2*Hdot*(1 - kX) = -dphi_dot^2*(1+3kH^2) + 2k*H*dphi_dot*ddphi_dot
// Hdot = H' * H  (where ' = d/dN, H' = -eps_H*H)
// dphi_dot = H*phi'
// ddphi_dot = H^2*(phi'' - eps_H*phi')
// Substituting:
// -2*eps_H*H^2*(1 - kH^2*phi'^2/2) = -H^2*phi'^2*(1+3kH^2)
//                                     + 2k*H*H*phi'*H^2*(phi''-eps_H*phi')
// Dividing by H^2:
// -2*eps_H*(1 - kH^2*phi'^2/2) = -phi'^2*(1+3kH^2) + 2kH^2*phi'*(phi''-eps_H*phi')
// Expanding RHS eps_H term:
// -2*eps_H*(1 - kX) = -phi'^2*(1+3kH^2) + 2kH^2*phi'*phi'' - 2kH^2*phi'^2*eps_H
// Collecting eps_H:
// eps_H*[-2*(1-kX) + 2kH^2*phi'^2] = -phi'^2*(1+3kH^2) + 2kH^2*phi'*phi''
// Note: -2*(1-kX) + 2kH^2*phi'^2 = -2 + 2kX + 4kX = -2 + 6kX ... no
// kX = kappa*H^2*phi'^2/2, so 2kX = kH^2*phi'^2
// -2*(1-kX) + 2kH^2*phi'^2 = -2 + 2kX + 4kX = -2 + 6kX
// Hmm that doesn't simplify nicely. Let me use kX directly:
// -2*(1-kX) + 2*2kX = -2 + 2kX + 4kX = -2 + 6kX
// So: eps_H = [phi'^2*(1+3kH^2) - 2kH^2*phi'*phi''] / [2*(1-kX) - 4kX]
//           = [phi'^2*(1+3kH^2) - 2kH^2*phi'*phi''] / [2 - 2kX - 4kX]
//           = [phi'^2*(1+3kH^2) - 2kH^2*phi'*phi''] / [2*(1 - 3kX)]
//
// FINAL EXACT FORMULA:
//   eps_H = [phi'^2*(1+3kH^2) - 2kH^2*phi'*phi''] / [2*(1 - 3*kappa*X)]
// where X = H^2*phi'^2/2.
// ---------------------------------------------------------------
inline double compute_eps_H(double dphi, double H, double ddphi) {
    const double kap = KAPPA;
    const double H2  = H * H;
    const double dp2 = dphi * dphi;
    const double kH2 = kap * H2;
    const double kX  = kap * H2 * dp2 / 2.0;   // kappa * X

    double num   = dp2 * (1.0 + 3.0 * kH2)
                 - 2.0 * kH2 * dphi * ddphi;
    double denom = 2.0 * (1.0 - 3.0 * kX);

    // Guard: if denom ~ 0 (strong coupling), fall back to kinematic estimate
    if (std::abs(denom) < 1.0e-10)
        return dp2 / 2.0;   // GR fallback

    return num / denom;
}

// ---------------------------------------------------------------
// Self-consistent solve for (eps_H, ddphi) by fixed-point iteration.
// Typically converges in 3-5 iterations.
// ---------------------------------------------------------------
inline void solve_eps_ddphi(double phi, double dphi, double H,
                             double& eps_H, double& ddphi) {
    // Initial guess: slow-roll eps_H ~ dphi^2/2
    eps_H = 0.5 * dphi * dphi;
    ddphi = compute_ddphi(phi, dphi, H, eps_H);

    for (int iter = 0; iter < 20; ++iter) {
        double eps_new   = compute_eps_H(dphi, H, ddphi);
        double ddphi_new = compute_ddphi(phi, dphi, H, eps_new);

        double diff = std::abs(eps_new - eps_H);
        eps_H = eps_new;
        ddphi = ddphi_new;

        if (diff < 1.0e-14) break;
    }
}

// ---------------------------------------------------------------
// Compute all perturbation coefficients from (phi, dphi, H, eps_H, ddphi)
// ---------------------------------------------------------------
inline void compute_perturbation_coeffs(double /*phi*/, double dphi,
                                        double H, double eps_H,
                                        double ddphi, BgState& s) {
    const double kap = KAPPA;
    const double H2  = H * H;
    const double X   = 0.5 * H2 * dphi * dphi;   // X = H^2 phi'^2 / 2
    const double kX  = kap * X;

    // --- Tensor coefficients (exact, KYY 4.4-4.5 with G5phi = -kappa/2) ---
    s.FT  = 1.0 + kX;
    s.GT  = 1.0 - kX;
    s.cT2 = s.FT / s.GT;     // superluminal: cT^2 > 1 for kappa > 0

    // --- Sigma and Theta (exact, KYY 4.25-4.26) ---
    s.Sigma = X * (1.0 + 18.0 * kap * H2) - 3.0 * H2;
    s.Theta = H * (1.0 - 3.0 * kX);

    // --- GS (exact, KYY 4.33) ---
    double GT2   = s.GT * s.GT;
    double Th2   = s.Theta * s.Theta;
    s.GS = (s.Sigma / Th2) * GT2 + 3.0 * s.GT;

    // --- GT' and Theta' needed for FS ---
    // GT' = kap * H^2 * dphi * (ddphi - eps_H * dphi)
    double GT_prime = kap * H2 * dphi * (ddphi - eps_H * dphi);

    // Theta' = -eps_H * Theta + H * (-3*kap/2) * d(H^2*dphi^2)/dN
    //        = -eps_H * Theta + H * (-3*kap) * H^2 * dphi * (ddphi - eps_H*dphi)
    double dX_dN    = H2 * dphi * (ddphi - eps_H * dphi);  // X' = d(H^2 phi'^2/2)/dN
    double Th_prime = -eps_H * s.Theta + H * (-3.0 * kap) * dX_dN;

    // --- FS (exact, KYY 4.32):
    //   FS = H * (GT^2/Theta)' + H * GT^2/Theta - FT
    //      = H * [2*GT*GT'*Theta - GT^2*Theta'] / Theta^2
    //        + H * GT^2/Theta - FT
    double bracket = (2.0 * s.GT * GT_prime * s.Theta
                      - GT2 * Th_prime) / Th2;
    s.FS = H * bracket + H * GT2 / s.Theta - s.FT;

    // --- Scalar sound speed ---
    s.cS2 = s.FS / s.GS;
}

// ---------------------------------------------------------------
// Fill a BgState from (N, phi, dphi, H).
// Solves for eps_H, ddphi self-consistently, then fills all derived quantities.
// ---------------------------------------------------------------
inline BgState make_bg_state(double N, double phi, double dphi, double H) {
    BgState s{};
    s.N     = N;
    s.phi   = phi;
    s.dphi  = dphi;
    s.H     = H;
    s.X     = 0.5 * H * H * dphi * dphi;

    solve_eps_ddphi(phi, dphi, H, s.eps_H, s.ddphi);

    s.Hprime = -s.eps_H * H;

    compute_perturbation_coeffs(phi, dphi, H, s.eps_H, s.ddphi, s);

    // ln(zS) and ln(zT): a = exp(N), stored as ln(zS) = N + (1/4)*ln(FS*GS) + ln(sqrt(2))
    // We store ln(z) for numerical differentiation
    s.lnzS = N + 0.5 * std::log(2.0)
               + 0.25 * std::log(std::abs(s.FS * s.GS));
    s.lnzT = N - std::log(2.0)
               + 0.25 * std::log(std::abs(s.FT * s.GT));

    return s;
}

} // namespace nmdc
