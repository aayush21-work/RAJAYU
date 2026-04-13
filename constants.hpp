#pragma once
// constants.hpp
// Global constants and model parameters.
// Units: Mpl = 1 throughout.

namespace nmdc {

// ---------------------------------------------------------------
// Model parameters — set before running
// ---------------------------------------------------------------

// NMDC coupling constant kappa = 1/M^2
// For F = kappa*H^2 ~ 3 at phi~10 (H~6.3e-5 Mpl): kappa ~ 1.5e9
// This gives r ~ 0.03, consistent with Yang et al. phi^2 NMDC result
inline constexpr double KAPPA  = 0.05;
inline constexpr double LAMBDA = 1.0e-10;
inline constexpr double N_POW  = 1.0;

// ---------------------------------------------------------------
// Solver parameters
// ---------------------------------------------------------------

inline constexpr double N_INI  = 0.0;
inline constexpr double N_END  = 200.0;   // phi^(1/3) inflates ~70 e-folds
inline constexpr double DN     = 1.0e-3;  // finer step for fractional n

// Background output file
inline constexpr const char* BG_FILE = "background.dat";

// ---------------------------------------------------------------
// Perturbation solver parameters
// ---------------------------------------------------------------

// Number of k modes
inline constexpr int    N_K       = 100;

// Bunch-Davies: initialise when k*cs = BD_FACTOR * aH
inline constexpr double BD_FACTOR  = 100.0;

// Freeze-out: declare convergence when k*cs < this * aH
inline constexpr double FO_FACTOR  = 0.01;

// Convergence: |dP/dN| / P < this over one e-fold
inline constexpr double PS_CONV   = 1.0e-3;

// Power spectrum output file
inline constexpr const char* PS_FILE = "power_spectrum.dat";

} // namespace nmdc
