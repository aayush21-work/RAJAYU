#pragma once
// power_spectrum.hpp
// Computes scalar and tensor power spectra over a CMB k grid.
//
// OVERFLOW SAFETY: For long inflation runs (N_end > 709), exp(N) overflows
// double precision. We therefore store ln(k) = ln(aH) at horizon crossing
// instead of k itself, and pass ln_k to the MS solver.
// All comparisons use ln(k) - ln(aH) = ln(k/aH) which is O(1) near crossing.
//
// k grid: modes parameterised by N_cross = N_end - N_exit in [45, 65].
// Pivot:  N* = 55 e-folds before end of inflation.

#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ms_solver.hpp"
#include "bg_interp.hpp"
#include "constants.hpp"

namespace nmdc {

struct PowerSpectrum {
    std::vector<double> ln_k;    // ln(k) — overflow-safe
    std::vector<double> ln_PS;   // ln(P_scalar) — overflow-safe
    std::vector<double> ln_PT;   // ln(P_tensor) — overflow-safe
    std::vector<double> Ncross;  // N_cross for each mode
};

struct Observables {
    double ns;
    double nT;
    double r;
    double AS;
    double AT;
};

// ---------------------------------------------------------------
// ln(aH) = N + ln(H) at a given e-fold. Safe for any N.
// ---------------------------------------------------------------
inline double ln_aH_at(const std::vector<BgState>& bg, double N_exit) {
    if (N_exit <= bg.front().N)
        return bg.front().N + std::log(bg.front().H);
    if (N_exit >= bg.back().N)
        return bg.back().N  + std::log(bg.back().H);

    int lo = 0, hi = static_cast<int>(bg.size()) - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (bg[mid].N <= N_exit) lo = mid;
        else                     hi = mid;
    }
    double dN = bg[hi].N - bg[lo].N;
    double t  = (dN > 0.0) ? (N_exit - bg[lo].N) / dN : 0.0;
    double H  = (1.0-t)*bg[lo].H + t*bg[hi].H;
    return N_exit + std::log(H);
}

// ---------------------------------------------------------------
// Build ln_k grid. Each entry = ln(aH) at horizon crossing.
// N_cross in [45, 65] e-folds before end.
// ---------------------------------------------------------------
inline std::vector<double> make_ln_k_grid(const std::vector<BgState>& bg,
                                           std::vector<double>& Ncross_out) {
    double N_end = bg.back().N;

    // CMB window: ideally N_cross in [45, 65].
    // If N_end < 75, shrink window to fit what we have,
    // keeping a 5 e-fold buffer at each end.
    double BUFFER = 5.0;
    double ideal_min = 45.0, ideal_max = 65.0;

    double N_CROSS_MAX = std::min(ideal_max, N_end - BUFFER);
    double N_CROSS_MIN = std::max(ideal_min, BUFFER);

    // If even the minimum window is not available, use whatever we have
    if (N_CROSS_MIN >= N_CROSS_MAX) {
        // Last resort: just use middle 60% of available e-folds
        N_CROSS_MIN = 0.2 * N_end;
        N_CROSS_MAX = 0.8 * N_end;
    }

    if (N_CROSS_MIN >= N_CROSS_MAX)
        throw std::runtime_error(
            "make_ln_k_grid: N_end=" + std::to_string(N_end) +
            " too small to build any k grid. Increase phi0.");

    std::cout << "CMB window: N_cross in ["
              << N_CROSS_MIN << ", " << N_CROSS_MAX << "]\n";
    if (N_CROSS_MAX < 55.0)
        std::cout << "WARNING: N_end=" << N_end
                  << " too small for standard N*=55 pivot. "
                  << "Pivot will be set to N_end/2.\n";

    std::vector<double> ln_kgrid;
    ln_kgrid.reserve(N_K);
    Ncross_out.reserve(N_K);

    double step = (N_CROSS_MAX - N_CROSS_MIN) / (N_K - 1);
    for (int i = 0; i < N_K; ++i) {
        double Nc     = N_CROSS_MIN + i * step;
        double N_exit = N_end - Nc;
        ln_kgrid.push_back(ln_aH_at(bg, N_exit));
        Ncross_out.push_back(Nc);
    }
    return ln_kgrid;
}

// ---------------------------------------------------------------
// Pivot ln(k) = ln(aH) at N* e-folds before end.
// ---------------------------------------------------------------
inline double pivot_ln_k(const std::vector<BgState>& bg,
                          double N_star = 55.0) {
    double N_end   = bg.back().N;
    // If N_end < N_star + 5, use N_end/2 as pivot
    double N_pivot = (N_end > N_star + 5.0) ? N_star : N_end * 0.5;
    return ln_aH_at(bg, N_end - N_pivot);
}

// ---------------------------------------------------------------
// Compute power spectra for all modes. Parallelised over k.
// ---------------------------------------------------------------
inline PowerSpectrum compute_power_spectra(const std::vector<BgState>& bg) {
    BgInterp interp(bg);

    PowerSpectrum ps;
    ps.ln_k  = make_ln_k_grid(bg, ps.Ncross);
    ps.ln_PS.resize(N_K, std::numeric_limits<double>::quiet_NaN());
    ps.ln_PT.resize(N_K, std::numeric_limits<double>::quiet_NaN());

    std::cout << "ln_k grid: " << ps.ln_k.front()
              << " to " << ps.ln_k.back() << "\n";
    std::cout << "N_cross range: " << ps.Ncross.front()
              << " to " << ps.Ncross.back() << " e-folds before end\n";
    std::cout << "Computing " << N_K << " modes...\n";

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for (int i = 0; i < N_K; ++i) {
        double ln_k = ps.ln_k[i];
        try {
            ps.ln_PS[i] = solve_mode_ln(ln_k, Sector::SCALAR, interp);
            ps.ln_PT[i] = solve_mode_ln(ln_k, Sector::TENSOR, interp);
        } catch (const std::exception& e) {
            // NaN signals failure — skipped in fit
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            std::cerr << "Mode N_cross=" << ps.Ncross[i]
                      << " failed: " << e.what() << "\n";
        }
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        { if ((i+1) % 10 == 0) std::cout << "  " << (i+1) << "/" << N_K << "\n"; }
    }

    return ps;
}

// ---------------------------------------------------------------
// Linear regression: lnP = slope * ln_k + intercept
// Input: ln_k and lnP directly (no log taken inside).
// Skips NaN values.
// ---------------------------------------------------------------
struct LinFit { double slope, intercept; };

inline LinFit log_log_fit(const std::vector<double>& ln_k,
                           const std::vector<double>& lnP,
                           double ln_k_pivot,
                           double fit_decades = 1.0) {
    double window = fit_decades * std::log(10.0);
    double sx=0, sy=0, sxx=0, sxy=0;
    int n = 0;

    for (int i = 0; i < static_cast<int>(ln_k.size()); ++i) {
        if (!std::isfinite(lnP[i])) continue;
        if (std::abs(ln_k[i] - ln_k_pivot) > window) continue;
        sx  += ln_k[i];   sy  += lnP[i];
        sxx += ln_k[i]*ln_k[i];  sxy += ln_k[i]*lnP[i];
        ++n;
    }

    if (n < 2)
        throw std::runtime_error(
            "log_log_fit: fewer than 2 valid points in window.");

    double D = n*sxx - sx*sx;
    return { (n*sxy - sx*sy)/D, (sy - (n*sxy-sx*sy)/D * sx)/n };
}

// ---------------------------------------------------------------
// Compute observables from log-log fit around pivot.
// ---------------------------------------------------------------
inline Observables compute_observables(const PowerSpectrum& ps,
                                        const std::vector<BgState>& bg) {
    double N_end   = bg.back().N;
    double N_star  = (N_end > 60.0) ? 55.0 : N_end * 0.5;
    double lnkp    = pivot_ln_k(bg, N_star);
    std::cout << "Pivot: N* = " << N_star << " e-folds before end\n";
    std::cout << "Pivot ln(k) = " << lnkp << "\n";

    // Fit ln(P) vs ln(k) directly — no overflow
    LinFit fitS = log_log_fit(ps.ln_k, ps.ln_PS, lnkp, 1.0);
    LinFit fitT = log_log_fit(ps.ln_k, ps.ln_PT, lnkp, 1.0);

    // ns-1 = d(lnP)/d(lnk) = slope of fit
    // But our lnP = 3*ln_k + ln(|u/z|^2) - ln(2pi^2) [from solve_mode_ln]
    // So slope of lnP vs ln_k = 3 + d(ln|u/z|^2)/d(ln_k)
    // ns - 1 = slope - 3 + 1 ... NO.
    // Actually P_zeta = k^3/(2pi^2) * |u/z|^2
    // ln P_zeta = 3*ln_k + ln(|u/z|^2) - ln(2pi^2)
    // d(ln P_zeta)/d(ln_k) = 3 + d(ln|u/z|^2)/d(ln_k)
    // But ns - 1 = d(ln P_zeta)/d(ln_k) at fixed time... 
    // Standard definition: ns - 1 = d ln P / d ln k  (total)
    // So ns - 1 = fitS.slope directly. Correct.
    Observables obs;
    obs.ns = 1.0 + fitS.slope;
    obs.nT =       fitT.slope;

    // Amplitudes: lnP(k*) = fitS.intercept + fitS.slope * lnkp
    // AS = exp(lnP(k*)) -- may overflow if lnP(k*) > 700
    // Store ln(AS) and ln(AT) too
    double lnAS = fitS.intercept + fitS.slope * lnkp;
    double lnAT = fitT.intercept + fitT.slope * lnkp;
    // r = AT/AS = exp(lnAT - lnAS) -- always finite
    obs.r  = std::exp(lnAT - lnAS);
    // AS and AT themselves may overflow — store as exp() if safe, else NaN
    obs.AS = (lnAS < 700.0) ? std::exp(lnAS) : std::numeric_limits<double>::quiet_NaN();
    obs.AT = (lnAT < 700.0) ? std::exp(lnAT) : std::numeric_limits<double>::quiet_NaN();

    // Print sample points
    std::cout << "Sample points near pivot:\n";
    int shown = 0;
    for (int i = 0; i < N_K && shown < 5; ++i) {
        if (!std::isfinite(ps.ln_PS[i])) continue;
        if (std::abs(ps.ln_k[i] - lnkp) > std::log(10.0)) continue;
        std::cout << "  N_cross=" << ps.Ncross[i]
                  << " lnPS=" << ps.ln_PS[i]
                  << " lnPT=" << ps.ln_PT[i]
                  << " r=" << std::exp(ps.ln_PT[i] - ps.ln_PS[i]) << "\n";
        ++shown;
    }

    return obs;
}

// ---------------------------------------------------------------
// Write output file.
// ---------------------------------------------------------------
inline void write_power_spectrum(const PowerSpectrum& ps,
                                  const Observables& obs,
                                  const std::vector<BgState>& bg) {
    std::ofstream fout(PS_FILE);
    if (!fout.is_open())
        throw std::runtime_error("Cannot open " + std::string(PS_FILE));

    fout.precision(15);
    fout << "# NMDC inflation power spectrum\n";
    fout << "# kappa  = " << KAPPA       << "\n";
    fout << "# lambda = " << LAMBDA      << "\n";
    fout << "# n_pow  = " << N_POW       << "\n";
    fout << "# N_end  = " << bg.back().N << "\n";
    fout << "# ns     = " << obs.ns      << "\n";
    fout << "# nT     = " << obs.nT      << "\n";
    fout << "# r      = " << obs.r       << "\n";
    fout << "# ln(AS) = " << (std::isfinite(obs.AS) ? std::log(obs.AS) : 0.0) << "\n";
    fout << "# ln(k)   N_cross   ln(P_scalar)   ln(P_tensor)\n";

    for (int i = 0; i < N_K; ++i)
        if (std::isfinite(ps.ln_PS[i]) && std::isfinite(ps.ln_PT[i]))
            fout << ps.ln_k[i]   << "  "
                 << ps.Ncross[i] << "  "
                 << ps.ln_PS[i]  << "  "
                 << ps.ln_PT[i]  << "\n";

    std::cout << "\n=== Observables (N* = 55 e-folds before end) ===\n";
    std::cout << "  ns  = " << obs.ns << "\n";
    std::cout << "  nT  = " << obs.nT << "\n";
    std::cout << "  r   = " << obs.r  << "\n";
    if (std::isfinite(obs.AS))
        std::cout << "  AS  = " << obs.AS << "\n";
    else
        std::cout << "  AS overflows double (ln AS too large)\n";
    std::cout << "Written to " << PS_FILE << "\n";
}

} // namespace nmdc
