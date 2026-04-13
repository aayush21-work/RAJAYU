// main.cpp
// NMDC inflation solver.
// Solves background + Mukhanov-Sasaki equations exactly (no slow-roll).
// Outputs: background.dat, power_spectrum.dat
//
// Usage: ./nmdc_solver
//
// Initial conditions: set phi0, dphi0 below.
// Model parameters:   set in constants.hpp

#include <iostream>
#include <stdexcept>
#include "bg_solver.hpp"
#include "power_spectrum.hpp"

int main() {
    try {
        // -------------------------------------------------------
        // Initial conditions at N = 0
        // For power-law V = lambda*phi^2/2:
        //   phi0 ~ 15 Mpl gives ~60 e-folds in GR
        //   dphi0 ~ small (slow-roll start)
        // -------------------------------------------------------
        // Auto-compute phi0 to give ~70 e-folds for any N_POW.
        // In GR slow-roll: N ~ (phi0^2 - phi_end^2) / (2*n)
        // phi_end ~ n/sqrt(2) (where eps=1 in GR)
        // So phi0 ~ sqrt(2*n*N_target + phi_end^2)
        double N_target  = 70.0;
        double n         = nmdc::N_POW;
        double phi_end   = n / std::sqrt(2.0);
        double phi0      = std::sqrt(2.0*n*N_target + phi_end*phi_end);
        // Clamp to reasonable range
        if (phi0 < 1.0) phi0 = 2.0;
        if (phi0 > 25.0) phi0 = 25.0;

        // Compute slow-roll IC for dphi self-consistently.
        // From KG in slow-roll: dphi' = -V_phi / [3*H^2*(1+3F)]
        // Iterate twice: first with GR H^2~V/3, then with exact Friedmann.
        double dphi0;
        {
            double Vp = nmdc::V_phi(phi0);
            // First estimate with GR H^2
            double H2 = nmdc::V(phi0) / 3.0;
            double F  = nmdc::KAPPA * H2;
            dphi0 = -Vp / (3.0 * H2 * (1.0 + 3.0 * F));
            // Refine: solve exact Friedmann with this dphi0
            try {
                H2    = nmdc::solve_H2(phi0, dphi0);
                F     = nmdc::KAPPA * H2;
                dphi0 = -Vp / (3.0 * H2 * (1.0 + 3.0 * F));
                // Second refinement
                H2    = nmdc::solve_H2(phi0, dphi0);
                F     = nmdc::KAPPA * H2;
                dphi0 = -Vp / (3.0 * H2 * (1.0 + 3.0 * F));
            } catch (...) {
                // fallback to first estimate if quadratic fails
            }
        }

        std::cout << "=== NMDC Inflation Solver ===\n";
        std::cout << "kappa  = " << nmdc::KAPPA  << "\n";
        std::cout << "lambda = " << nmdc::LAMBDA << "\n";
        std::cout << "n_pow  = " << nmdc::N_POW  << "\n";
        std::cout << "phi0   = " << phi0   << "\n";
        std::cout << "dphi0  = " << dphi0  << "\n\n";

        // -------------------------------------------------------
        // Step 1: Background
        // -------------------------------------------------------
        std::cout << "--- Background solver ---\n";
        auto bg = nmdc::run_background(phi0, dphi0, /*write_file=*/true);
        std::cout << "Background written to " << nmdc::BG_FILE << "\n\n";

        // -------------------------------------------------------
        // Step 2: Power spectra
        // -------------------------------------------------------
        std::cout << "--- Perturbation solver ---\n";
        auto ps  = nmdc::compute_power_spectra(bg);
        auto obs = nmdc::compute_observables(ps, bg);
        nmdc::write_power_spectrum(ps, obs, bg);

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
