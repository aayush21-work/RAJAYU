#include <iostream>
#include <fstream>
#include <cmath>
#include "rk4.h"

// ---- Parameters --------------------------------------------
const double m         = 0.01;    // inflaton mass (Planck units)
const double phi_init  = 15.0;    // large field: need phi > sqrt(2) for inflation
const int    N_steps   = 2000000;
const double h         = 1e-3;    // cosmic time step (smaller for accuracy)
const double eps_end   = 1.0;     // stop when slow-roll parameter reaches 1

// State: { phi, dphi, N, a, tau }
//   phi  = inflaton field
//   dphi = dφ/dt
//   N    = e-fold number
//   a    = scale factor
//   tau  = conformal time  (dτ/dt = 1/a)
static const int NDIM = 5;

// ---- Potential (m²φ²/2 chaotic inflation) ------------------
double V  (double phi)
{
    return 0.5*m*m*phi*phi;
}
double dV (double phi)
{
    return m*m*phi;
}
double d2V(double /*phi*/)
{
    return m*m;
}

double Hubble(double phi, double dphi)
{
    return std::sqrt(0.5*dphi*dphi + V(phi));
}


double epsilon(double phi) 
{
    return 2.0/(phi*phi);
}

// Define g = ż/a = -2φ̇ - V'/H + 3φ̇³/(2H²)
//
// Then z''/z = a²H(2Hg + ġ)/φ̇,  where:
//   ġ = 6Hφ̇ + 2V' - V''φ̇/H - 6V'φ̇²/H² - (27/2)φ̇³/H + (9/2)φ̇⁵/H³
//
double zpp_over_z(double phi, double dphi, double H, double a)
{
    if (std::abs(dphi) < 1e-30) return 0.0;  // guard against t=0 with dphi=0

    double Vp  = dV(phi);
    double Vpp = d2V(phi);
    double H2  = H*H;
    double H3  = H2*H;
    double dp2 = dphi*dphi;
    double dp3 = dp2*dphi;
    double dp5 = dp3*dp2;

    double g    = -2.0*dphi - Vp/H + 1.5*dp3/H2;

    double gdot = 6.0*H*dphi + 2.0*Vp
                - Vpp*dphi/H
                - 6.0*Vp*dp2/H2
                - 13.5*dp3/H
                + 4.5*dp5/H3;

    return a*a*H*(2.0*H*g + gdot)/dphi;
}

void equations_of_motion(const double y[NDIM], double dydt[NDIM])
{
    double phi  = y[0];
    double dphi = y[1];
    double a    = y[3];
    double H    = Hubble(phi, dphi);

    dydt[0] = dphi;                     
    dydt[1] = -3.0*H*dphi - dV(phi);   
    dydt[2] = H;                        
    dydt[3] = a*H;                      
    dydt[4] = 1.0/a;        
}

int main()
{
    std::ofstream out("background.dat");
    out << "# t  tau  phi  dphi  a  H  zppz  N\n";

    //slow-roll approximation for initial conditions only

    double H0     = Hubble(phi_init, 0.0);
    double dphi0  = -dV(phi_init) / (3.0*H0);
    double y[NDIM] = { phi_init, dphi0, 0.0, 1.0, 0.0 };

    auto callback = [&](int i, const double y[NDIM]) -> bool
    {
        double phi  = y[0];
        double dphi = y[1];
        double N    = y[2];
        double a    = y[3];
        double tau  = y[4];
        double H    = Hubble(phi, dphi);
        double eps  = epsilon(phi);
        double zppz = zpp_over_z(phi, dphi, H, a);

        if (i % 100 == 0)
            out << i*h    << " "
                << tau    << " "
                << phi    << " "
                << dphi   << " "
                << a      << " "
                << H      << " "
                << zppz   << " "
                << N      << "\n";

        if (eps >= eps_end) {
            std::cout << "Inflation ended at t=" << i*h
                      << "  N_efolds=" << N << "\n";
            return false;
        }
        return true;
    };

    std::cout << "Evolving background...\n";
    rk4_integrate<NDIM>(y, h, N_steps, equations_of_motion, callback);
    std::cout << "Done. Saved to background.dat\n";
    return 0;
}
