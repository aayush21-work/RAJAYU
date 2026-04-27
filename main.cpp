#include <cmath>
#include <fstream>
#include <iostream>
#include "model.hpp"


// ddH/dt^2 from the NMDC constraint
double ddH_cosmic(double dphi_t, double H, double dH_t) {
  return 2.0 * H *
         (-2.0 * dH_t * lambda2 +
          dphi_t * dphi_t * (-3.0 * H * H - 4.0 * dH_t - lambda2)) /
         (dphi_t * dphi_t);
}

// ddphi/dt^2 from the NMDC constraint
double ddphi_cosmic(double phi, double dphi_t, double H, double dH_t) {
  double num = 9.0 * H * H * H * dphi_t * dphi_t +
               9.0 * H * dH_t * dphi_t * dphi_t + 12.0 * H * dH_t * lambda2 +
               3.0 * H * dphi_t * dphi_t * lambda2 - dV(phi) * dphi_t * lambda2;
  double den = dphi_t * (3.0 * H * H + 3.0 * dH_t + lambda2);
  return num / den;
}

// Convert to e-fold derivatives:
//   phi'' = ddphi_t / H^2 - phi' * Hdot / H^2
//   H''   = ddH_t / H^2 - H'^2 / H
// where phi' = dphi_t / H,  H' = Hdot / H,  Hdot = dH_t

double phi_pp(double phi, double phi_p, double H, double H_p) {
  // recover cosmic-time quantities
  double dphi_t = H * phi_p; // dphi/dt = H * phi'
  double Hdot = H * H_p;     // dH/dt = H * H'
  double ddphi_t = ddphi_cosmic(phi, dphi_t, H, Hdot);
  // phi'' = ddphi_t / H^2 - phi' * H' / H
  //       = ddphi_t / H^2 - phi' * Hdot / H^2
  return ddphi_t / (H * H) - phi_p * H_p / H;
}

double H_pp(double phi_p, double H, double H_p) {
  double dphi_t = H * phi_p;
  double Hdot = H * H_p;
  double ddH_t = ddH_cosmic(dphi_t, H, Hdot);
  // H'' = ddH_t / H^2 - H'^2 / H
  return ddH_t / (H * H) - H_p * H_p / H;
}

int main() {
  double phi = 15.0;

  //  slow-roll IC (same iteration as before) 
  double H = std::sqrt(V(phi) / 3.0);
  double Hdot = 0.0;
  double dphi_t = 0.0;

  for (int i = 0; i < 200; i++) {
    double K = Hdot + H * H;
    double gamma = 1.0 + 3.0 * K / lambda2;
    dphi_t = -dV(phi) / (3.0 * H * gamma);
    for (int j = 0; j < 50; j++) {
      double K2 = Hdot + H * H;
      double g2 = 1.0 + 3.0 * K2 / lambda2;
      double H2 = std::sqrt((0.5 * dphi_t * dphi_t * g2 + V(phi)) / 3.0);
      if (std::abs(H2 - H) < 1e-14)
        break;
      H = H2;
    }
    double K2 = Hdot + H * H;
    double g2 = 1.0 + 3.0 * K2 / lambda2;
    double Hdot_new = -0.5 * dphi_t * dphi_t * g2;
    if (std::abs(Hdot_new - Hdot) < 1e-14)
      break;
    Hdot = Hdot_new;
  }

  // Convert to e-fold variables
  double phi_p = dphi_t / H; // phi' = dphi/dt / H
  double H_p = Hdot / H;     // H' = Hdot / H

  double N = 0.0;
  double t = 0.0;
  double dN = 1e-4; // e-fold step size

  double eps = -Hdot / (H * H); // = -H'/H

  std::cout << "Initial: phi=" << phi << " phi'=" << phi_p << " H=" << H
            << " H'=" << H_p << " eps=" << eps << std::endl;

  std::ofstream out("background_nmdc.dat");

  while (eps < 1.0) {
    // ── RK4 in N ──
    double k1_phi = phi_p;
    double k1_phip = phi_pp(phi, phi_p, H, H_p);
    double k1_H = H_p;
    double k1_Hp = H_pp(phi_p, H, H_p);
    double k1_t = 1.0 / H; // dt/dN = 1/H

    double phi2 = phi + dN / 2 * k1_phi;
    double phip2 = phi_p + dN / 2 * k1_phip;
    double H2 = H + dN / 2 * k1_H;
    double Hp2 = H_p + dN / 2 * k1_Hp;
    double k2_phi = phip2;
    double k2_phip = phi_pp(phi2, phip2, H2, Hp2);
    double k2_H = Hp2;
    double k2_Hp = H_pp(phip2, H2, Hp2);
    double k2_t = 1.0 / H2;

    double phi3 = phi + dN / 2 * k2_phi;
    double phip3 = phi_p + dN / 2 * k2_phip;
    double H3 = H + dN / 2 * k2_H;
    double Hp3 = H_p + dN / 2 * k2_Hp;
    double k3_phi = phip3;
    double k3_phip = phi_pp(phi3, phip3, H3, Hp3);
    double k3_H = Hp3;
    double k3_Hp = H_pp(phip3, H3, Hp3);
    double k3_t = 1.0 / H3;

    double phi4 = phi + dN * k3_phi;
    double phip4 = phi_p + dN * k3_phip;
    double H4 = H + dN * k3_H;
    double Hp4 = H_p + dN * k3_Hp;
    double k4_phi = phip4;
    double k4_phip = phi_pp(phi4, phip4, H4, Hp4);
    double k4_H = Hp4;
    double k4_Hp = H_pp(phip4, H4, Hp4);
    double k4_t = 1.0 / H4;

    phi += (dN / 6) * (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi);
    phi_p += (dN / 6) * (k1_phip + 2 * k2_phip + 2 * k3_phip + k4_phip);
    H += (dN / 6) * (k1_H + 2 * k2_H + 2 * k3_H + k4_H);
    H_p += (dN / 6) * (k1_Hp + 2 * k2_Hp + 2 * k3_Hp + k4_Hp);
    t += (dN / 6) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t);
    N += dN;

    eps = -H_p / H;
    double Hdot_now = H * H_p;
    double K = Hdot_now + H * H;
    double dphi_t_now = H * phi_p;
    double Kdot = ddH_cosmic(dphi_t_now, H, Hdot_now) + 2.0 * H * Hdot_now;
    double phi_pp_now = phi_pp(phi, phi_p, H, H_p);

    double H_pp_now = H_pp(phi_p, H, H_p);

    // Output: N  phi  phi'  phi''  eps  H  H'  H''  K  Kdot  t
    out << N << " " << phi << " " << phi_p << " " << phi_pp_now << " " << eps
        << " " << H << " " << H_p << " " << H_pp_now << " " << K << " " << Kdot
        << " " << t << "\n";
  }

  std::cout << "Inflation ended: N=" << N << "  phi=" << phi
            << "  phi'=" << phi_p << "  eps=" << eps << "  t=" << t
            << std::endl;

  return 0;
}
