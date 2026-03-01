#include <cmath>
#include <fstream>
#include <iostream>

double V(double phi) { return 0.5 * phi * phi; }
double dV(double phi) { return phi; }
double H(double phi, double dphi) {
  return std::sqrt(0.5 * dphi * dphi + V(phi));
}
double Hdot(double dphi) { return -0.5 * dphi * dphi; }

double epsilon(double phi, double dphi) {
  double h = H(phi, dphi);
  return -Hdot(dphi) / (h * h);
}

void rk4(double t, double phi, double dphi, double a, double h = 1e-4) {
  double kp1, kp2, kp3, kp4;
  double kd1, kd2, kd3, kd4;
  double ka1, ka2, ka3, ka4;
  double N = 0;
  int k = 0;
  double a0 = a;

  std::ofstream out("background.dat");

  while (epsilon(phi, dphi) < 1.0) {
    kp1 = dphi;
    kd1 = -3 * H(phi, dphi) * dphi - dV(phi);
    ka1 = H(phi, dphi) * a;

    kp2 = dphi + h / 2 * kd1;
    kd2 = -3 * H(phi + h / 2 * kp1, dphi + h / 2 * kd1) * (dphi + h / 2 * kd1) -
          dV(phi + h / 2 * kp1);
    ka2 = H(phi + h / 2 * kp1, dphi + h / 2 * kd1) * (a + h / 2 * ka1);

    kp3 = dphi + h / 2 * kd2;
    kd3 = -3 * H(phi + h / 2 * kp2, dphi + h / 2 * kd2) * (dphi + h / 2 * kd2) -
          dV(phi + h / 2 * kp2);
    ka3 = H(phi + h / 2 * kp2, dphi + h / 2 * kd2) * (a + h / 2 * ka2);

    kp4 = dphi + h * kd3;
    kd4 = -3 * H(phi + h * kp3, dphi + h * kd3) * (dphi + h * kd3) -
          dV(phi + h * kp3);
    ka4 = H(phi + h * kp3, dphi + h * kd3) * (a + h * ka3);

    phi = phi + (h / 6) * (kp1 + 2 * kp2 + 2 * kp3 + kp4);
    dphi = dphi + (h / 6) * (kd1 + 2 * kd2 + 2 * kd3 + kd4);
    a = a + (h / 6) * (ka1 + 2 * ka2 + 2 * ka3 + ka4);
    t += h;
    // N += H(phi, dphi) * h;
    k += 1;

    if (k % 10 == 0) {
      out << t << " " << phi << " " << dphi << " " << epsilon(phi, dphi) << " "
          << std::log(a / a0) << " " << a << std::endl;
    }
  }

  std::cout << "Inflation ended at t=" << t << "  phi=" << phi
            << "  dphi=" << dphi << "  epsilon=" << epsilon(phi, dphi) << " "
            << N << " " << a << " " << std::endl;
}

int main() {
  double phi = 10, dphi = 0.0, t = 0.0, a = 1e-20;
  rk4(t, phi, dphi, a);
  return 0;
}
