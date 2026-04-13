// NMDC background solver in e-fold time (N = ln a)
// Variables: phi, phi' = dphi/dN, H, H' = dH/dN
// Relations: dphi/dt = H * phi', dH/dt = H * H'
//            ddphi/dt^2 = H^2 * phi'' + H * Hdot * phi'/H
//                       = H^2 * phi'' + Hdot * phi'
//            ddH/dt^2   = H^2 * H'' + Hdot * H'

const double lambda2 = 0.05;

double V(double phi) { return 0.5 * phi * phi; }
double dV(double phi) { return 1 * phi; }

// Background EOMs return cosmic-time second derivatives (same as before)
// Then we convert to N-derivatives in the RK4 driver
