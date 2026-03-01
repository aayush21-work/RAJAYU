#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

struct BgState {
  double t, phi, dphi, epsilon, N, a;
};

std::vector<BgState> load_background(const std::string &filename) {
  std::vector<BgState> bg;
  std::ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Cannot open " << filename << "\n";
    std::exit(1);
  }
  BgState s;
  while (in >> s.t >> s.phi >> s.dphi >> s.epsilon >> s.N >> s.a)
    bg.push_back(s);
  in.close();
  std::cout << "Loaded " << bg.size() << " background points.\n";
  return bg;
}

double V(double phi) { return 0.5 * phi * phi; }
double dV(double phi) { return phi; }
double d2V(double phi) { return 1.0; }
double H(double phi, double dphi) {
  return std::sqrt(0.5 * dphi * dphi + V(phi));
}

double ddphi_rhs(double phi, double dphi) {
  return -3.0 * H(phi, dphi) * dphi - dV(phi);
}

double zpp_over_z(double phi, double dphi, double ddphi, double a) {
  double h = H(phi, dphi);
  return a * a *
         (2.5 * dphi * dphi + 2.0 * dphi * ddphi / h + 2.0 * h * h +
          0.5 * (dphi * dphi * dphi * dphi) / (h * h) - d2V(phi));
}

double f_vk(double vk, double dvk, double phi, double dphi, double ddphi,
            double a, double k) {
  double h = H(phi, dphi);
  double mass2 = (zpp_over_z(phi, dphi, ddphi, a) - k * k) / (a * a);
  return -h * dvk + mass2 * vk;
}

void coupled_step(double &phi, double &dphi, double &a, double &vk_re,
                  double &dvk_re, double &vk_im, double &dvk_im, double k,
                  double h_step) {
  double kp1, kp2, kp3, kp4;
  double kd1, kd2, kd3, kd4;
  double ka1, ka2, ka3, ka4;
  double kvre1, kvre2, kvre3, kvre4;
  double kdvre1, kdvre2, kdvre3, kdvre4;
  double kvim1, kvim2, kvim3, kvim4;
  double kdvim1, kdvim2, kdvim3, kdvim4;

  kp1 = dphi;
  kd1 = ddphi_rhs(phi, dphi);
  ka1 = H(phi, dphi) * a;
  kvre1 = dvk_re;
  kdvre1 = f_vk(vk_re, dvk_re, phi, dphi, kd1, a, k);
  kvim1 = dvk_im;
  kdvim1 = f_vk(vk_im, dvk_im, phi, dphi, kd1, a, k);

  double p2 = phi + h_step / 2 * kp1;
  double dp2 = dphi + h_step / 2 * kd1;
  double a2 = a + h_step / 2 * ka1;
  kp2 = dp2;
  kd2 = ddphi_rhs(p2, dp2);
  ka2 = H(p2, dp2) * a2;
  kvre2 = dvk_re + h_step / 2 * kdvre1;
  kdvre2 = f_vk(vk_re + h_step / 2 * kvre1, dvk_re + h_step / 2 * kdvre1, p2,
                dp2, kd2, a2, k);
  kvim2 = dvk_im + h_step / 2 * kdvim1;
  kdvim2 = f_vk(vk_im + h_step / 2 * kvim1, dvk_im + h_step / 2 * kdvim1, p2,
                dp2, kd2, a2, k);

  double p3 = phi + h_step / 2 * kp2;
  double dp3 = dphi + h_step / 2 * kd2;
  double a3 = a + h_step / 2 * ka2;
  kp3 = dp3;
  kd3 = ddphi_rhs(p3, dp3);
  ka3 = H(p3, dp3) * a3;
  kvre3 = dvk_re + h_step / 2 * kdvre2;
  kdvre3 = f_vk(vk_re + h_step / 2 * kvre2, dvk_re + h_step / 2 * kdvre2, p3,
                dp3, kd3, a3, k);
  kvim3 = dvk_im + h_step / 2 * kdvim2;
  kdvim3 = f_vk(vk_im + h_step / 2 * kvim2, dvk_im + h_step / 2 * kdvim2, p3,
                dp3, kd3, a3, k);

  double p4 = phi + h_step * kp3;
  double dp4 = dphi + h_step * kd3;
  double a4 = a + h_step * ka3;
  kp4 = dp4;
  kd4 = ddphi_rhs(p4, dp4);
  ka4 = H(p4, dp4) * a4;
  kvre4 = dvk_re + h_step * kdvre3;
  kdvre4 = f_vk(vk_re + h_step * kvre3, dvk_re + h_step * kdvre3, p4, dp4, kd4,
                a4, k);
  kvim4 = dvk_im + h_step * kdvim3;
  kdvim4 = f_vk(vk_im + h_step * kvim3, dvk_im + h_step * kdvim3, p4, dp4, kd4,
                a4, k);

  phi += (h_step / 6) * (kp1 + 2 * kp2 + 2 * kp3 + kp4);
  dphi += (h_step / 6) * (kd1 + 2 * kd2 + 2 * kd3 + kd4);
  a += (h_step / 6) * (ka1 + 2 * ka2 + 2 * ka3 + ka4);
  vk_re += (h_step / 6) * (kvre1 + 2 * kvre2 + 2 * kvre3 + kvre4);
  dvk_re += (h_step / 6) * (kdvre1 + 2 * kdvre2 + 2 * kdvre3 + kdvre4);
  vk_im += (h_step / 6) * (kvim1 + 2 * kvim2 + 2 * kvim3 + kvim4);
  dvk_im += (h_step / 6) * (kdvim1 + 2 * kdvim2 + 2 * kdvim3 + kdvim4);
}

int main() {
  std::vector<BgState> bg = load_background("background.dat");

  // sample uniformly in N space instead of index space
  int N_modes = 100;
  double N_end = bg[(int)bg.size() - 10].N;
  double Ne_min = 8.0;
  double Ne_max = N_end - bg[10].N - 3.0;

  std::cout << "N_end  = " << N_end << "\n";
  std::cout << "Ne_min = " << Ne_min << "\n";
  std::cout << "Ne_max = " << Ne_max << "\n";

  std::vector<int> exits;
  int prev = -1;
  for (int n = 0; n < N_modes; n++) {
    double Ne_target = Ne_min + n * (Ne_max - Ne_min) / (N_modes - 1);
    double N_target = N_end - Ne_target;
    for (int j = 10; j < (int)bg.size() - 10; j++) {
      if (bg[j].N >= N_target && j != prev) {
        exits.push_back(j);
        prev = j;
        break;
      }
    }
  }

  std::cout << "Sampled " << exits.size() << " modes.\n";

  int N = exits.size();
  std::vector<double> k_out(N), phi_out(N), dphi_out(N);
  std::vector<double> a_out(N), eps_out(N), vkre_out(N);
  std::vector<double> vkim_out(N), vksq_out(N), Ps_out(N, -1.0);
  double a_norm = bg[0].a;

  for (int idx = 0; idx < N; idx++) {
    int i_exit = exits[idx];
    double k_file = bg[i_exit].a * H(bg[i_exit].phi, bg[i_exit].dphi);

    int i_start = -1;
    double N_exit  = bg[i_exit].N;
	double N_start = N_exit - 5.0;
	for(int j = 0; j < i_exit; j++)
	{
    	if(bg[j].N >= N_start)
    	{ 
    		i_start = j; 
    		break; 
    	}
	}

    if (i_start < 0) {
      std::cout << "SKIP: k=" << k_file << " never sub-Hubble at ratio 50\n";
      continue;
    }

    
    double k = k_file;

    double phi = bg[i_start].phi;
    double dphi = bg[i_start].dphi;
    double a = bg[i_start].a ;

    double aH_start = a * H(phi, dphi);
    if (k < 10.0 * aH_start) {
      std::cout << "SKIP: k=" << k_file
                << " not sub-Hubble at i_start, k/aH=" << k / aH_start << "\n";
      continue;
    }

    double vk_re = 1.0 / std::sqrt(2.0 * k);
    double dvk_re = 0.0;
    double vk_im = 0.0;
    double dvk_im = -k / (a * std::sqrt(2.0 * k));

    double h_pert = 0.1*a / k;

    for (int i = i_start; i + 1 < (int)bg.size(); i++) {
      double h_bg = bg[i + 1].t - bg[i].t;
      double t_end = bg[i].t + h_bg;
      double t_curr = bg[i].t;

      while (t_curr < t_end) {
        double h = std::min(h_pert, t_end - t_curr);
        coupled_step(phi, dphi, a, vk_re, dvk_re, vk_im, dvk_im, k, h);
        t_curr += h;
        h_pert = 0.1 * a / k;
      }

      double aH = a * H(phi, dphi);
      if (k / aH < 0.01)
        break;
    }

    double aH_final = a * H(phi, dphi);
    if (k / aH_final > 0.01) {
      std::cout << "SKIP: k=" << k_file << " never froze, k/aH=" << k / aH_final
                << "\n";
      continue;
    }
    
    
    //check
    
    std::cout << "k=" << k_file 
          << " Ne=" << N_end - bg[i_exit].N
          << " i_start=" << i_start
          << " i_exit=" << i_exit
          << " bg_size=" << (int)bg.size()
          << " k/aH_freeze=" << k/aH_final << "\n";

    double h_end = H(phi, dphi);
    double eps = 0.5 * dphi * dphi / (h_end * h_end);
    //check slow roll params
    double phi_exit = bg[i_exit].phi;
	double H_exit   = H(bg[i_exit].phi, bg[i_exit].dphi);
	double eps_exit = 0.5*bg[i_exit].dphi*bg[i_exit].dphi/(H_exit*H_exit);
	double P_SR     = H_exit*H_exit / (8.0*M_PI*M_PI*eps_exit);

	
          
    //check ends
    double vk_sq = vk_re * vk_re + vk_im * vk_im;
    double P_zeta = (eps > 1e-10) ? (k * k * k) / (2.0 * M_PI * M_PI) * vk_sq /
                                        (2.0 * eps * a * a)
                                  : 0.0;
                                  
    std::cout << "k=" << k_file 
          << "  Ne=" << N_end - bg[i_exit].N
          << "  P_zeta=" << P_zeta 
          << "  P_SR=" << P_SR
          << "  ratio=" << P_zeta/P_SR << "\n";

    k_out[idx] = k_file;
    phi_out[idx] = phi;
    dphi_out[idx] = dphi;
    a_out[idx] = a;
    eps_out[idx] = eps;
    vkre_out[idx] = vk_re;
    vkim_out[idx] = vk_im;
    vksq_out[idx] = vk_sq;
    Ps_out[idx] = P_zeta;

    std::cout << "k=" << k_file << "  P_zeta=" << P_zeta << "\n";
  }

  std::ofstream out_ps("power_spectrum.dat");
  out_ps << "# k  phi  dphi  a  epsilon  vk_re  vk_im  |vk|^2  P_zeta\n";
  for (int idx = 0; idx < N; idx++) {
    if (Ps_out[idx] < 0)
      continue;
    out_ps << k_out[idx] << "  " << (N_end - bg[exits[idx]].N) << "  " <<  phi_out[idx] << "  " << dphi_out[idx]
           << "  " << a_out[idx] << "  " << eps_out[idx] << "  "
           << vkre_out[idx] << "  " << vkim_out[idx] << "  " << vksq_out[idx]
           << "  " << Ps_out[idx] << "\n";
  }
  out_ps.close();
  return 0;
}
