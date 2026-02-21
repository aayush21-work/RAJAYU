#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include "rk4.h"




// ---- Parameters --------------------------------------------
const double N_STAR = 60.0;

// ---- Background data struct --------------------------------
struct BkgPoint {
    double t, tau, phi, dphi, a, H, zppz, N;
};

// ---- Load background.dat -----------------------------------
std::vector<BkgPoint> load_background(const std::string& fname)
{
    std::vector<BkgPoint> data;
    std::ifstream f(fname);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open " << fname << "\n";
        std::exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        BkgPoint p;
        std::istringstream ss(line);
        ss >> p.t >> p.tau >> p.phi >> p.dphi >> p.a >> p.H >> p.zppz >> p.N;
        data.push_back(p);
    }
    std::cout << "Loaded " << data.size() << " background points.\n";
    return data;
}

// ---- Linear interpolation of background at a given t -------
void interp_background(const std::vector<BkgPoint>& bkg,
                       double t_query,
                       double& a_out, double& H_out, double& zppz_out,
                       double& tau_out, double& phi_out, double& dphi_out)
{
    // Binary search for the right interval
    int lo = 0, hi = (int)bkg.size() - 2;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (bkg[mid].t < t_query) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= (int)bkg.size() - 1) lo = (int)bkg.size() - 2;
    if (lo < 0) lo = 0;

    double t0 = bkg[lo].t, t1 = bkg[lo+1].t;
    double alpha = (t_query - t0) / (t1 - t0);
    auto lerp = [&](double v0, double v1) { return v0 + alpha*(v1-v0); };

    a_out    = lerp(bkg[lo].a,    bkg[lo+1].a);
    H_out    = lerp(bkg[lo].H,    bkg[lo+1].H);
    zppz_out = lerp(bkg[lo].zppz, bkg[lo+1].zppz);
    tau_out  = lerp(bkg[lo].tau,  bkg[lo+1].tau);
    phi_out  = lerp(bkg[lo].phi,  bkg[lo+1].phi);
    dphi_out = lerp(bkg[lo].dphi, bkg[lo+1].dphi);
}

double compute_z(double a, double dphi, double H) { return a*dphi/H; }

static const int PNDIM = 4;

double solve_mode(double k,
                  const std::vector<BkgPoint>& bkg,
                  double h_step,
                  double horizon_entry_ratio = 50.0,   
                  double horizon_exit_ratio  = 0.01)  
{
     int i_start = -1;
    for (int i = 0; i < (int)bkg.size()-1; i++) {
        double aH = bkg[i].a * bkg[i].H;
        if (k / aH <= horizon_entry_ratio) {
            i_start = i;
            break;
        }
    }
    if (i_start < 0) {
        std::cerr << "  k=" << k << " never inside horizon at ratio "
                  << horizon_entry_ratio << " — skipping.\n";
        return -1.0;
    }

    

    int i_end = -1;
    for (int i = i_start; i < (int)bkg.size(); i++) {
        double aH = bkg[i].a * bkg[i].H;
        if (k / aH <= horizon_exit_ratio) {
            i_end = i;
            break;
        }
    }
    if (i_end < 0) {
        std::cerr << "  k=" << k << " never fully exits horizon — skipping.\n";
        return -1.0;
    }

    double t_i = bkg[i_start].t;
    double t_f = bkg[i_end].t;


    double tau_i      = bkg[i_start].tau;
    double a_i        = bkg[i_start].a;
    double inv_sqrt2k = 1.0 / std::sqrt(2.0*k);
    double ktau_i     = k * tau_i;

    double y[PNDIM];
    y[0] =  std::cos(ktau_i) * inv_sqrt2k;              
    y[1] = -std::sin(ktau_i) * inv_sqrt2k;              
    y[2] = -k * std::sin(ktau_i) * inv_sqrt2k / a_i;    
    y[3] = -k * std::cos(ktau_i) * inv_sqrt2k / a_i;    

    double t_curr = t_i;
    int    n_pert = (int)((t_f - t_i) / h_step) + 1;
    double k2     = k*k;

    DerivFunc<PNDIM> ms_eom = [&](const double yp[PNDIM], double dypdt[PNDIM])
    {
        double a_loc, H_loc, zppz_loc, tau_loc, phi_loc, dphi_loc;
        interp_background(bkg, t_curr, a_loc, H_loc, zppz_loc, tau_loc, phi_loc, dphi_loc);

        double omega2 = (k2 - zppz_loc) / (a_loc*a_loc);

        dypdt[0] =  yp[2];
        dypdt[1] =  yp[3];
        dypdt[2] = -H_loc*yp[2] - omega2*yp[0];
        dypdt[3] = -H_loc*yp[3] - omega2*yp[1];
    };

    for (int step = 0; step < n_pert; step++) {
        rk4_step<PNDIM>(y, h_step, ms_eom);
        t_curr += h_step;
        if (t_curr > t_f) break;
    }

    double a_f, H_f, zppz_f, tau_f, phi_f, dphi_f;
    interp_background(bkg, t_f, a_f, H_f, zppz_f, tau_f, phi_f, dphi_f);

    double z_f = compute_z(a_f, dphi_f, H_f);
    double vk2 = y[0]*y[0] + y[1]*y[1];
    double Ps  = (k2*k) / (2.0*M_PI*M_PI) * vk2 / (z_f*z_f);

    return Ps;
}




double find_k_star(const std::vector<BkgPoint>& bkg)
{
    double N_end  = bkg.back().N;
    double N_star = N_end - N_STAR;

    if (N_star < 0) {
        std::cerr << "Warning: total e-folds (" << N_end
                  << ") < N_STAR (" << N_STAR << "). "
                  << "Reduce N_STAR or run background longer.\n";
        N_star = N_end / 2.0;
    }

   
    int i_star = 0;
    for (int i = 0; i < (int)bkg.size()-1; i++) {
        if (bkg[i].N >= N_star) { i_star = i; break; }
    }

    double k_star = bkg[i_star].a * bkg[i_star].H;
    std::cout << "N_end  = " << N_end  << " e-folds\n"
              << "N_star = " << N_star << " e-folds  (N★ = N_end - " << N_STAR << ")\n"
              << "k_star = " << k_star << " (Planck units)\n\n";
    return k_star;
}


void report_cmb_observables(const std::vector<double>& k_vals,
                             const std::vector<double>& Ps_vals,
                             int valid,
                             double k_star)
{
    if (valid < 2) {
        std::cerr << "Not enough valid modes to interpolate at k_star.\n";
        return;
    }

    
    if (k_star < k_vals[0] || k_star > k_vals[valid-1]) {
        std::cerr << "Warning: k_star=" << k_star
                  << " is outside sampled k range ["
                  << k_vals[0] << ", " << k_vals[valid-1] << "]\n"
                  << "The k range in main() is automatically set to bracket k_star.\n"
                  << "If you see this, check N_STAR or background total e-folds.\n";
        return;
    }

    double logk_star = std::log(k_star);

    for (int n = 1; n < valid; n++) {
        if (std::log(k_vals[n]) >= logk_star) {

            
            double alpha = (logk_star          - std::log(k_vals[n-1]))
                         / (std::log(k_vals[n]) - std::log(k_vals[n-1]));
            double logPs_star = std::log(Ps_vals[n-1])
                              + alpha*(std::log(Ps_vals[n]) - std::log(Ps_vals[n-1]));
            double As = std::exp(logPs_star);

            // n_s = 1 + d ln P_s / d ln k
            double ns = 1.0 + (std::log(Ps_vals[n]) - std::log(Ps_vals[n-1]))
                            / (std::log(k_vals[n])  - std::log(k_vals[n-1]));

            std::cout  << "  A_s  = " << As  << "\n"  << "  n_s  = " << ns  << "\n" ;
            break;
        }
    }
}

// ---- Main --------------------------------------------------
int main()
{
    // Load background
    auto bkg = load_background("background.dat");
    if (bkg.size() < 2) {
        std::cerr << "Not enough background data.\n";
        return 1;
    }

    double h_bkg = bkg[1].t - bkg[0].t;

    
    double k_star = find_k_star(bkg);

    int    N_modes = 40;
    double k_min   = k_star * 1e-2;
    double k_max   = k_star * 1e2;

    std::cout << "Sampling k in [" << k_min << ", " << k_max << "]\n"
              << "Solving " << N_modes << " modes...\n\n";

    std::ofstream out("power_spectrum.dat");
    out << "# k   P_s(k)   n_s(k)\n";

    std::vector<double> k_vals(N_modes), Ps_vals(N_modes);
    int valid = 0;

    for (int n = 0; n < N_modes; n++)
    {
        double log_k = std::log(k_min) + n * (std::log(k_max) - std::log(k_min)) / (N_modes-1);
        double k     = std::exp(log_k);

        std::cout << "  k = " << k << " ... " << std::flush;
        double Ps = solve_mode(k, bkg, h_bkg);

        if (Ps > 0) {
            k_vals[valid]  = k;
            Ps_vals[valid] = Ps;
            valid++;
            std::cout << "P_s = " << Ps << "\n";
        }
    }


    // Write out
    for (int n = 0; n < valid; n++) {
        double ns = 0.0;
        if (n > 0 && n < valid-1) {
            ns = 1.0 + (std::log(Ps_vals[n+1]) - std::log(Ps_vals[n-1]))
                     / (std::log(k_vals[n+1])  - std::log(k_vals[n-1]));
        }
        out << k_vals[n] << " " << Ps_vals[n] << " " << ns << "\n";
    }

    std::cout << "\nFull power spectrum saved to power_spectrum.dat\n";

    // Compare to Planck
    report_cmb_observables(k_vals, Ps_vals, valid, k_star);

    return 0;
}
