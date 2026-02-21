#ifndef RK4_H
#define RK4_H

#include <functional>

// ============================================================
//  rk4.h  —  Generic 4th-order Runge-Kutta ODE integrator
//
//  Uses fixed-size stack arrays via templates — no heap allocation.
//
//  STATE VECTOR CONVENTION (for chaotic inflation):
//    y[0] = phi   (inflaton field)
//    y[1] = dphi  (field velocity)
//    y[2] = N     (number of e-folds)
//
//  Usage:
//    #include "rk4.h"
//    double y[3] = {phi0, dphi0, 0.0};
//    for (int i = 0; i < N_steps; i++) {
//        write_output(i, y);
//        rk4_step<3>(y, h, equations_of_motion);
//    }
// ============================================================

// Derivative function signature: f(y, dydt) fills dydt in-place
template<int N>
using DerivFunc = std::function<void(const double y[N], double dydt[N])>;

// ---- Core RK4 step  (in-place update of y) -----------------

/**
 * Advances y[] by one step of size h using RK4.
 * Template parameter N is the number of state variables.
 * y[] is modified in-place.
 */
template<int N>
void rk4_step(double y[N], double h, const DerivFunc<N>& f)
{
    double k1[N], k2[N], k3[N], k4[N], tmp[N];

    // k1 = f(y)
    f(y, k1);

    // k2 = f(y + 0.5*h*k1)
    for (int i = 0; i < N; i++) tmp[i] = y[i] + 0.5*h*k1[i];
    f(tmp, k2);

    // k3 = f(y + 0.5*h*k2)
    for (int i = 0; i < N; i++) tmp[i] = y[i] + 0.5*h*k2[i];
    f(tmp, k3);

    // k4 = f(y + h*k3)
    for (int i = 0; i < N; i++) tmp[i] = y[i] + h*k3[i];
    f(tmp, k4);

    // y += (h/6) * (k1 + 2k2 + 2k3 + k4)
    for (int i = 0; i < N; i++)
        y[i] += (h/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

// ---- Convenience: integrate for a fixed number of steps ----

/**
 * Runs the integrator for N_steps steps, calling a callback
 * at every step for output or early stopping.
 *
 * @param y         State array (modified in-place)
 * @param h         Step size
 * @param N_steps   Number of steps
 * @param f         Derivative function
 * @param callback  Called as callback(step_index, y[]).
 *                  Return false to stop integration early.
 */
template<int N>
void rk4_integrate(
    double y[N],
    double h,
    int    N_steps,
    const  DerivFunc<N>& f,
    const  std::function<bool(int, const double[N])>& callback = nullptr)
{
    for (int i = 0; i < N_steps; i++) {
        if (callback && !callback(i, y))
            break;
        rk4_step<N>(y, h, f);
    }
}

#endif // RK4_H
