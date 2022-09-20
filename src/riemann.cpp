#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
using std::cout, std::endl;
using std::string;

#include <riemann.h>

// Compute speed of sound
double compute_c(double g, double p, double r) {
    return sqrt(g * p / r);
}

// Density functions - from Toro
double r_star_shock(double r, double p_star, double p, double g, double C) {
    double r_star = r;
    r_star *= ((g - 1) / (g + 1)) + p_star / p + C;
    r_star /= ((g - 1) / (g + 1)) * p_star / p + 1 + C;
    return r_star;
}

double r_star_expansion(double r, double p_star, double p, double g) {
    return r * pow(p_star / p, 1/g);
}

// Pressure functions - from Toro
double fLR(double p, double pLR, double ALR, double BLR, double DLR, double cLR,
        double g) {
    if (p > pLR) {
        return (p - pLR) * sqrt(ALR / (p + BLR + DLR));
    } else {
        return (2 * cLR / (g - 1)) * ( pow(p/pLR, (g - 1) / (2*g)) - 1 );
    }
}

double f(double p, double pL, double pR, double AL, double AR, double BL,
        double BR, double DL, double DR, double cL, double cR, double uL,
        double uR, double gL, double gR) {
    return fLR(p, pL, AL, BL, DL, cL, gL) + fLR(p, pR, AR, BR, DR, cR, gR)
            + uR - uL;
}

// See Toro, pg. 134.
double compute_shock_speed(double r, double u, double p_star, double A,
        double B, double D, bool left) {
    auto Q = sqrt((p_star + B + D) / A);
    if (left) {
        return u - Q / r;
    } else {
        return u + Q / r;
    }
}

void compute_exact_riemann_problem(double rL, double pL, double uL, double rR,
        double pR, double uR, double gL, double gR, double psgL, double psgR,
        vector_ref<double> result) {
    // TODO: Go through and make all the g's and psg's have a left and right!
    // Constants
    auto AL = 2 / ((gL + 1) * rL);
    auto AR = 2 / ((gR + 1) * rR);
    auto BL = ((gL - 1) / (gL + 1)) * pL;
    auto BR = ((gR - 1) / (gR + 1)) * pR;
    auto DL = (2 * gL / (gL + 1)) * psgL;
    auto DR = (2 * gR / (gR + 1)) * psgR;
    auto CL = DL / pL;
    auto CR = DR / pR;
    // Compute speed of sound
    auto cL = compute_c(gL, pL, rL);
    auto cR = compute_c(gR, pR, rR);

    // Solve nonlinear equation for pressure in the star region
    bool success = false;
    std::vector<double> guesses = {.25*pL + .75*pR, .5*(pL + pR), .75*pL + .25*pR, 0};
    double p_star;
    for (auto p : guesses) {
        double old_guess;
        int iter_max = 500;
        auto tol = fmax(pL, pR) * 1e-6;
        for (int i = 0; i < iter_max; i++) {
            old_guess = p;
            // Compute RHS
            auto rhs = f(p, pL, pR, AL, AR, BL, BR, DL, DR, cL, cR, uL, uR, gL, gR);
            // Compute derivative
            auto delta_p = p * 1e-6;
            auto rhs_plus = f(p + delta_p, pL, pR, AL, AR, BL, BR, DL, DR, cL,
                    cR, uL, uR, gL, gR);
            auto d_rhs_d_p = (rhs_plus - rhs) / delta_p;
            // Newton iteration
            p -= .2 * rhs / d_rhs_d_p;
            // Check convergence
            if (abs(p - old_guess) < tol) {
                success = true;
                break;
            }
        }
        if (success) {
            p_star = p;
            break;
        }
    }
    // Make sure that the solver did not fail
    if (not success) {
        std::stringstream ss;
        ss << "Nonlinear solver in Riemann problem failed!" << endl
                << "The inputs were:" << endl
                << "rL, uL, pL | rR, uR, pR | gL/gR = "
                << rL << ", " << uL << ", " << pL << " | " << rR << ", "
                << uR << ", " << pR << " | " << gL << "/" << gR << ", " << endl;
        throw std::runtime_error(ss.str());
    }

    // Use this to get the velocity in the star region
    auto u_star = .5 * (uL + uR) + .5 * (fLR(p_star, pR, AR, BR, DR, cR, gR)
            - fLR(p_star, pL, AL, BL, DL, cL, gL));


    // Compute density
    double r_starL, r_starR;
    if (p_star > pL) {
        r_starL = r_star_shock(rL, p_star, pL, gL, CL);
    } else {
        r_starL = r_star_expansion(rL, p_star, pL, gL);
    }
    if (p_star > pR) {
        r_starR = r_star_shock(rR, p_star, pR, gR, CR);
    } else {
        r_starR = r_star_expansion(rR, p_star, pR, gR);
    }

    // Now the star state is known, so choose the solution at x/t = 0.
    double r_0;
    double u_0;
    double p_0;
    // Check for a left shock
    if (p_star > pL) {
        // If the shock speed is positive, then x/t = 0 is to the left of the
        // shock, therefore U|_x/t=0 = U_L.
        auto S = compute_shock_speed(rL, uL, p_star, AL, BL, DL, true);
        if (S > 0) {
            r_0 = rL;
            u_0 = uL;
            p_0 = pL;
        // If the shock speed is negative and u_star is positive, then x/t = 0
        // is in between the shock and the contact.
        } else if (u_star > 0) {
            r_0 = r_starL;
            u_0 = u_star;
            p_0 = p_star;
        // If the shock speed is negative and u_star is negative, then it's
        // between the contact and the rightmost wave. Need to first find out
        // if it's a right shock or not.
        } else if (p_star > pR) {
            // If the shock speed is positive, then x/t = 0 is to the left of
            // the shock, therefore U|_x/t=0 = U_starR.
            S = compute_shock_speed(rR, uR, p_star, AL, BL, DL, false);
            if (S > 0) {
                r_0 = r_starR;
                u_0 = u_star;
                p_0 = p_star;
            // Otherwise, then x/t = 0 is the right of the shock.
            } else {
                r_0 = rR;
                u_0 = uR;
                p_0 = pR;
            }
        // Otherwise, it's a right expansion
        } else {
    // Otherwise, it's a left expansion
    } else {
        auto S_H = uL - cL;
        auto c_starL = sqrt(gL * p_star / r_starL);
        auto S_T = u_star - c_starL;
        // If the head of the expansion has positive speed, then x/t = 0 is to
        // the left of the expansion, therefore U|_x/t=0 = U_L.
        if (S_H > 0) {
            r_0 = rL;
            u_0 = uL;
            p_0 = pL;
        // Otherwise, if the tail of the expansion has positive speed, then
        // x/t = 0 lies inside the expansion. In this case, use isentropic
        // relations and the Riemann invariant.
        } else if (S_T > 0) {


    // Select the fluid ID in the middle (at x/t = 0)
    double g_0;
    double psg_0;
    if (u_star > 0) {
        g_0 = gL;
        psg_0 = psgL;
    } else {
        g_0 = gR;
        psg_0 = psgR;
    }

    // Store result
    result[0] = r_0;
    result[1] = u_0;
    result[2] = p_0;
    result[3] = g_0;
    result[4] = psg_0;
}
