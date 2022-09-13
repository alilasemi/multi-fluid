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
        double uR, double g) {
    return fLR(p, pL, AL, BL, DL, cL, g) + fLR(p, pR, AR, BR, DR, cR, g)
            + uR - uL;
}

void compute_exact_riemann_problem(double rL, double pL, double uL, double rR,
        double pR, double uR, double g, double psg, vector_ref<double> result) {
    // TODO: Go through and make all the g's and psg's have a left and right!
    // Constants
    auto AL = 2 / ((g + 1) * rL);
    auto AR = 2 / ((g + 1) * rR);
    auto BL = ((g - 1) / (g + 1)) * pL;
    auto BR = ((g - 1) / (g + 1)) * pR;
    auto DL = (2 * g / (g + 1)) * psg;
    auto DR = (2 * g / (g + 1)) * psg;
    auto CL = DL / pL;
    auto CR = DR / pR;
    // Compute speed of sound
    auto cL = compute_c(g, pL, rL);
    auto cR = compute_c(g, pR, rR);

    // Solve nonlinear equation for pressure in the star region
    bool success = false;
    std::vector<double> guesses = {.25*pL + .75*pR, .5*(pL + pR), .75*pL + .25*pR, 0};
    double p_star;
    //std::vector<double> guesses = {.5*(pL + pR), 0};
    for (auto p : guesses) {
        double old_guess;
        int iter_max = 500;
        auto tol = 1e-8;
        for (int i = 0; i < iter_max; i++) {
            old_guess = p;
            // Compute RHS
            auto rhs = f(p, pL, pR, AL, AR, BL, BR, DL, DR, cL, cR, uL, uR, g);
            // Compute derivative
            auto delta_p = p * 1e-6;
            auto rhs_plus = f(p + delta_p, pL, pR, AL, AR, BL, BR, DL, DR, cL,
                    cR, uL, uR, g);
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
                << "rL, uL, pL, rR, uR, pR, g = "
                << rL << ", " << uL << ", " << pL << ", " << rR << ", "
                << uR << ", " << pR << ", " << g << endl;
        throw std::runtime_error(ss.str());
    }

    // Use this to get the velocity in the star region
    auto u_star = .5 * (uL + uR) + .5 * (fLR(p_star, pR, AR, BR, DR, cR, g)
            - fLR(p_star, pL, AL, BL, DL, cL, g));


    // Compute density
    double r_starL, r_starR;
    if (p_star > pL) {
        r_starL = r_star_shock(rL, p_star, pL, g, CL);
    } else {
        r_starL = r_star_expansion(rL, p_star, pL, g);
    }
    if (p_star > pR) {
        r_starR = r_star_shock(rR, p_star, pR, g, CR);
    } else {
        r_starR = r_star_expansion(rR, p_star, pR, g);
    }

    // Store result
    result[0] = p_star;
    result[1] = u_star;
    result[2] = r_starL;
    result[3] = r_starR;
}
