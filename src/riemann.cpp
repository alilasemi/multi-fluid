#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
using std::cout, std::endl;
using std::string;

#include <riemann.h>

// Compute speed of sound
double compute_c(double g, double p, double r, double psg) {
    return sqrt(g * (p + psg) / r);
}

// Density functions - from Toro
double r_star_shock(double r, double p_star, double p, double g, double C) {
    double r_star = r;
    r_star *= ((g - 1) / (g + 1)) + p_star / p + C;
    r_star /= ((g - 1) / (g + 1)) * p_star / p + 1 + C;
    return r_star;
}

double r_star_expansion(double r, double p_star, double p, double g, double psg) {
    return r * pow((p_star + psg) / (p + psg), 1/g);
}

double r_inside_expansion(double u, double rLR, double pLR, double g,
        double psg, double rRL) {
    //cout << "ahh " << u << ", " << rLR << ", " << pLR << ", " << g << ", " << psg << ", " << rRL << endl;
    int iter_max = 200;
    auto tol = 1e-6;
    // Initial guess
    double r_guess = .5 * (rLR + rRL);
    double r = r_guess;
    bool success = false;
    double r_min = fmin(rLR, rRL);
    double r_max = fmax(rLR, rRL);
    for (int i = 0; i < iter_max; i++) {
        // Store previous value
        r_guess = r;
        // Compute RHS
        auto rhs = r * pow(u, 2) / g - psg - pow(r, g) * pLR / pow(rLR, g);
        // Compute derivative
        auto d_rhs_d_r = pow(u, 2) / g - g * pow(r, g - 1) * pLR / pow(rLR, g);
        //cout << "r = " << r << endl;
        //cout << "rhs = " << rhs << endl;
        //cout << "d_rhs_d_p = " << d_rhs_d_r << endl;
        // Newton iteration
        r -= rhs / d_rhs_d_r;
        // Clip density to stay within the bounds
        if (r > r_max) {
            r = r_max;
        } else if (r < r_min) {
            r = r_min;
        }
        // Check convergence
        if (abs(r - r_guess) / (.5 * (r + r_guess)) < tol) {
            success = true;
            break;
        }
    }
    if (not success) {
        std::stringstream ss;
        ss << "Nonlinear solver for density inside expansion for Riemann problem failed!" << endl
                << "The inputs were:" << endl
                << "u, rLR, pLR, g, psg = "
                << u << ", " << rLR << ", " << pLR << ", " << g << ", " << psg
                << endl;
        throw std::runtime_error(ss.str());
    }
    return r;
}


// Pressure functions - from Toro
double fLR(double p, double rLR, double pLR, double ALR, double BLR, double DLR,
    double cLR, double g, double psg) {
    // For a shock
    if (p > pLR) {
        return (p - pLR) * sqrt(ALR / (p + BLR + DLR));
    // For an expansion
    } else {
        auto c_star_LR = compute_c(g, p, rLR * pow((p + psg) / (pLR + psg), 1 / g), psg);
        return (2 / (g - 1)) * (c_star_LR - cLR);
    }
}

double f(double p, double pL, double pR, double AL, double AR, double BL,
        double BR, double DL, double DR, double rL, double rR, double cL,
        double cR, double uL, double uR, double gL, double gR, double psgL,
        double psgR) {
    return fLR(p, rL, pL, AL, BL, DL, cL, gL, psgL)
            + fLR(p, rR, pR, AR, BR, DR, cR, gR, psgR)
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
    auto cL = compute_c(gL, pL, rL, psgL);
    auto cR = compute_c(gR, pR, rR, psgR);

    // Solve nonlinear equation for pressure in the star region
    bool success = false;
    std::vector<double> guesses = {.5*(pL + pR), .25*pL + .75*pR, .75*pL + .25*pR};
    double p_star;
    // Some interval information - from Toro
    double p_min = fmin(pL, pR);
    double p_max = fmax(pL, pR);
    auto f_min = f(p_min, pL, pR, AL, AR, BL, BR, DL, DR, rL, rR, cL, cR, uL,
            uR, gL, gR, psgL, psgR);
    auto f_max = f(p_max, pL, pR, AL, AR, BL, BR, DL, DR, rL, rR, cL, cR, uL,
            uR, gL, gR, psgL, psgR);
    for (auto p : guesses) {
        double old_guess;
        int iter_max = 500;
        auto tol = 1e-6;
        for (int i = 0; i < iter_max; i++) {
            old_guess = p;
            // Compute RHS
            auto rhs = f(p, pL, pR, AL, AR, BL, BR, DL, DR, rL, rR, cL, cR, uL,
                    uR, gL, gR, psgL, psgR);
            // Compute derivative
            auto delta_p = p * 1e-5;
            auto rhs_plus = f(p + delta_p, pL, pR, AL, AR, BL, BR, DL, DR, rL,
                    rR, cL, cR, uL, uR, gL, gR, psgL, psgR);
            auto rhs_minus = f(p - delta_p, pL, pR, AL, AR, BL, BR, DL, DR, rL,
                    rR, cL, cR, uL, uR, gL, gR, psgL, psgR);
            auto d_rhs_d_p = (rhs_plus - rhs_minus) / (2 * delta_p);
            // Newton iteration
            //cout << "p = " << p << endl;
            //cout << "rhs, rhs_plus/minus = " << rhs << ", " << rhs_plus << "  " << rhs_minus << endl;
            //cout << "d_rhs_d_p = " << d_rhs_d_p << endl;
            p -= .2 * rhs / d_rhs_d_p;
            //cout << "p_new = " << p << endl;
            // Bound pressure using bounds given in Toro, pg. 126
            if (f_min > 0 and f_max > 0) {
                if (p < 0) {
                    p = p_min * 1e-6;
                } else if (p > p_min) {
                    p = p_min;
                }
            } else if (f_min <= 0 and f_max >= 0) {
                if (p >= p_max) {
                    p = p_max;
                } else if (p <= p_min) {
                    p = p_min;
                }
            } else if (f_min < 0 and f_max < 0) {
                if (p < p_max) {
                    p = p_max;
                }
            }
            // Check convergence
            if (p > 0 and (abs(p - old_guess) / (.5 * (p + old_guess)) < tol)) {
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
        //TODO hack
        //cout << ss.str() << endl;
        throw std::runtime_error(ss.str());
    }

    // Use this to get the velocity in the star region
    auto u_star = .5 * (uL + uR) + .5 * (fLR(p_star, rR, pR, AR, BR, DR, cR, gR, psgR)
            - fLR(p_star, rL, pL, AL, BL, DL, cL, gL, psgL));


    // Compute density
    double r_starL, r_starR;
    if (p_star > pL) {
        r_starL = r_star_shock(rL, p_star, pL, gL, CL);
    } else {
        r_starL = r_star_expansion(rL, p_star, pL, gL, psgL);
    }
    if (p_star > pR) {
        r_starR = r_star_shock(rR, p_star, pR, gR, CR);
    } else {
        r_starR = r_star_expansion(rR, p_star, pR, gR, psgR);
    }
    //cout << p_star << "  " << u_star << "  " << r_starL << "  " << r_starR << endl;

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
            S = compute_shock_speed(rR, uR, p_star, AR, BR, DR, false);
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
            auto S_H = uR + cR;
            auto c_starR = compute_c(gR, p_star, r_starR, psgR);
            auto S_T = u_star + c_starR;
            // If the tail of the expansion has positive speed, then x/t = 0 is to
            // the left of the expansion, therefore U|_x/t=0 = U_starR.
            if (S_T > 0) {
                r_0 = r_starR;
                u_0 = u_star;
                p_0 = p_star;
            // Otherwise, if the head of the expansion has positive speed, then
            // x/t = 0 lies inside the expansion. In this case, use isentropic
            // relations and the Riemann invariant.
            } else if (S_H > 0) {
                u_0 = (2 / (gR + 1)) * (-cR + ((gR - 1) / 2) * uR);
                r_0 = r_inside_expansion(u_0, rR, pR, gR, psgR, r_starR);
                p_0 = r_0 * pow(u_0, 2) / gR - psgR;
            // Otherwise, if the head of the expansion has negative speed, then
            // x/t = 0 is to the right of the expansion.
            } else {
                r_0 = rR;
                u_0 = uR;
                p_0 = pR;
            }
        }
    // Otherwise, it's a left expansion
    } else {
        auto S_H = uL - cL;
        auto c_starL = compute_c(gL, p_star, r_starL, psgL);
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
            u_0 = (2 / (gL + 1)) * (cL + ((gL - 1) / 2) * uL);
            r_0 = r_inside_expansion(u_0, rL, pL, gL, psgL, r_starL);
            p_0 = r_0 * pow(u_0, 2) / gL - psgL;
        // Otherwise, if u_star is positive, then x/t = 0 is in between the
        // expansion and the contact.
        } else if (u_star > 0) {
            r_0 = r_starL;
            u_0 = u_star;
            p_0 = p_star;
        // If the expansion head and tail speed is negative and u_star is
        // negative, then it's between the contact and the rightmost wave. Need
        // to first find out if it's a right shock or not.
        } else if (p_star > pR) {
            // If the shock speed is positive, then x/t = 0 is to the left of
            // the shock, therefore U|_x/t=0 = U_starR.
            auto S = compute_shock_speed(rR, uR, p_star, AR, BR, DR, false);
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
            auto S_H = uR + cR;
            auto c_starR = compute_c(gR, p_star, r_starR, psgR);
            auto S_T = u_star + c_starR;
            // If the tail of the expansion has positive speed, then x/t = 0 is to
            // the left of the expansion, therefore U|_x/t=0 = U_starR.
            if (S_T > 0) {
                r_0 = r_starR;
                u_0 = u_star;
                p_0 = p_star;
            // Otherwise, if the head of the expansion has positive speed, then
            // x/t = 0 lies inside the expansion. In this case, use isentropic
            // relations and the Riemann invariant.
            } else if (S_H > 0) {
                u_0 = (2 / (gR + 1)) * (-cR + ((gR - 1) / 2) * uR);
                r_0 = r_inside_expansion(u_0, rR, pR, gR, psgR, r_starR);
                p_0 = r_0 * pow(u_0, 2) / gR - psgR;
            // Otherwise, if the head of the expansion has negative speed, then
            // x/t = 0 is to the right of the expansion.
            } else {
                r_0 = rR;
                u_0 = uR;
                p_0 = pR;
            }
        }
    }

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

    //TODO Hack to plot f vs p
    //auto p = result[1];
    //result[0] = f(p, pL, pR, AL, AR, BL, BR, DL, DR, rL, rR, cL, cR, uL,
    //        uR, gL, gR, psgL, psgR);
    //cout << fLR(p, rL, pL, AL, BL, DL, cL, gL, psgL) << "  "
    //    << fLR(p, rR, pR, AR, BR, DR, cR, gR, psgR) << endl;
    // Store result
    result[0] = r_0;
    result[1] = u_0;
    result[2] = p_0;
    result[3] = g_0;
    result[4] = psg_0;
    result[5] = r_starL;
    result[6] = r_starR;
    result[7] = u_star;
    result[8] = p_star;
}
