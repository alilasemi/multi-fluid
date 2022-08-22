#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
using std::cout, std::endl;
using std::string;

#include <unsupported/Eigen/NonLinearOptimization>

#include <riemann.h>

// Compute speed of sound
double compute_c(double g, double p, double r) {
    return sqrt(g * p / r);
}

// Density functions - from Toro
double r_star_shock(double r, double p_star, double p, double g) {
    double r_star = r;
    r_star *= ((g - 1) / (g + 1)) + p_star / p;
    r_star /= ((g - 1) / (g + 1)) * p_star / p + 1;
    return r_star;
}

double r_star_expansion(double r, double p_star, double p, double g) {
    return r * pow(p_star / p, 1/g);
}

// Pressure functions - from Toro
double fLR(double p, double pLR, double ALR, double BLR, double cLR, double g) {
    if (p > pLR) {
        return (p - pLR) * sqrt(ALR / (p + BLR));
    } else {
        return (2 * cLR / (g - 1)) * ( pow(p/pLR, (g - 1) / (2*g)) - 1 );
    }
}

double f(double p, double pL, double pR, double AL, double AR, double BL,
        double BR, double cL, double cR, double uL, double uR, double g) {
    return fLR(p, pL, AL, BL, cL, g) + fLR(p, pR, AR, BR, cR, g) + uR - uL;
}

// A functor that computes the RHS of the nonlinear pressure equation in the
// Riemann problem.
struct PressureFunctor {
    // ----- Info needed for Eigen's nonlinear solver ----- #
    using Scalar = double;
    enum {
      InputsAtCompileTime = 1,
      ValuesAtCompileTime = 1
    };
    using InputType = Scalar;
    using ValueType = Scalar;
    using JacobianType = Scalar;
    int inputs() const { return InputsAtCompileTime; }
    int values() const { return ValuesAtCompileTime; }

    // ----- Inputs that define the Riemann problem ----- #
    double pL; double pR; double AL; double AR; double BL;
    double BR; double cL; double cR; double uL; double uR; double g;

    // Set the input values
    PressureFunctor(double _pL, double _pR, double _AL, double _AR, double _BL,
            double _BR, double _cL, double _cR, double _uL, double _uR, double _g) :
            pL(_pL), pR(_pR), AL(_AL), AR(_AR), BL(_BL),
            BR(_BR), cL(_cL), cR(_cR), uL(_uL), uR(_uR), g(_g) {}

    // Compute the RHS of the nonlinear pressure equation
    int operator() (const vector<double>& p_vec, vector<double>& output) const {
        auto& p = p_vec(0);
        output(0) = f(p, pL, pR, AL, AR, BL, BR, cL, cR, uL, uR, g);
        cout << "f = " << output(0) << endl;
        return 0;
    }
};

void compute_exact_riemann_problem(double rL, double pL, double uL, double rR,
        double pR, double uR, double g, vector_ref<double> result) {
    // Constants
    auto AL = 2 / ((g + 1) * rL);
    auto AR = 2 / ((g + 1) * rR);
    auto BL = ((g - 1) / (g + 1)) * pL;
    auto BR = ((g - 1) / (g + 1)) * pR;
    // Compute speed of sound
    auto cL = compute_c(g, pL, rL);
    auto cR = compute_c(g, pR, rR);

    // Solve nonlinear equation for pressure in the star region
    vector<double> guess(1);
    bool success = false;
    int info;
    std::vector<double> guesses = {.25*pL + .75*pR, .5*(pL + pR), .75*pL + .25*pR, 0};
    //std::vector<double> guesses = {.5*(pL + pR), 0};
    for (auto& guess_value : guesses) {
        guess << guess_value;
        PressureFunctor p_functor(pL, pR, AL, AR, BL, BR, cL, cR, uL, uR, g);
        Eigen::NumericalDiff<PressureFunctor> func_with_num_diff(p_functor);
        Eigen::HybridNonLinearSolver<Eigen::NumericalDiff<PressureFunctor> > solver(func_with_num_diff);
        info = solver.hybrd1(guess, 1e-16);
        cout << guess_value << "  " << info << endl;
        if (info == 1) {
            success = true;
            break;
        }
    }
    // Make sure that the solver did not fail
    if (not success) {
        if (info != 1) {
            std::stringstream ss;
            ss << "Nonlinear solver in Riemann problem failed! Error code = "
                    << info << endl
                    << "The inputs were:" << endl
                    << "rL, uL, pL, rR, uR, pR, g = "
                    << rL << ", " << uL << ", " << pL << ", " << rR << ", "
                    << uR << ", " << pR << ", " << g << endl;
            throw std::runtime_error(ss.str());
        }
    }
    auto p_star = guess(0);

    // Use this to get the velocity in the star region
    auto u_star = .5 * (uL + uR) + .5 * (fLR(p_star, pR, AR, BR, cR, g) - fLR(p_star, pL, AL, BL, cL, g));


    // Compute density
    double r_starL, r_starR;
    if (p_star > pL) {
        r_starL = r_star_shock(rL, p_star, pL, g);
    } else {
        r_starL = r_star_expansion(rL, p_star, pL, g);
    }
    if (p_star > pR) {
        r_starR = r_star_shock(rR, p_star, pR, g);
    } else {
        r_starR = r_star_expansion(rR, p_star, pR, g);
    }

    // Store result
    result[0] = p_star;
    result[1] = u_star;
    result[2] = r_starL;
    result[3] = r_starR;
}
