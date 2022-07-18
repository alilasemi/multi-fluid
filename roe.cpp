//#include <roe.h>
#include <iostream>
using std::cout;
using std::endl;

#include <defines.h>
#include <cache/compute_A_RL.cpp>
#include <cache/compute_Lambda.cpp>
#include <cache/compute_Q_inv.cpp>
#include <cache/compute_Q.cpp>


matrix<double> convective_fluxes(matrix<double> U, double g) {
    // Unpack
    auto r =  U(0);
    auto ru = U(1);
    auto rv = U(2);
    auto re = U(3);
    auto p = (re - .5 * (pow(ru, 2) + pow(rv, 2)) / r) * (g - 1);
    // Compute flux
    auto F = matrix<double>(4, 2);
    F(0, 0) = ru;
    F(1, 0) = pow(ru, 2) / r + p;
    F(2, 0) = ru*rv / r;
    F(3, 0) = (re + p) * ru / r;
    F(0, 1) = rv;
    F(1, 1) = ru*rv / r;
    F(2, 1) = pow(rv, 2) / r + p;
    F(3, 1) = (re + p) * rv / r;
    return F;
}

void compute_flux(matrix<double>& U_L,
        matrix<double>& U_R, matrix<double>& area_normal, double g,
        vector<double>& F) {
    // Unit normals
    auto length = area_normal.norm();
    auto unit_normals = area_normal / length;
    auto nx = unit_normals(0);
    auto ny = unit_normals(1);

    // Convert to primitives
    auto rL = U_L(0);
    auto rR = U_R(0);
    auto uL = U_L(1) / rL;
    auto uR = U_R(1) / rR;
    auto vL = U_L(2) / rL;
    auto vR = U_R(2) / rR;
    auto hL = (U_L(3) - (1/(2*g))*(g - 1)*rL*(pow(uL, 2) + pow(vL, 2))) * g / rL;
    auto hR = (U_R(3) - (1/(2*g))*(g - 1)*rR*(pow(uR, 2) + pow(vR, 2))) * g / rR;

    // The RL state
    auto uRL = (sqrt(rR) * uR + sqrt(rL) * uL) / (sqrt(rR) + sqrt(rL));
    auto vRL = (sqrt(rR) * vR + sqrt(rL) * vL) / (sqrt(rR) + sqrt(rL));
    auto hRL = (sqrt(rR) * hR + sqrt(rL) * hL) / (sqrt(rR) + sqrt(rL));

    // Compute A_RL
    auto A_RL = matrix<double>(4, 4);
    compute_A_RL(uRL, vRL, hRL, nx, ny, g, A_RL.data());
    // Compute eigendecomp
    auto Lambda = matrix<double>(4, 4);
    compute_Lambda(uRL, vRL, hRL, nx, ny, g, Lambda.data());
    auto Q_inv = matrix<double>(4, 4);
    compute_Q_inv(uRL, vRL, hRL, nx, ny, g, Q_inv.data());
    auto Q = matrix<double>(4, 4);
    compute_Q(uRL, vRL, hRL, nx, ny, g, Q.data());

    auto Lambda_m = (Lambda - Lambda.cwiseAbs())/2;
    auto Lambda_p = Lambda - Lambda_m;
    auto A_RL_m = Q_inv * Lambda_m * Q;
    auto A_RL_p = Q_inv * Lambda_p * Q;
    auto abs_A_RL = A_RL_p - A_RL_m;

    // Compute flux
    F = length * (
            .5 * (convective_fluxes(U_L, g) + convective_fluxes(U_R, g)) * unit_normals
            - .5 * (abs_A_RL * (U_R - U_L)));
}

PYBIND11_MODULE(roe, m) {
    m.doc() = "doc"; // optional module docstring
    m.def("compute_flux", &compute_flux,
            "A function that computes...");
}
