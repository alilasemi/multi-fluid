#include <iostream>
using std::cout;
using std::endl;

#include <cache/compute_A_RL.cpp>
#include <cache/compute_Lambda.cpp>
#include <cache/compute_Q_inv.cpp>
#include <cache/compute_Q.cpp>
#include <roe.h>
#include <riemann.h>
#include <face_residual.h>


matrix<double> convective_fluxes(matrix_ref<double> U, double g, double psg) {
    // Unpack
    auto r =  U(0, 0);
    auto ru = U(1, 0);
    auto rv = U(2, 0);
    auto re = U(3, 0);
    auto p = (g - 1) * (re - .5 * (ru*ru + rv*rv) / r) - g * psg;
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

void compute_flux_roe(matrix_ref<double> U_L,
        matrix_ref<double> U_R, matrix<double>& area_normal, double g,
        vector_ref<double> F) {
    // Unit normals
    auto length = area_normal.norm();
    auto unit_normals = area_normal / length;
    auto nx = unit_normals(0);
    auto ny = unit_normals(1);

    // Convert to primitives
    auto rL = U_L(0, 0);
    auto rR = U_R(0, 0);
    auto uL = U_L(1, 0) / rL;
    auto uR = U_R(1, 0) / rR;
    auto vL = U_L(2, 0) / rL;
    auto vR = U_R(2, 0) / rR;
    auto hL = (U_L(3, 0) - (1/(2*g))*(g - 1)*rL*(pow(uL, 2) + pow(vL, 2))) * g / rL;
    auto hR = (U_R(3, 0) - (1/(2*g))*(g - 1)*rR*(pow(uR, 2) + pow(vR, 2))) * g / rR;

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
    // TODO
    cout << "Roe not implemented for stiffened gas!!" << endl;
    F = length * (
            .5 * (convective_fluxes(U_L, g, 0) + convective_fluxes(U_R, g, 0)) * unit_normals
            - .5 * (abs_A_RL * (U_R - U_L)));
}

void compute_flux(matrix_ref<double> U_L, matrix_ref<double> U_R,
        matrix<double>& area_normal, double gL, double gR, double psgL,
        double psgR, vector_ref<double> F) {
    // Package the normals
    double length = area_normal.norm();
    vector<double> n_hat = area_normal(all, 0).normalized();

    // Convert left and right to primitive
    auto V_L = conservative_to_primitive(U_L, gL, psgL);
    auto V_R = conservative_to_primitive(U_R, gR, psgR);
    // Get normal/tangential component of velocity
    auto u_n_L = V_L(seq(1, 2)).dot(n_hat);
    auto u_n_R = V_R(seq(1, 2)).dot(n_hat);
    // Solve exact fluid-fluid Riemann problem
    auto r = vector<double>(2);
    auto u_n = vector<double>(2);
    auto p = vector<double>(2);
    vector<double> result(9);
    compute_exact_riemann_problem(V_L(0), V_L(3), u_n_L, V_R(0), V_R(3),
            u_n_R, gL, gR, psgL, psgR, result);
    auto& r_0 = result(0);
    auto& u_0 = result(1);
    auto& p_0 = result(2);
    auto& g_0 = result(3);
    auto& psg_0 = result(4);
    // Convert back to conservative
    vector<double> V_0(4);
    V_0 << r_0, u_0 * n_hat(0), u_0 * n_hat(1), p_0;
    matrix<double> U_0 = primitive_to_conservative(V_0, g_0, psg_0);
    // Compute flux
    matrix<double> full_F = convective_fluxes(U_0, g_0, psg_0);
    F = full_F * area_normal;
}
