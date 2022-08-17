#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
using std::cout, std::endl;
using std::string;

#include <unsupported/Eigen/NonLinearOptimization>

#include <defines.h>
#include <face_residual.h>
#include <forced_interface.h>
#include <roe.h>


// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(matrix_ref<double> U,
        matrix_ref<double> U_L, matrix_ref<double> U_R,
        vector_ref<long> interior_face_IDs, matrix_ref<long> edge,
        matrix_ref<double> limiter, std::vector<double> gradU,
        matrix_ref<double> xy, matrix_ref<double> area_normals_p1,
        matrix_ref<double> area, double g, matrix_ref<double> residual) {
    // Sizing
    auto n_faces = U_L.rows();
    // Create buffers
    vector<double> F(4);
    // Loop over all faces
    for (int face_ID = 0; face_ID < n_faces; face_ID++) {
        // Left and right cell IDs
        auto L = edge(face_ID, 0);
        auto R = edge(face_ID, 1);

        // Gradients for these cells
        matrix_map<double> gradU_L(&gradU[L*4*2], 4, 2);
        matrix_map<double> gradU_R(&gradU[R*4*2], 4, 2);

        // Evaluate solution at faces on left and right
        // -- First order component -- #
        U_L(face_ID, all) = U(L, all);
        U_R(face_ID, all) = U(R, all);

        // -- Second order component -- #
        matrix<double> edge_point = .5 * (xy(L, all) + xy(R, all)).transpose();
        // TODO: There has to be a cleaner way...
        for (int k = 0; k < 4; k++) {
            U_L(face_ID, k) += limiter(L, k) * (gradU_L(k, all).transpose().cwiseProduct(edge_point - xy(L, all).transpose()).sum());
            U_R(face_ID, k) += limiter(R, k) * (gradU_R(k, all).transpose().cwiseProduct(edge_point - xy(R, all).transpose()).sum());
        }
    }

    // Loop over interior faces
    for (const auto face_ID : interior_face_IDs) {
        // Left and right cell IDs
        auto L = edge(face_ID, 0);
        auto R = edge(face_ID, 1);
        // Evaluate fluxes
        matrix<double> area_normals = area_normals_p1(face_ID, all).transpose();
        compute_flux(U_L(face_ID, all).transpose(),
                U_R(face_ID, all).transpose(), area_normals, g, F);

        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L, 0) * F;
        residual(R, all) +=  1 / area(R, 0) * F;
    }
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
    const double u1; const double u4;
    const double c1; const double c4;
    const double p1; const double p4;
    const double g;

    // Set the input values
    PressureFunctor(double _u1, double _u4, double _c1, double _c4,
            double _p1, double _p4, double _g) : u1(_u1), u4(_u4), c1(_c1),
            c4(_c4), p1(_p1), p4(_p4), g(_g) {}

    // Compute the RHS of the nonlinear pressure equation
    int operator() (const vector<double>& p2p1_vec, vector<double>& output) const {
        auto& p2p1 = p2p1_vec(0);
        output(0) = p2p1 * pow(
            1 + (g - 1) / (2 * c4) * (
                u4 - u1 - (c1/g) * (
                    (p2p1 - 1) /
                    sqrt(((g+1) / (2 * g)) * (p2p1 - 1) + 1)
                )
            ), (-(2 * g) / (g - 1))) - p4/p1;
        return 0;
    }
};


void exact_riemann_problem(double r4, double u4, double p4, double r1,
        double u1, double p1, double g, double xL, double xR, double t,
        vector<double>& r, vector<double>& u, vector<double>& p) {
    // Points at which to evaluate Riemann problem
    Eigen::Array<double, 2, 1> x;
    x << xL, xR;

    // Compute speed of sound
    auto compute_c = [&](double pressure, double rho) {
        return sqrt(g * pressure / rho);
    };
    auto c1 = compute_c(p1, r1);
    auto c4 = compute_c(p4, r4);

    //TODO Hack, for checking: set V_L_Riemann to V_L, same for right
    //r[0] = r4;
    //r[1] = r1;
    //u[0] = u4;
    //u[1] = u1;
    //p[0] = p4;
    //p[1] = p1;
    //return;

    vector<double> guess(1);
    if (p1 > p4) {
        guess << p4/p1;
    } else {
        guess << p1/p4;
    }

    bool success = false;
    int info;
    // TODO: Improve guesses, maybe a fit for this?
    vector<double> guesses = vector<double>::LinSpaced(100, 0, std::max(p1/p4, p4/p1));
    //vector<double> guesses = vector<double>::LinSpaced(1, guess(0), guess(0));
    for (auto& guess_value : guesses) {
        guess << guess_value;
        PressureFunctor p_functor(u1, u4, c1, c4, p1, p4, g);
        Eigen::NumericalDiff<PressureFunctor> func_with_num_diff(p_functor);
        Eigen::HybridNonLinearSolver<Eigen::NumericalDiff<PressureFunctor> > solver(func_with_num_diff);
        info = solver.hybrd1(guess);
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
                    << "u1, u4, c1, c4, p1, p4, g = "
                    << u1 << ", " << u4 << ", " << c1 << ", " << c4 << ", "
                    << p1 << ", " << p4 << ", " << g << endl;
            throw std::runtime_error(ss.str());
        }
    }
    double p2p1 = guess(0);
    auto p2 = p2p1 * p1;

    // Compute u2
    auto u2 = u1 + (c1 / g) * (p2p1-1) / (sqrt( ((g+1)/(2*g)) * (p2p1-1) + 1));
    // Compute V
    auto V = u1 + c1 * sqrt( ((g+1)/(2*g)) * (p2p1-1) + 1);
    // Compute c2
    auto c2 = c1 * sqrt(
            p2p1 * (
                (((g+1)/(g-1)) + p2p1
                ) / (
                1 + ((g+1)/(g-1)) * p2p1)
                )
    );
    // Compute r2
    auto compute_r = [&](double pressure, double c) {
        return g * pressure / (c*c);
    };
    auto r2 = compute_r(p2, c2);

    // p and u same across contact
    auto u3 = u2;
    auto p3 = p2;
    // Compute c3
    auto c3 = .5 * (g - 1) * (u4 + ((2*c4)/(g-1)) - u3);
    // Compute r3
    auto r3 = compute_r(p3, c3);

    // TODO: Figure out what x/t should be. I have changed it to x/t = 0, not
    // sure if this is right or not.

    // Flow inside expansion
    //auto u_exp = (2/(g+1)) * (x/t + ((g-1)/2) * u4 + c4);
    //auto c_exp = (2/(g+1)) * (x/t + ((g-1)/2) * u4 + c4) - x/t;
    auto u_exp = (2/(g+1)) * (((g-1)/2) * u4 + c4);
    auto c_exp = (2/(g+1)) * (((g-1)/2) * u4 + c4);
//    // Clip the speed of sound to be positive. This is not entirely necessary
//    // (the spurious negative speed of sound is only outside the expansion,
//    // so in the expansion everything is okay) but not doing this makes Numpy
//    // give warnings when computing pressure.
//    // TODO: Is this really necessary for this application...
//    c_exp[c_exp < 0] = 1e-16
    auto p_exp = p4 * pow(c_exp/c4, 2*g/(g-1));
    //auto r_exp = vector<double>(2);
    //for (int i = 0; i < x.rows(); i++) {
    //    r_exp(i) = compute_r(p_exp(i), c_exp(i));
    //}
    auto r_exp = compute_r(p_exp, c_exp);

    // Figure out which flow region each point is in
    for (int i = 0; i < x.rows(); i++) {
        //auto xt = x(i) / t;
        double xt = 0;
        // Left of expansion
        if (xt < (u4 - c4)) {
            r(i) = r4;
            u(i) = u4;
            p(i) = p4;
        // Inside expansion
        } else if (xt < (u3 - c3)) {
            r(i) = r_exp;
            u(i) = u_exp;
            p(i) = p_exp;
        // Right of expansion
        } else if (xt < u3) {
            r(i) = r3;
            u(i) = u3;
            p(i) = p3;
        // Left of shock
        } else if (xt < V) {
            r(i) = r2;
            u(i) = u2;
            p(i) = p2;
        // Right of shock
        } else if (xt > V) {
            r(i) = r1;
            u(i) = u1;
            p(i) = p1;
        }
    }
    ////TODO hack to try this
    //r(0) = r3;
    //r(1) = r2;
    //u(0) = u3;
    //u(1) = u2;
    //p(0) = p3;
    //p(1) = p2;
}


// Compute the fluid-fluid faces' contributions to the residual.
void compute_fluid_fluid_face_residual(matrix_ref<double> U,
        vector_ref<long> interface_IDs, matrix_ref<long> edge,
        matrix_ref<double> quad_wts, std::vector<double> quad_pts_phys,
        matrix_ref<double> limiter, std::vector<double> gradU,
        matrix_ref<double> xy, std::vector<double> area_normals_p2,
        matrix_ref<double> area, double g, double dt,
        matrix_ref<double> residual) {
    // Create buffers
    matrix<double> U_L(4, 1);
    matrix<double> U_R(4, 1);
    vector<double> F(4);
    // Loop over faces
    for (const auto face_ID : interface_IDs) {
        // Left and right cell IDs
        auto L = edge(face_ID, 0);
        auto R = edge(face_ID, 1);

        // Gradients for these cells
        matrix_map<double> gradU_L(&gradU[L*4*2], 4, 2);
        matrix_map<double> gradU_R(&gradU[R*4*2], 4, 2);

        // Loop over quadrature points
        vector<double> F_integral_L = vector<double>::Zero(4);
        vector<double> F_integral_R = vector<double>::Zero(4);
        for (auto i = 0; i < 2; i++) {
            // Evaluate solution at faces on left and right
            // -- First order component -- #
            U_L = U(L, all).transpose();
            U_R = U(R, all).transpose();

            // -- Second order component -- #
            // Get quadrature point in physical space
            // TODO: This looks a bit jank...issue is that Eigen cannot
            // handle 3D arrays. Wrapper of vector with strides??
            matrix<double> quad_pt(2, 1);
            quad_pt(0) = quad_pts_phys[face_ID*2*2 + i*2 + 0];
            quad_pt(1) = quad_pts_phys[face_ID*2*2 + i*2 + 1];
            // TODO: There has to be a cleaner way...
            for (int k = 0; k < 4; k++) {
                U_L(k) += limiter(L, k) * (gradU_L(k, all).transpose().cwiseProduct(quad_pt - xy(L, all).transpose()).sum());
                U_R(k) += limiter(R, k) * (gradU_R(k, all).transpose().cwiseProduct(quad_pt - xy(R, all).transpose()).sum());
            }

            // Package the normals
            matrix<double> area_normal(2, 1);
            area_normal(0, 0) = area_normals_p2[face_ID*2*2 + i*2 + 0];
            area_normal(1, 0) = area_normals_p2[face_ID*2*2 + i*2 + 1];
            vector<double> n_hat = area_normal(all, 0).normalized();
            vector<double> t_hat(2);
            t_hat << -n_hat(1), n_hat(0);

            // TODO: Should this be before or after adding the second order
            // component??
            // Convert left and right to primitive
            auto V_L = conservative_to_primitive(U_L, g);
            auto V_R = conservative_to_primitive(U_R, g);
            // Get normal/tangential component of velocity
            auto u_n_L = V_L(seq(1, 2)).dot(n_hat);
            auto u_n_R = V_R(seq(1, 2)).dot(n_hat);
            auto u_t_L = V_L(seq(1, 2)).dot(t_hat);
            auto u_t_R = V_R(seq(1, 2)).dot(t_hat);
            // Solve exact fluid-fluid Riemann problem
            auto r = vector<double>(2);
            auto u_n = vector<double>(2);
            auto p = vector<double>(2);
            // TODO What should this be??
            auto xL = -1e-5 * (xy(L, all) - xy(R, all)).norm();
            auto xR = -xL;
            exact_riemann_problem(V_L(0), u_n_L, V_L(3), V_R(0), u_n_R, V_R(3),
                    g, xL, xR, dt, r, u_n, p);
            // Velocity, after the Riemann problem (which modifies the normal
            // component)
            vector<double> vel_L = u_n[0] * n_hat + u_t_L * t_hat;
            vector<double> vel_R = u_n[1] * n_hat + u_t_R * t_hat;
            // Convert back to conservative
            V_L << r[0], vel_L(0), vel_L(1), p[0];
            V_R << r[1], vel_R(0), vel_R(1), p[1];
            matrix<double> U_L_Riemann = primitive_to_conservative(V_L, g);
            matrix<double> U_R_Riemann = primitive_to_conservative(V_R, g);

            // Evaluate left interior flux
            compute_flux(U_L, U_L_Riemann, area_normal, g, F);
            // Add contribution to quadrature
            F_integral_L += F * quad_wts(i, 0);
            // Evaluate right interior flux
            compute_flux(U_R_Riemann, U_R, area_normal, g, F);
            // Add contribution to quadrature
            F_integral_R += F * quad_wts(i, 0);
        }

        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L, 0) * F_integral_L;
        residual(R, all) +=  1 / area(R, 0) * F_integral_R;
    }
}


// Forward declare
// TODO: Header files!! Organize!!
vector<double> compute_ghost_state(vector<double> U, matrix_ref<double> U_global,
        long bc, matrix<double> quad_pt, double t,
        vector<double> bc_area_normal, matrix<double> bc_data,
        ComputeForcedInterfaceVelocity*);
// Compute the boundary faces' contributions to the residual.
void compute_boundary_face_residual(matrix_ref<double> U,
        matrix_ref<long> bc_type, matrix_ref<double> quad_wts,
        std::vector<double> quad_pts_phys, matrix_ref<double> limiter,
        std::vector<double> gradU, matrix_ref<double> xy,
        std::vector<double> area_normals_p2, matrix_ref<double> area,
        double g, long num_boundaries, matrix<double> bc_data,
        string problem_name, double t, matrix_ref<double> residual) {
    // Sizing
    int n_faces = bc_type.rows();

    // Create forced interface velocity functor
    ComputeForcedInterfaceVelocity* compute_interface_velocity = nullptr;
    if (problem_name == "RiemannProblem" or problem_name == "AdvectedContact"
            or problem_name == "AdvectedBubble") {
        compute_interface_velocity = new ComputeAdvectionInterfaceVelocity;
    } else if (problem_name == "CollapsingCylinder") {
        compute_interface_velocity = new ComputeCollapsingCylinderVelocity;
    } else if (problem_name == "Star") {
        compute_interface_velocity = new ComputeStarVelocity;
    } else if (problem_name == "Cavitation") {
        // Cavitation does not use a forced interface
    } else {
        cout << "Problem name invalid! Given problem_name = " << problem_name << endl;
    }

    // Create buffers
    matrix<double> U_L(4, 1);
    vector<double> F(4);
    // Loop over faces
    for (auto face_ID = 0; face_ID < n_faces; face_ID++) {
        // Left cell ID
        auto L = bc_type(face_ID, 0);
        // Type of BC
        auto bc = bc_type(face_ID, 1);

        // Gradients for this cell
        matrix_map<double> gradU_L(&gradU[L*4*2], 4, 2);

        // TODO: This nq thing is a hack
        int nq;
        if (face_ID < num_boundaries) {
            nq = 1;
        } else {
            nq = 2;
        }

        // Loop over quadrature points
        vector<double> F_integral = vector<double>::Zero(4);
        for (auto i = 0; i < nq; i++) {
            // Evaluate solution at face on left
            // -- First order component -- #
            U_L = U(L, all).transpose();
            // -- Second order component -- #
            matrix<double> quad_pt(2, 1);
            if (nq == 2) {
                // Get quadrature point in physical space
                // TODO: This looks a bit jank...issue is that Eigen cannot
                // handle 3D arrays
                quad_pt(0) = quad_pts_phys[face_ID*2*2 + i*2 + 0];
                quad_pt(1) = quad_pts_phys[face_ID*2*2 + i*2 + 1];
            } else {
                // Use the midpoint
                // TODO: This is wrong! I am using the node location when I
                // should be using the midpoint! I just don't have it calculated
                // yet.
                quad_pt(0) = xy(L, 0);
                quad_pt(1) = xy(L, 1);
            }
            // TODO: There has to be a cleaner way...
            for (int k = 0; k < 4; k++) {
                U_L(k) += limiter(L, k) * (gradU_L(k, all).transpose().cwiseProduct(quad_pt - xy(L, all).transpose()).sum());
            }

            // Package the normals
            matrix<double> area_normal(2, 1);
            area_normal(0, 0) = area_normals_p2[face_ID*2*2 + i*2 + 0];
            area_normal(1, 0) = area_normals_p2[face_ID*2*2 + i*2 + 1];

            // TODO Get rid of this matrix vs vector thing...
            vector<double> area_normal_vec(2);
            area_normal_vec(0) = area_normal(0, 0);
            area_normal_vec(1) = area_normal(1, 0);
            // Compute ghost state
            auto U_ghost_vec = compute_ghost_state(U_L, U, bc, quad_pt, t,
                    area_normal_vec, bc_data, compute_interface_velocity);
            // TODO: Fix matrix vs vector!
            matrix<double> U_ghost(4, 1);
            U_ghost << U_ghost_vec(0), U_ghost_vec(1), U_ghost_vec(2), U_ghost_vec(3);

            // Evaluate boundary fluxes
            compute_flux(U_L, U_ghost, area_normal, g, F);
            // Add contribution to quadrature
            if (nq == 2) {
                F_integral += F * quad_wts(i, 0);
            } else {
                F_integral += F;
            }
            if (face_ID == 126 or face_ID == 128 or face_ID == 130) {
                //cout << "Face " << face_ID << ", quad pt " << i << endl;
                //cout << U_ghost_vec << endl;
                //cout << F << endl;
                //cout << F_integral << endl;
            }
        }
        // Update residual of cell on the left
        residual(L, all) += -1 / area(L, 0) * F_integral;
    }
}


vector<double> compute_ghost_interface(vector<double> V, vector<double> bc_area_normal,
        vector<double> wall_velocity, double ghost_u = 0, double ghost_v = 0,
        double ghost_pressure = -1) {
    /*
    Compute the ghost state for a wall/interface BC.

    Inputs:
    -------
    V - array of primitive variables (4,)
    bc_area_normal - array of area-weighted normal vector (2,)
    wall_velocity - velocity of wall (2,)

    Outputs:
    --------
    V_ghost - array of primitive ghost state (4,)
    */
    vector<double> V_ghost(4);
    // The density is kept the same in the ghost state
    V_ghost(0) = V[0];
    // If this is an advected interface
    if (ghost_pressure >= 0) {
        // Use the velocity of the ghost fluid
        V_ghost(1) = ghost_u;
        V_ghost(2) = ghost_v;
        // Use the pressure of the ghost fluid
        V_ghost(3) = ghost_pressure;
    } else {
        // -- Compute ghost velocity for forced interface -- #
        // Compute unit normal vector
        vector<double> n_hat = bc_area_normal.normalized();
        // Tangent vector is normal vector, rotated 90 degrees
        vector<double> t_hat(2);
        t_hat(0) = -n_hat(1);
        t_hat(1) =  n_hat(0);
        // Create rotation matrix
        matrix<double> rotation(2, 2);
        rotation(0, all) = n_hat;
        rotation(1, all) = t_hat;
        // Rotate velocity into normal - tangential frame
        matrix<double> velocity(2, 1);
        velocity(0) = V(1);
        velocity(1) = V(2);
        matrix<double> velocity_nt = rotation * velocity;
        matrix<double> wall_velocity_nt = rotation * wall_velocity.reshaped(2, 1);
        // The normal velocity of the fluid is set so that the mean of the normal
        // velocity of the fluid vs. the ghost will equal the wall velocity.
        // This is represented by: 1/2 (U_fluid + U_ghost) = U_wall. Solving for
        // U_ghost gives:
        velocity_nt(0) = 2 * wall_velocity_nt(0) - velocity_nt(0);
        // Rotate back to original frame
        matrix<double> velocity_new = rotation.transpose() * velocity_nt;
        V_ghost(1) = velocity_new(0);
        V_ghost(2) = velocity_new(1);
        // Pressure is kept the same for a forced interface
        V_ghost(3) = V[3];
    }
    return V_ghost;
}

vector<double> compute_ghost_wall(vector<double> V, vector<double> bc_area_normal) {
    // A wall is just an interface that isn't moving
    auto wall_velocity = vector<double>::Zero(2);
    return compute_ghost_interface(V, bc_area_normal, wall_velocity);
}

vector<double> compute_ghost_advected_interface(
        vector<double> V, vector<double> V_ghost_cell,
        vector<double> bc_area_normal) {
    // An advected interface has the same wall velocity as the fluid velocity
    auto wall_velocity = vector<double>(2);
    wall_velocity << V[1], V[2];
    return compute_ghost_interface(V, bc_area_normal, wall_velocity,
            V_ghost_cell[1], V_ghost_cell[2], V_ghost_cell[3]);
}

vector<double> compute_ghost_state(vector<double> U, matrix_ref<double> U_global,
        long bc, matrix<double> quad_pt, double t,
        vector<double> bc_area_normal, matrix<double> bc_data,
        ComputeForcedInterfaceVelocity* compute_interface_velocity) {
    // Get gamma
    // TODO: bc_data is not storing what you think...the initial bc_type vs.
    // after being modified are different!
    auto g = bc_data(1, 4);
    vector<double> V_ghost(4);
    // Compute interface ghost state
    if (bc == 0) {
        auto V = conservative_to_primitive(U, g);
        vector<double> data = bc_data(bc, all);
        auto wall_velocity = (*compute_interface_velocity)(
                quad_pt(0), quad_pt(1), t, data);
        V_ghost = compute_ghost_interface(V, bc_area_normal, wall_velocity);
        //cout << quad_pt.transpose() << "  " << V.transpose() << "   " << V_ghost.transpose() << endl;
        //cout << quad_pt.transpose() << "  " << wall_velocity.transpose() << endl;
    // Compute wall ghost state
    } else if (bc == 1) {
        auto V = conservative_to_primitive(U, g);
        V_ghost = compute_ghost_wall(V, bc_area_normal);
    // Compute full state ghost state
    } else if (bc == 2) {
        V_ghost = bc_data(bc, seq(0, 3));
    } else if (bc > 2) {
        auto V = conservative_to_primitive(U, g);
        // Get ghost cell ID
        auto ghost_cell_ID = bc - 3;
        // Get ghost cell's state
        vector<double> U_ghost_cell = U_global(ghost_cell_ID, all).transpose();
        auto V_ghost_cell = conservative_to_primitive(U_ghost_cell, g);
        // Pass the ghost fluid state in to compute the ghost state
        V_ghost = compute_ghost_advected_interface(V, V_ghost_cell, bc_area_normal);
    } else {
        printf("ERROR: Invalid BC type given! bc = %li\n", bc);
    }
    // Convert to conservative
    auto U_ghost = primitive_to_conservative(V_ghost, g);
    return U_ghost;
}

vector<double> conservative_to_primitive(vector<double> U, double g) {
    // Unpack
    auto& r = U(0);
    auto& ru = U(1);
    auto& rv = U(2);
    auto& re = U(3);
    // Compute primitive variables
    vector<double> V(4);
    V(0) = r;
    V(1) = ru / r;
    V(2) = rv / r;
    V(3) = (re - .5 * (ru*ru + rv*rv) / r) * (g - 1);
    return V;
}

vector<double> primitive_to_conservative(vector<double> V, double g) {
    // Unpack
    auto& r = V(0);
    auto& u = V(1);
    auto& v = V(2);
    auto& p = V(3);
    // Compute conservative variables
    vector<double> U(4);
    U(0) = r;
    U(1) = r * u;
    U(2) = r * v;
    U(3) = p / (g - 1) + .5 * r * (u*u + v*v);
    return U;
}

