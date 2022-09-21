#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
using std::cout, std::endl;
using std::string;

#include <defines.h>
#include <face_residual.h>
#include <forced_interface.h>
#include <riemann.h>
#include <roe.h>


// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(matrix_ref<double> U,
        matrix_ref<double> U_L, matrix_ref<double> U_R,
        vector_ref<long> interior_face_IDs, matrix_ref<long> edge,
        matrix_ref<double> limiter, std::vector<double> gradU,
        matrix_ref<double> xy, matrix_ref<double> area_normals_p1,
        matrix_ref<double> area, vector_ref<long> fluid_ID,
        std::vector<double> g, std::vector<double> psg,
        matrix_ref<double> residual) {
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
        // Get fluid data on left and right
        auto gL = g[fluid_ID(L)];
        auto gR = g[fluid_ID(R)];
        auto psgL = psg[fluid_ID(L)];
        auto psgR = psg[fluid_ID(R)];
        // Evaluate fluxes
        matrix<double> area_normals = area_normals_p1(face_ID, all).transpose();
        compute_flux(U_L(face_ID, all).transpose(),
                U_R(face_ID, all).transpose(), area_normals,
                gL, gR, psgL, psgR, F);

        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L, 0) * F;
        residual(R, all) +=  1 / area(R, 0) * F;
    }
}


// Compute the fluid-fluid faces' contributions to the residual.
void compute_fluid_fluid_face_residual(matrix_ref<double> U,
        vector_ref<long> interface_IDs, matrix_ref<long> edge,
        matrix_ref<double> quad_wts, std::vector<double> quad_pts_phys,
        matrix_ref<double> limiter, std::vector<double> gradU,
        matrix_ref<double> xy, std::vector<double> area_normals_p2,
        matrix_ref<double> area, vector_ref<long> fluid_ID,
        std::vector<double> g, std::vector<double> psg,
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
        // Get fluid data on left and right
        auto gL = g[fluid_ID(L)];
        auto gR = g[fluid_ID(R)];
        auto psgL = psg[fluid_ID(L)];
        auto psgR = psg[fluid_ID(R)];

        //TODO
        //std::vector<long> int_IDs = {2, 4, 7, 10, 11, 12};
        std::vector<long> int_IDs = {2, 12};
        bool print = std::find(int_IDs.begin(), int_IDs.end(), face_ID) != int_IDs.end();
        if (print) cout << face_ID << endl;

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
            auto V_L = conservative_to_primitive(U_L, gL, psgL);
            auto V_R = conservative_to_primitive(U_R, gR, psgR);
            // Get normal/tangential component of velocity
            auto u_n_L = V_L(seq(1, 2)).dot(n_hat);
            auto u_n_R = V_R(seq(1, 2)).dot(n_hat);
            auto u_t_L = V_L(seq(1, 2)).dot(t_hat);
            auto u_t_R = V_R(seq(1, 2)).dot(t_hat);
            // Solve exact fluid-fluid Riemann problem
            vector<double> result(9);
            compute_exact_riemann_problem(V_L(0), V_L(3), u_n_L, V_R(0), V_R(3),
                    u_n_R, gL, gR, psgL, psgR, result);
            auto& r_starL = result(5);
            auto& r_starR = result(6);
            auto& u_star = result(7);
            auto& p_star = result(8);
            // Velocity, after the Riemann problem (which modifies the normal
            // component)
            vector<double> vel_L = u_star * n_hat + u_t_L * t_hat;
            vector<double> vel_R = u_star * n_hat + u_t_R * t_hat;
            // Convert back to conservative
            V_L << r_starL, vel_L(0), vel_L(1), p_star;
            V_R << r_starR, vel_R(0), vel_R(1), p_star;
            matrix<double> U_L_Riemann = primitive_to_conservative(V_L, gL, psgL);
            matrix<double> U_R_Riemann = primitive_to_conservative(V_R, gR, psgR);
            //if (print) {
            //    cout << "U_Riemann" << endl;
            //    if (L == 4) {
            //        cout << U_L_Riemann << endl;
            //    } else {
            //        cout << U_R_Riemann << endl;
            //    }
            //}

            // Evaluate left interior flux, using the left fluid data (this is
            // why gL and psgL are repeated)
            compute_flux(U_L, U_L_Riemann, area_normal, gL, gL, psgL, psgL, F);
            //if (L == 549 or R == 549) {
            //    cout << "L = " << L << ", R = " << R << endl;
            //    cout << result.transpose() << endl;
            //    cout << U_L_Riemann.transpose() << endl;
            //    cout << U_R_Riemann.transpose() << endl;
            //    cout << F << endl;
            //}
            // Add contribution to quadrature
            F_integral_L += F * quad_wts(i, 0);
            // Evaluate right interior flux, using the right fluid data (this is
            // why gR and psgR are repeated)
            compute_flux(U_R_Riemann, U_R, area_normal, gR, gR, psgR, psgR, F);
            // Add contribution to quadrature
            F_integral_R += F * quad_wts(i, 0);
        }

        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L, 0) * F_integral_L;
        residual(R, all) +=  1 / area(R, 0) * F_integral_R;

        if (print) {
            cout << "integral" << endl;
            if (L == 4) {
                cout << F_integral_L << endl;
            } else {
                cout << F_integral_R << endl;
            }
        }
    }
    cout << "EEEEEEND " << endl;
}


// Forward declare
// TODO: Header files!! Organize!!
vector<double> compute_ghost_state(vector<double> U, matrix_ref<double> U_global,
        long bc, matrix<double> quad_pt, double t,
        vector<double> bc_area_normal, matrix<double> bc_data,
        ComputeForcedInterfaceVelocity*, double g, double psg);
// Compute the boundary faces' contributions to the residual.
void compute_boundary_face_residual(matrix_ref<double> U,
        matrix_ref<long> bc_type, matrix_ref<double> quad_wts,
        std::vector<double> quad_pts_phys, matrix_ref<double> limiter,
        std::vector<double> gradU, matrix_ref<double> xy,
        std::vector<double> area_normals_p2, matrix_ref<double> area,
        vector_ref<long> fluid_ID, std::vector<double> g,
        std::vector<double> psg, long num_boundaries, matrix<double> bc_data,
        std::string problem_name, double t, matrix_ref<double> residual) {
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
        // Get fluid data on left
        auto gL = g[fluid_ID(L)];
        auto psgL = psg[fluid_ID(L)];

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
                    area_normal_vec, bc_data, compute_interface_velocity, gL, psgL);
            // TODO: Fix matrix vs vector!
            matrix<double> U_ghost(4, 1);
            U_ghost << U_ghost_vec(0), U_ghost_vec(1), U_ghost_vec(2), U_ghost_vec(3);

            // Evaluate boundary fluxes. Assume the ghost fluid is of the same
            // fluid ID - this is why the gL and psgL are repeated.
            compute_flux(U_L, U_ghost, area_normal, gL, gL, psgL, psgL, F);
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
        ComputeForcedInterfaceVelocity* compute_interface_velocity, double g,
        double psg) {
    // TODO: bc_data is not storing what you think...the initial bc_type vs.
    // after being modified are different!
    vector<double> V_ghost(4);
    // Compute interface ghost state
    if (bc == 0) {
        auto V = conservative_to_primitive(U, g, psg);
        vector<double> data = bc_data(bc, all);
        auto wall_velocity = (*compute_interface_velocity)(
                quad_pt(0), quad_pt(1), t, data);
        V_ghost = compute_ghost_interface(V, bc_area_normal, wall_velocity);
        //cout << quad_pt.transpose() << "  " << V.transpose() << "   " << V_ghost.transpose() << endl;
        //cout << quad_pt.transpose() << "  " << wall_velocity.transpose() << endl;
    // Compute wall ghost state
    } else if (bc == 1) {
        auto V = conservative_to_primitive(U, g, psg);
        V_ghost = compute_ghost_wall(V, bc_area_normal);
    // Compute full state ghost state
    } else if (bc == 2) {
        V_ghost = bc_data(bc, seq(0, 3));
    } else if (bc > 2) {
        auto V = conservative_to_primitive(U, g, psg);
        // Get ghost cell ID
        auto ghost_cell_ID = bc - 3;
        // Get ghost cell's state
        vector<double> U_ghost_cell = U_global(ghost_cell_ID, all).transpose();
        auto V_ghost_cell = conservative_to_primitive(U_ghost_cell, g, psg);
        // Pass the ghost fluid state in to compute the ghost state
        V_ghost = compute_ghost_advected_interface(V, V_ghost_cell, bc_area_normal);
    } else {
        printf("ERROR: Invalid BC type given! bc = %li\n", bc);
    }
    // Convert to conservative
    auto U_ghost = primitive_to_conservative(V_ghost, g, psg);
    return U_ghost;
}

vector<double> conservative_to_primitive(vector<double> U, double g,
        double psg) {
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
    V(3) = (g - 1) * (re - .5 * (ru*ru + rv*rv) / r) - g * psg;
    return V;
}

vector<double> primitive_to_conservative(vector<double> V, double g,
        double psg) {
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
    U(3) = (p + g * psg) / (g - 1) + .5 * r * (u*u + v*v);
    return U;
}

