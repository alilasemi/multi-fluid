#include <math.h>
#include <iostream>
using std::cout, std::endl;

#include <defines.h>
//TODO
#include <roe.cpp>

// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(matrix_ref<double> U,
        matrix_ref<long> edge, matrix_ref<double> quad_wts,
        np_array<double> quad_pts_phys_np, matrix_ref<double> limiter,
        np_array<double> gradU_np, matrix_ref<double> xy,
        np_array<double> area_normals_p2_np, matrix_ref<double> area,
        double g, matrix_ref<double> residual) {
    // Sizing
    auto n_faces = edge.rows();

    // Create buffers
    // TODO: This nq thing is a hack
    int nq = 2;
    matrix<double> U_L(4, 1);
    matrix<double> U_R(4, 1);
    vector<double> F(4);
    // Loop over faces
    for (auto face_ID = 0; face_ID < n_faces; face_ID++) {
        // Left and right cell IDs
        auto L = edge(face_ID, 0);
        auto R = edge(face_ID, 1);
        // Skip interfaces - those are handled seperately
        if (L == -1 and R == -1) {
            continue;
        }
        // Gradients for these cells
        auto gradU_ptr = (double*) gradU_np.request().ptr;
        matrix_map<double> gradU_L(gradU_ptr + L*4*2, 4, 2);
        matrix_map<double> gradU_R(gradU_ptr + R*4*2, 4, 2);

        // Loop over quadrature points
        vector<double> F_integral = vector<double>::Zero(4);
        for (auto i = 0; i < nq; i++) {
            // Evaluate solution at faces on left and right
            // -- First order component -- #
            U_L = U(L, all).transpose();
            U_R = U(R, all).transpose();
            // -- Second order component -- #
            matrix<double> quad_pt(2, 1);
            if (nq == 2) {
                // Get quadrature point in physical space
                // TODO: This looks a bit jank...issue is that Eigen cannot
                // handle 3D arrays
                auto ptr = (double*) quad_pts_phys_np.request().ptr;
                quad_pt(0) = ptr[face_ID * 2 * 2 + i * 2 + 0];
                quad_pt(1) = ptr[face_ID * 2 * 2 + i * 2 + 1];
            } else {
                // Use the midpoint
                // TODO: This def wont work
                printf("nq = 1 is bugged!\n");
            }
            // TODO: There has to be a cleaner way...
            for (int k = 0; k < 4; k++) {
                U_L(k) += limiter(L, k) * (gradU_L(k, all).transpose().cwiseProduct(quad_pt - xy(L, all).transpose()).sum());
                U_R(k) += limiter(R, k) * (gradU_R(k, all).transpose().cwiseProduct(quad_pt - xy(R, all).transpose()).sum());
            }

            // Package the normals
            matrix<double> area_normal(2, 1);
            auto area_normals_ptr = (double*) area_normals_p2_np.request().ptr;
            area_normal(0, 0) = area_normals_ptr[face_ID * 2 * 2 + i * 2 + 0];
            area_normal(1, 0) = area_normals_ptr[face_ID * 2 * 2 + i * 2 + 1];
            // Evaluate interior fluxes
            compute_flux(U_L, U_R, area_normal, g, F);
            // Add contribution to quadrature
            if (nq == 2) {
                F_integral += F * quad_wts(i, 0);
            } else {
                F_integral += F;
            }
        }
        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L, 0) * F_integral;
        residual(R, all) +=  1 / area(R, 0) * F_integral;
    }
}

// Compute the boundary faces' contributions to the residual.
void compute_boundary_face_residual(matrix_ref<double> U,
        np_array<double> U_ghost_np,
        matrix_ref<long> bc_type, matrix_ref<double> quad_wts,
        np_array<double> quad_pts_phys_np, matrix_ref<double> limiter,
        np_array<double> gradU_np, matrix_ref<double> xy,
        np_array<double> area_normals_p2_np, matrix_ref<double> area,
        double g, long num_boundaries, matrix_ref<double> residual) {
    // Sizing
    auto n_faces = bc_type.rows();

    // Create buffers
    matrix<double> U_L(4, 1);
    vector<double> F(4);
    // Loop over faces
    for (auto face_ID = 0; face_ID < n_faces; face_ID++) {
        // Left cell ID
        auto L = bc_type(face_ID, 0);

        // Gradients for this cell
        auto gradU_ptr = (double*) gradU_np.request().ptr;
        matrix_map<double> gradU_L(gradU_ptr + L*4*2, 4, 2);

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
            // Get ghost state
            auto U_ghost_ptr = (double*) U_ghost_np.request().ptr;
            matrix<double> U_ghost = matrix_map<double>(U_ghost_ptr + face_ID*2*4 + i*4, 4, 1);
            // Evaluate solution at face on left
            // -- First order component -- #
            U_L = U(L, all).transpose();
            // -- Second order component -- #
            matrix<double> quad_pt(2, 1);
            if (nq == 2) {
                // Get quadrature point in physical space
                // TODO: This looks a bit jank...issue is that Eigen cannot
                // handle 3D arrays
                auto ptr = (double*) quad_pts_phys_np.request().ptr;
                quad_pt(0) = ptr[face_ID * 2 * 2 + i * 2 + 0];
                quad_pt(1) = ptr[face_ID * 2 * 2 + i * 2 + 1];
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
            auto area_normals_ptr = (double*) area_normals_p2_np.request().ptr;
            area_normal(0, 0) = area_normals_ptr[face_ID * 2 * 2 + i * 2 + 0];
            area_normal(1, 0) = area_normals_ptr[face_ID * 2 * 2 + i * 2 + 1];
            // Evaluate interior fluxes
            compute_flux(U_L, U_ghost, area_normal, g, F);
            // Add contribution to quadrature
            if (nq == 2) {
                F_integral += F * quad_wts(i, 0);
            } else {
                F_integral += F;
            }
        }
        // Update residual of cell on the left
        residual(L, all) += -1 / area(L, 0) * F_integral;
    }
}


vector<double> compute_ghost_interface(vector<double> V, vector<double> bc_area_normal,
        vector<double> wall_velocity) {
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
    // The density and pressure are kept the same in the ghost state
    auto r = V[0];
    auto p = V[3];
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
    vector<double> velocity(2);
    velocity(0) = V(1);
    velocity(1) = V(2);
    vector<double> velocity_nt = rotation * velocity;
    vector<double> wall_velocity_nt = rotation * wall_velocity;
    // The normal velocity of the fluid is set so that the mean of the normal
    // velocity of the fluid vs. the ghost will equal the wall velocity.
    // This is represented by: 1/2 (U_fluid + U_ghost) = U_wall. Solving for
    // U_ghost gives:
    velocity_nt(0) = 2 * wall_velocity_nt(0) - velocity_nt(0);
    // Rotate back to original frame
    vector<double> velocity_new = rotation.transpose() * velocity_nt;
    vector<double> V_ghost(4);
    V_ghost(0) = r;
    V_ghost(1) = velocity_new(0);
    V_ghost(2) = velocity_new(1);
    V_ghost(3) = p;
    return V_ghost;
}

vector<double> compute_ghost_wall(vector<double> V, vector<double> bc_area_normal) {
    // A wall is just an interface that isn't moving
    auto wall_velocity = vector<double>::Zero(2);
    return compute_ghost_interface(V, bc_area_normal, wall_velocity);
}

//TODO Fill me!
vector<double> conservative_to_primitive(vector<double> U, double g) {
    return U;
}
//TODO Fill me!
vector<double> primitive_to_conservative(vector<double> V, double g) {
    return V;
}

vector<double> compute_ghost_state(vector<double> U, long bc,
        vector<double> bc_area_normal, matrix<double> bc_data) {
    // TODO: Create bc_data, and place gamma in the final index for all cases
    // TODO: Get the exact wall_velocity in here somehow
    // Compute interface ghost state
    auto g = bc_data(bc, 4);
    vector<double> V_ghost(4);
    // Compute interface ghost state
    if (bc == 0) {
        auto V = conservative_to_primitive(U, g);
        vector<double> wall_velocity(4); // TODO: Fill me! from bc_data? But how...this changes?
        V_ghost = compute_ghost_interface(V, bc_area_normal, wall_velocity);
    // Compute wall ghost state
    } else if (bc == 1) {
        auto V = conservative_to_primitive(U, g);
        V_ghost = compute_ghost_wall(V, bc_area_normal);
    // Compute inflow/outflow ghost state
    } else if (bc == 2 or bc == 3) {
        //V_ghost = bc_data(bc, 0:4); //TODO
    } else {
        printf("ERROR: Invalid BC type given! bc = %li\n", bc);
    }
    // Convert to conservative
    auto U_ghost = primitive_to_conservative(V_ghost, g);
    return U_ghost;
}

PYBIND11_MODULE(interior_face_residual, m) {
    m.doc() = "doc"; // optional module docstring
    m.def("compute_interior_face_residual", &compute_interior_face_residual,
            "A function that computes...");
    m.def("compute_boundary_face_residual", &compute_boundary_face_residual,
            "A function that computes...");
}
