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
        matrix_ref<double> edge_points, double g,
        matrix_ref<double> residual) {
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
                // TODO: This def wont work?
                printf("nq = 1 is bugged!\n");
                //quad_pt = edge_points(face_ID);
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
            if (face_ID==0) {
            }
            // Add contribution to quadrature
            F_integral += F * quad_wts(i, 0);
        }
        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L, 0) * F_integral;
        residual(R, all) +=  1 / area(R, 0) * F_integral;
    }
}

PYBIND11_MODULE(interior_face_residual, m) {
    m.doc() = "doc"; // optional module docstring
    m.def("compute_interior_face_residual", &compute_interior_face_residual,
            "A function that computes...");
}
