#include <math.h>
#include <iostream>
using std::cout, std::endl;

#include <defines.h>
//TODO
#include <roe.cpp>
using StateVector = Eigen::Vector4d;

class Roe {
    public:
        static void compute_flux(StateVector U_L, StateVector U_R,
                Eigen::Vector2d area_normal, StateVector F);
};

// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(np_array<double> U_np,
        np_array<long> edge_np, np_array<double> quad_wts_np,
        np_array<double> quad_pts_phys_np, np_array<double> limiter_np,
        np_array<double> gradU_np, np_array<double> xy_np,
        np_array<double> area_normals_p2_np, np_array<double> area_np,
        np_array<double> edge_points_np, double g,
        np_array<double>& residual_np) {

    // Convert Numpy arrays to Eigen
    matrix<double> U = numpy_to_eigen(U_np);
    matrix<long> edge = numpy_to_eigen(edge_np);
    matrix<double> quad_wts = numpy_to_eigen(quad_wts_np);
    matrix<double> limiter = numpy_to_eigen(limiter_np);
    matrix<double> xy = numpy_to_eigen(xy_np);
    matrix<double> area = numpy_to_eigen(area_np);
    matrix<double> edge_points = numpy_to_eigen(edge_points_np);
    matrix<double> residual = numpy_to_eigen(residual_np);

    // Sizing
    auto n_faces = edge.rows();

    // Create buffers
    // TODO: This nq thing is a hack
    int nq = 2;
    matrix<double> U_L(4, 1);
    matrix<double> U_R(4, 1);
    matrix<double> F(nq, 4);
    matrix<double> F_integral = matrix<double>::Zero(1, 4);
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
                // TODO: This def wont work? And remember to ditch the auto!
                auto quad_pt = edge_points(face_ID);
            }
            // TODO: There has to be a cleaner way...
            U_L += (limiter(L, all).transpose().array() * (gradU_L * (quad_pt - xy(L, all).transpose())).array()).matrix();
            U_R += (limiter(R, all).transpose().array() * (gradU_R * (quad_pt - xy(R, all).transpose())).array()).matrix();

            // Package the normals
            matrix<double> area_normal(2, 1);
            auto area_normals_ptr = (double*) area_normals_p2_np.request().ptr;
            area_normal(0) = area_normals_ptr[face_ID * 2 * 2 + i * 2 + 0];
            area_normal(1) = area_normals_ptr[face_ID * 2 * 2 + i * 2 + 1];
            // Evaluate interior fluxes
            compute_flux(U_L, U_R, area_normal, g, F(i, all).transpose());
            // Add contribution to quadrature
            F_integral += F(i, all) * quad_wts(i);
        }
        // Update residual of cells on the left and right
        residual(L, all) += -1 / area(L) * F_integral;
        residual(R, all) +=  1 / area(R) * F_integral;
    }
}

PYBIND11_MODULE(interior_face_residual, m) {
    m.doc() = "doc"; // optional module docstring
    m.def("compute_interior_face_residual", &compute_interior_face_residual,
            "A function that computes...");
}

void Roe::compute_flux(StateVector U_L, StateVector U_R,
        Eigen::Vector2d area_normal, StateVector F) {
    int c=5;
}
