#include <math.h>
#include <iostream>
using std::cout, std::endl;

#include <defines.h>
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
        np_array<double> edge_points_np, np_array<double> residual_np) {

    // Convert Numpy arrays to Eigen
    auto U = numpy_to_eigen(U_np);
    auto edge = numpy_to_eigen(edge_np);
    auto quad_wts = numpy_to_eigen(quad_wts_np);
    auto limiter = numpy_to_eigen(limiter_np);
    auto xy = numpy_to_eigen(xy_np);
    auto area_normals_p2 = numpy_to_eigen(area_normals_p2_np);
    auto area = numpy_to_eigen(area_np);
    auto edge_points = numpy_to_eigen(edge_points_np);
    auto residual = numpy_to_eigen(residual_np);

    // Sizing
    auto n_faces = edge.rows();

    // Create buffers
    // TODO: This nq thing is a hack
    int nq = 2;
    auto U_L = matrix<double>(4, 1);
    auto U_R = matrix<double>(4, 1);
    auto F = matrix<double>(nq, 4);
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
            auto quad_pt = matrix<double>(2, 1);
            if (nq == 2) {
                // Get quadrature point in physical space
                // TODO: This looks a bit jank...issue is that Eigen cannot
                // handle 3D arrays
                auto ptr = (double*) quad_pts_phys_np.request().ptr;
                quad_pt(0) = ptr[face_ID * 2 * 2 + i * 2 + 0];
                quad_pt(1) = ptr[face_ID * 2 * 2 + i * 2 + 1];
            } else {
                // Use the midpoint
                auto quad_pt = edge_points(face_ID);
            }
            // TODO: There has to be a cleaner way...
            U_L += (limiter(L, all).transpose().array() * (gradU_L * (quad_pt - xy(L, all).transpose())).array()).matrix();
            U_R += (limiter(R, all).transpose().array() * (gradU_R * (quad_pt - xy(R, all).transpose())).array()).matrix();
        }
    }


//
//            # Evalute interior fluxes
//            # TODO: For some reason, I have to pass 2D arrays into the
//            # C++/Pybind code. Otherwise, it says they are size 1 with ndim=0.
//            # It is annoying and I don't know why this happens (there are 1D
//            # array examples on the internet that work fine). Need a separate
//            # small reproducer.
//            # TODO: Check data ownership. F is created inside C++...not good.
//            F[i] = data.flux.compute_flux(U_L[i].reshape(1, -1), U_R[i].reshape(1, -1), mesh.area_normals_p2[face_ID, i])
//        # TODO: Unhardcode these quadrature weights (.5, .5)
//        F = np.mean(F, axis=0)
//
//        # Update residual of cells on the left and right
//        residual[L] += -1 / mesh.area[L] * F
//        residual[R] +=  1 / mesh.area[R] * F







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
