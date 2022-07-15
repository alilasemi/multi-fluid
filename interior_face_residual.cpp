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
        np_array<long> edge_np) {
    // Convert Numpy arrays to Eigen
    auto U = numpy_to_eigen(U_np);
    auto edge = numpy_to_eigen(edge_np);

    // Sizing
    auto n_faces = edge.rows();

    // Create buffers
    // TODO: This nq thing is a hack
    int nq = 2;
    auto U_L = Eigen::MatrixXd(nq, 4);
    auto U_R = Eigen::MatrixXd(nq, 4);
    auto F = Eigen::MatrixXd(nq, 4);
    // Loop over faces
    for (auto face_ID = 0; face_ID < n_faces; face_ID++) {
        // Left and right cell IDs
        auto L = edge(face_ID, 0);
        auto R = edge(face_ID, 1);

        // Evaluate solution at faces on left and right, for both quadrature
        // points
        // -- First order component -- #
        for (auto iq = 0; iq < 2; iq++) {
            for (auto k = 0; k < 4; k++) {
                U_L(iq, k) = U(L, k);
                U_R(iq, k) = U(L, k);
            }
        }
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
