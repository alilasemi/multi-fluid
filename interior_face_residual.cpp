#include <math.h>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;
using std::cout, std::endl;

using StateVector = Eigen::Vector4d;
using Eigen::placeholders::all;

// Custom type for 2D matrices using Eigen maps. It is important to specify
// RowMajor since the default of Eigen is column major
template <class T> using matrix = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
// Custom type for Numpy arrays
template <class T> using np_array = py::array_t<T, py::array::c_style>;

class Roe {
    public:
        static void compute_flux(StateVector U_L, StateVector U_R,
                Eigen::Vector2d area_normal, StateVector F);
};

template <class T>
matrix<T> numpy_to_eigen_2D(np_array<T> A) {
    // Get pointer
    T* A_ptr = (T*) A.request().ptr;
    // Get shape
    const auto& shape = A.request().shape;
    // Turn into Eigen map
    matrix<T> A_map(A_ptr, shape[0], shape[1]);
    return A_map;
}

// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(np_array<double> U_np,
        np_array<long> edge_np) {
    // Convert Numpy arrays to Eigen
    auto U = numpy_to_eigen_2D(U_np);
    auto edge = numpy_to_eigen_2D(edge_np);

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
