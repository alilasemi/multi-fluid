#include <math.h>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using StateVector = Eigen::Vector4d;

class Roe {
    public:
        static void compute_flux(StateVector U_L, StateVector U_R,
                Eigen::Vector2d area_normal, StateVector F);
};

// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(py::array_t<double> U) {
    // Get pointers
    double* U_ptr = (double*) U.request().ptr;
}

PYBIND11_MODULE(compute_interior_face_residual, m) {
    m.doc() = "doc"; // optional module docstring
    m.def("compute_interior_face_residual", &compute_interior_face_residual,
            "A function that computes...");
}


void Roe::compute_flux(StateVector U_L, StateVector U_R,
        Eigen::Vector2d area_normal, StateVector F) {
    int c=5;
}
