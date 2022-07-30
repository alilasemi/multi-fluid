#include <defines.h>
#include <face_residual.h>

PYBIND11_MODULE(pybind_bindings, m) {
    //TODO docstrings
    m.doc() = "doc"; // optional module docstring

    m.def("compute_interior_face_residual", &compute_interior_face_residual,
            "A function that...");
    m.def("compute_boundary_face_residual", &compute_boundary_face_residual,
            "A function that...");
}
