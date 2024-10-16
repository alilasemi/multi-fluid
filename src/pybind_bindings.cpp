#include <defines.h>
#include <face_residual.h>
#include <gradient.h>
#include <riemann.h>
#include <roe.h>
#include <optimization.h>

PYBIND11_MODULE(libpybind_bindings, m) {
    //TODO docstrings
    m.doc() = "doc"; // optional module docstring

    m.def("compute_interior_face_residual", &compute_interior_face_residual,
            "A function that...");
    m.def("compute_fluid_fluid_face_residual", &compute_fluid_fluid_face_residual,
            "A function that...");
    m.def("compute_boundary_face_residual", &compute_boundary_face_residual,
            "A function that...");
    m.def("evaluate_solution_at_interior_faces", &evaluate_solution_at_interior_faces,
            "A function that...");
    m.def("evaluate_solution_at_interfaces", &evaluate_solution_at_interfaces,
            "A function that...");

    m.def("compute_gradient", &compute_gradient,
            "A function that...");
    m.def("compute_gradient_phi", &compute_gradient_phi,
            "A function that...");

    m.def("compute_exact_riemann_problem", &compute_exact_riemann_problem,
            "A function that...");
    m.def("compute_flux", &compute_flux,
            "A function that...");
    m.def("compute_flux_roe", &compute_flux_roe,
            "A function that...");

    m.def("f_edge", &f_edge,
            "A function that...");
    m.def("f_edge_jac", &f_edge_jac,
            "A function that...");
    m.def("f_vol", &f_vol,
            "A function that...");
    m.def("f_vol_jac", &f_vol_jac,
            "A function that...");
}
