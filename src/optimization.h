#include <defines.h>


// Compute objective function for an edge point.
double f_edge(double zeta, matrix_ref<double> mesh_xy,
        matrix_ref<long> primal_cell_to_nodes, matrix_ref<double> phi_c,
        long primal_ID, matrix_ref<double> face_node_coords);

// Compute the Jacobian of the objective function for an edge point.
double f_edge_jac(double zeta, matrix_ref<double> mesh_xy,
        matrix_ref<long> primal_cell_to_nodes, matrix_ref<double> phi_c,
        long primal_ID, matrix_ref<double> face_node_coords);

// Compute objective function for a volume point.
double f_vol(vector_ref<double> xi_eta, matrix_ref<double> phi_c,
        matrix_ref<double> node_coords, long primal_ID,
        matrix_ref<double> edge_point_coords);

// Compute the Jacobian of the objective function for a volume point.
vector<double> f_vol_jac(vector_ref<double> xi_eta, matrix_ref<double> phi_c,
        matrix_ref<double> node_coords, long primal_ID,
        matrix_ref<double> edge_point_coords);
