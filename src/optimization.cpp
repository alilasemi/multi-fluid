#include <defines.h>

#include <optimization.h>


// Convert from element reference space to physical space.
//
//    Inputs:
//        xi_eta: Two values of xi and eta, the element reference coordinates
//        node_coords: Coordinates in physical space of the element nodes
//    Outputs:
//        xy: Resulting coordinates in physical space
//
vector<double> elemref_to_physical(vector_ref<double> xi_eta,
        matrix_ref<double> node_coords) {
    vector_ref<double> xy1 = node_coords(0, all);
    vector_ref<double> xy2 = node_coords(1, all);
    vector_ref<double> xy3 = node_coords(2, all);
    vector<double> xy = (1 - xi_eta(0) - xi_eta(1)) * xy1 + xi_eta(0) * xy2 + xi_eta(1) * xy3;
    return xy;
}

// TODO Solve this linear system exactly
vector<double> physical_to_elemref(vector_ref<double> xy,
        matrix_ref<double> node_coords) {
    vector_ref<double> xy1 = node_coords(0, all);
    vector_ref<double> xy2 = node_coords(1, all);
    vector_ref<double> xy3 = node_coords(2, all);
    matrix<double> A(2, 2);
    A(all, 0) = -xy1 + xy2;
    A(all, 1) = -xy1 + xy3;
    vector<double> b = xy - xy1;
    return A.partialPivLu().solve(b);
}

// Compute Jacobian of physical space with respect to element reference
// space.
//
//    Inputs:
//        xi_eta: Two values of xi and eta, the element reference coordinates
//        node_coords: Coordinates in physical space of the element nodes
//    Outputs:
//        jac: Jacobian matrix, d(x, y)/d(xi, eta)
//
matrix<double> elemref_to_physical_jacobian(matrix_ref<double> node_coords) {
    vector_ref<double> xy1 = node_coords(0, all);
    vector_ref<double> xy2 = node_coords(1, all);
    vector_ref<double> xy3 = node_coords(2, all);
    matrix<double> jac(2, 2);
    jac(all, 0) = -xy1 + xy2;
    jac(all, 1) = -xy1 + xy3;
    return jac;
}

// Convert from face reference space to physical space
//
//    Inputs:
//        zeta: Value of zeta, the face reference coordinate
//        node_coords: Coordinates in physical space of the face nodes
//    Outputs:
//        xy: Resulting coordinates in physical space
//
vector<double> faceref_to_physical(double zeta,
        matrix_ref<double> node_coords) {
    vector_ref<double> xy1 = node_coords(0, all);
    vector_ref<double> xy2 = node_coords(1, all);
    vector<double> xy = zeta*xy2 + (1 - zeta) * xy1;
    return xy;
}

// Compute Jacobian of physical space with respect to face reference
// space.
//
//    Inputs:
//        node_coords: The coordinates in physical space of the face nodes
//    Outputs:
//        jac: Jacobian array, d(x, y)/d(zeta)
//    """
vector<double> faceref_to_physical_jacobian(matrix_ref<double> node_coords) {
    vector_ref<double> xy1 = node_coords(0, all);
    vector_ref<double> xy2 = node_coords(1, all);
    vector<double> jac = xy2 - xy1;
    return jac;
}

// Compute Jacobian of element reference space with respect to face
// reference space by using the chain rule.
//
//    Inputs:
//        elem_node_coords: The coordinates in physical space of the element nodes
//        face_node_coords: The coordinates in physical space of the face nodes
//    Outputs:
//        d_xi_eta_d_zeta: Jacobian array, d(xi, eta)/d(zeta)
//
vector<double> faceref_to_elemref_jacobian( matrix_ref<double> elem_node_coords,
        matrix_ref<double> face_node_coords) {
    matrix<double> d_xy_d_xi_eta = elemref_to_physical_jacobian(elem_node_coords);
    vector<double> d_xy_d_zeta = faceref_to_physical_jacobian(face_node_coords);
    vector<double> d_xi_eta_d_zeta = d_xy_d_xi_eta.inverse() * d_xy_d_zeta;
    return d_xi_eta_d_zeta;
}

double evaluate_level_set_fit(vector_ref<double> phi_c,
        vector_ref<double> xi_eta) {
    auto& xi = xi_eta(0);
    auto& eta = xi_eta(1);
    // Compute basis
    vector<double> basis(6);
    basis << 1, xi, eta, pow(xi, 2), pow(eta, 2), xi*eta;
    // Evaluate phi
    double phi = phi_c.dot(basis);
    return phi;
}

vector<double> evaluate_level_set_gradient(vector_ref<double> phi_c,
        vector_ref<double> xi_eta) {
    auto& xi = xi_eta(0);
    auto& eta = xi_eta(1);
    // Compute basis gradient
    vector<double> d_basis_d_xi(6);
    d_basis_d_xi << 0, 1, 0, 2*xi, 0, eta;
    vector<double> d_basis_d_eta(6);
    d_basis_d_eta << 0, 0, 1, 0, 2*eta, xi;
    // Evaluate phi gradient
    vector<double> d_phi_d_xi_eta(2);
    d_phi_d_xi_eta(0) = phi_c.dot(d_basis_d_xi);
    d_phi_d_xi_eta(1) = phi_c.dot(d_basis_d_eta);
    return d_phi_d_xi_eta;
}

// Compute objective function for an edge point.
double f_edge(double zeta, matrix_ref<double> mesh_xy,
        matrix_ref<long> primal_cell_to_nodes, matrix_ref<double> phi_c,
        long primal_ID, matrix_ref<double> face_node_coords) {
    // Compute physical coordinates
    vector<double> xy = faceref_to_physical(zeta, face_node_coords);
    // Get nodes of this primal cell
    vector_ref<long> node_IDs = primal_cell_to_nodes(primal_ID, all);
    matrix<double> elem_node_coords = mesh_xy(node_IDs, all);
    // Compute element reference coordinates
    vector<double> xi_eta = physical_to_elemref(xy, elem_node_coords);
    // Compute phi from the fit
    double phi = evaluate_level_set_fit(phi_c(primal_ID, all), xi_eta);
    // Compute objective function
    double f = .5 * pow(phi, 2);
    return f;
}

// Compute the Jacobian of the objective function for an edge point.
double f_edge_jac(double zeta, matrix_ref<double> mesh_xy,
        matrix_ref<long> primal_cell_to_nodes, matrix_ref<double> phi_c,
        long primal_ID, matrix_ref<double> face_node_coords) {
    // Compute physical coordinates
    vector<double> xy = faceref_to_physical(zeta, face_node_coords);
    // Get nodes of this primal cell
    vector_ref<long> node_IDs = primal_cell_to_nodes(primal_ID, all);
    matrix<double> elem_node_coords = mesh_xy(node_IDs, all);
    // Compute element reference coordinates
    vector<double> xi_eta = physical_to_elemref(xy, elem_node_coords);
    // Compute phi from the fit
    double phi = evaluate_level_set_fit(phi_c(primal_ID, all), xi_eta);
    // Compute d(phi)/d(xi, eta) from the fit
    vector<double> d_phi_d_xi_eta = evaluate_level_set_gradient(
            phi_c(primal_ID, all), xi_eta);
    // Compute Jacobian
    vector<double> d_xi_eta_d_zeta = faceref_to_elemref_jacobian(
            elem_node_coords, face_node_coords);
    // Combine using chain rule to get d(phi)/d(zeta)
    double d_phi_d_zeta = d_phi_d_xi_eta.dot(d_xi_eta_d_zeta);
    // Use chain rule to compute d(f)/d(zeta)
    double f_jac = phi * d_phi_d_zeta;
    return f_jac;
}

double factor = 0.1;
// Compute objective function for a volume point.
double f_vol(vector_ref<double> xi_eta, matrix_ref<double> phi_c,
        matrix_ref<double> node_coords, long primal_ID,
        matrix_ref<double> edge_point_coords) {
    vector<double> xy = elemref_to_physical(xi_eta, node_coords);
    // Evaluate level set
    double phi = evaluate_level_set_fit(phi_c(primal_ID, all), xi_eta);
    // Compute objective function
    double f = .5 * pow(phi, 2);
    for (int i = 0; i < edge_point_coords.rows(); i++) {
        f += factor * .5 * (xy - edge_point_coords(i, all).transpose()).squaredNorm();
    }
    return f;
}

// Compute the Jacobian of the objective function for a volume point.
vector<double> f_vol_jac(vector_ref<double> xi_eta, matrix_ref<double> phi_c,
        matrix_ref<double> node_coords, long primal_ID,
        matrix_ref<double> edge_point_coords) {
    vector<double> xy = elemref_to_physical(xi_eta, node_coords);
    // Evaluate level set
    double phi = evaluate_level_set_fit(phi_c(primal_ID, all), xi_eta);
    // Compute d(phi)/d(xi, eta) from the fit
    vector<double> d_phi_d_xi_eta = evaluate_level_set_gradient(
            phi_c(primal_ID, all), xi_eta);
    // Use chain rule to compute d(f)/d(xi, eta)
    vector<double> f_jac = phi * d_phi_d_xi_eta;
    // Add contribution from edge point terms
    matrix<double> jac = elemref_to_physical_jacobian(node_coords);
    for (int i = 0; i < edge_point_coords.rows(); i++) {
        f_jac += factor * (xy.transpose() - edge_point_coords(i, all)) * jac;
    }
    return f_jac;
}
