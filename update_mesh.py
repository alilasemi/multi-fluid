import numpy as np
import scipy.optimize

from transformation import (elemref_to_physical_jacobian, physical_to_elemref,
        faceref_to_elemref_jacobian, elemref_to_physical, faceref_to_physical)
from build.src.libpybind_bindings import f_edge, f_edge_jac, f_vol, f_vol_jac


def update_mesh(mesh, data, problem):
    '''
    Update the dual mesh to fit the interface better.
    '''

    # Construct the least squares fit to the level set, if needed
    if not problem.has_exact_phi:
        construct_level_set_fit(mesh, data)

    # Mapping of volume points to their edge points
    vol_points_to_edge_points = np.empty(mesh.n_primal_cells, dtype=object)
    for i in range(mesh.n_primal_cells):
        vol_points_to_edge_points[i] = []

    # First, loop to update edge points
    for face_ID in range(mesh.n_faces):
        # TODO: Only works for interior faces
        if mesh.is_boundary(face_ID): continue

        # Get dual mesh neighbors
        i, j = mesh.edge[face_ID]
        # Check for interface
        if data.phi[i] * data.phi[j] < 0:
            # Optimize
            optimize_edge_point(mesh, data, problem, i, j, face_ID)
            # Store neighboring volume points
            primal_ID_L = mesh.face_points[face_ID, 0]
            primal_ID_R = mesh.face_points[face_ID, 2]
            vol_points_to_edge_points[primal_ID_L].append(face_ID)
            vol_points_to_edge_points[primal_ID_R].append(face_ID)

    # Now that edge points are updated, update volume points
    for primal_ID in range(mesh.n_primal_cells):
        # Skip volume points that are not involved in an interface
        if len(vol_points_to_edge_points[primal_ID]) == 0: continue
        # Get edge points connected to this volume point
        edge_point_coords = mesh.edge_points[
                vol_points_to_edge_points[primal_ID]]
        # Optimize
        optimize_vol_point(mesh, data, problem, primal_ID, face_ID,
                edge_point_coords)


def construct_level_set_fit(mesh, data):
    # TODO: Put this in the data class
    data.phi_c = np.empty((mesh.n_primal_cells, 6))
    # Loop over primal cells
    for primal_ID in range(mesh.n_primal_cells):
        # Get nodes of this primal cell
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        node_coords = mesh.xy[node_IDs]
        # Element reference coordinates of nodes
        xi1, eta1 = [0, 0]
        xi2, eta2 = [1, 0]
        xi3, eta3 = [0, 1]
        # Calculate Jacobian matrix of barycentric transformation
        jac = elemref_to_physical_jacobian(node_coords)
        # Get gradient of phi wrt barycentric coordinates
        grad_phi_xy1 = data.grad_phi[node_IDs[0]] @ jac
        grad_phi_xy2 = data.grad_phi[node_IDs[1]] @ jac
        grad_phi_xy3 = data.grad_phi[node_IDs[2]] @ jac
        # -- Construct A matrix -- #
        A = np.empty((9, 6))
        # Equations for phi at the nodes
        A[0] = [1, 0, 0, 0, 0, 0]
        A[1] = [1, 1, 0, 1, 0, 0]
        A[2] = [1, 0, 1, 0, 1, 0]
        # Equations for gradient of phi at the nodes
        A[3] = [0, 1, 0, 2*xi1, 0,        eta1]
        A[4] = [0, 0, 1, 0,        2*eta1, xi1]
        A[5] = [0, 1, 0, 2*xi2, 0,        eta2]
        A[6] = [0, 0, 1, 0,        2*eta2, xi2]
        A[7] = [0, 1, 0, 2*xi3, 0,        eta3]
        A[8] = [0, 0, 1, 0,        2*eta3, xi3]
        # Construct b vector
        b = np.array([
            data.phi[node_IDs[0]], data.phi[node_IDs[1]], data.phi[node_IDs[2]],
            grad_phi_xy1[0], grad_phi_xy1[1],
            grad_phi_xy2[0], grad_phi_xy2[1],
            grad_phi_xy3[0], grad_phi_xy3[1],
        ])
        # Solve with least squares
        data.phi_c[primal_ID] = np.linalg.solve(A.T @ A, A.T @ b)

def evaluate_level_set_fit(data, primal_ID, xi_eta):
    xi, eta = xi_eta
    # Compute basis
    basis = np.array([1, xi, eta, xi**2, eta**2, xi*eta])
    # Evaluate phi
    phi = data.phi_c[primal_ID] @ basis
    return phi

def evaluate_level_set_gradient(data, primal_ID, xi_eta):
    xi, eta = xi_eta
    # Compute basis gradient
    d_basis_d_xi = np.array([0, 1, 0, 2*xi, 0, eta])
    d_basis_d_eta = np.array([0, 0, 1, 0, 2*eta, xi])
    # Evaluate phi gradient
    d_phi_d_xi_eta = np.empty(2)
    d_phi_d_xi_eta[0] = data.phi_c[primal_ID] @ d_basis_d_xi
    d_phi_d_xi_eta[1] = data.phi_c[primal_ID] @ d_basis_d_eta
    return d_phi_d_xi_eta

def optimize_edge_point(mesh, data, problem, i, j, face_ID):
    coords = mesh.edge_points[mesh.face_points[face_ID, 1]]
    face_node_coords = mesh.xy[[i, j]]
    # Solve optimization problem for the new node locations,
    # by moving them as close as possible to the interface
    # (phi = 0) while still keeping the point between nodes
    # i and j
    # The guess value is important - several values are
    # tried and the minimum across all guesses is taken as
    # the final answer.
    # This is done for both neighboring primal cells and the result is averaged,
    # since the reconstruction of phi need not be continuous.

    if problem.has_exact_phi:
        print("Exact level set is no longer supported!")

    # Get the two primal cells on either side of this edge
    i_primals = mesh.nodes_to_primal_cells[i]
    j_primals = mesh.nodes_to_primal_cells[j]
    primal_cells = np.intersect1d(i_primals, j_primals)
    # Loop over both primal cells
    new_coords = np.empty((2, 2))
    for index, primal_ID in enumerate(primal_cells):
        guesses = np.linspace(0, 1, 5)
        success = False
        minimum_phi = 1e99
        # TODO: Figure out what this should be
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        tol = 1e-2 * .5 * np.min(data.phi[node_IDs]**2)
        for guess in guesses:
            optimization = scipy.optimize.minimize(
                    f_edge, guess,
                    args=(mesh.xy, mesh.primal_cell_to_nodes, data.phi_c,
                        primal_ID, face_node_coords),
                    jac=f_edge_jac, tol=tol,
                    bounds=((0, 1),), method='slsqp')
            if optimization.success:
                success = True
                if optimization.fun < minimum_phi:
                    best_opt = optimization
                    minimum_phi = optimization.fun
                    optimal_zeta = optimization.x.copy()
        if success:
            new_coords[index] = faceref_to_physical(optimal_zeta, face_node_coords)
        else:
            print(f'Oh no! Edge point of face {face_ID} failed to optimize!')

    # Use the average of the results from the two primal cells
    coords[:] = .5 * (new_coords[0] + new_coords[1])


def optimize_vol_point(mesh, data, problem, cell_ID, face_ID, edge_point_coords):
    def constraint_func(xi_eta, node_coords):
        '''
        All barycentric coordinates need to be positive.
        Since s and t are already positive by the bounds
        given, then only 1 - s - t needs to be constrained
        to be positive.

        Thank you to andreasdr on Stack Overflow:
        https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        '''
        xi, eta = xi_eta
        constraint = 1 - xi - eta
        return constraint
    def constraint_jac(xi_eta, node_coords):
        '''
        Compute the Jacobian of the constraint.
        '''
        jac = np.array([-1, -1])
        return jac
    coords = mesh.vol_points[cell_ID]
    # Get primal cell nodes
    node_IDs = mesh.primal_cell_to_nodes[cell_ID]
    node_coords = mesh.xy[node_IDs]
    # Various guesses around the primal cell
    guesses = [
            np.array([1/3, 1/3]),
            np.array([2/3, 1/6]),
            np.array([1/6, 2/3]),
            np.array([1/6, 1/6]),
    ]
    # Solve optimization problem for the new node locations,
    # by moving them as close as possible to the interface
    # (phi = 0) while still keeping the point within the
    # triangle
    success = False
    minimum_phi = 1e99
    constraints = [{
            'type': 'ineq',
            'fun': constraint_func,
            'jac': constraint_jac,
            'args': (node_coords,)}]
    #TODO: Figure out what this should be
    tol = 1e-2 * .5 * np.min(data.phi[node_IDs]**2)
    for guess in guesses:
        optimization = scipy.optimize.minimize(
                f_vol, guess,
                args=(data.phi_c, node_coords, cell_ID, edge_point_coords),
                jac=f_vol_jac, tol=tol,
                constraints=constraints,
                bounds=((0, None), (0, None)))
        if optimization.success:
            success = True
            if optimization.fun < minimum_phi:
                best_opt = optimization
                minimum_phi = optimization.fun
                optimal_xi_eta = optimization.x.copy()
    if success:
        coords[:] = elemref_to_physical(optimal_xi_eta, node_coords)
    else:
        print(f'Oh no! Volume point of primal cell {cell_ID} failed to optimize!')
