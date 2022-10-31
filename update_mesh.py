import numpy as np
import scipy.optimize

from transformation import (elemref_to_physical_jacobian, physical_to_elemref,
        faceref_to_elemref_jacobian, elemref_to_physical, faceref_to_physical)


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

    # Get the two primal cells on either side of this edge
    i_primals = mesh.nodes_to_primal_cells[i]
    j_primals = mesh.nodes_to_primal_cells[j]
    primal_cells = np.intersect1d(i_primals, j_primals)
    # Loop over both primal cells
    new_coords = np.empty((2, 2))
    for index, primal_ID in enumerate(primal_cells):
        print('Optimizing: ', coords, ' Primal =', primal_ID)
        guesses = np.linspace(0, 1, 5)
        success = False
        minimum_phi = 1e99
        # TODO: Figure out what this should be
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        tol = 1e-5 * .5 * np.min(data.phi[node_IDs]**2)
        for guess in guesses:
            print('Guess = ', guess)
            optimization = scipy.optimize.minimize(
                    f_edge, guess,
                    args=(mesh, data, problem, primal_ID, face_node_coords, data.t,),
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
    tol = 1e-5 * .5 * np.min(data.phi[node_IDs]**2)
    for guess in guesses:
        optimization = scipy.optimize.minimize(
                f_vol, guess,
                args=(data, problem, node_coords, cell_ID, edge_point_coords),
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
        if np.all(np.isclose(coords,
                np.array([-(.5 + 1/3) * (.4 / 19), -(.5 + 2/3) * (.4 /19)]))):
            breakpoint()
        coords[:] = elemref_to_physical(optimal_xi_eta, node_coords)
    else:
        print(f'Oh no! Volume point of primal cell {cell_ID} failed to optimize!')


def f_edge(zeta, mesh, data, problem, primal_ID, face_node_coords, t):
    """Compute objective function for an edge point."""
    # If the problem has an exact level set
    if problem.has_exact_phi:
        print("This part of the code (exact level set) may be outdated!")
        # Convert reference to physical coordinates
        coords = xi_to_xy(xi, xy1, xy2).reshape(1, -1)
        # Compute objective function
        f = .5 * problem.compute_exact_phi(coords, t)**2
    # If using the numerically computed phi
    else:
        # Compute physical coordinates
        xy = faceref_to_physical(zeta, face_node_coords)
        # Get nodes of this primal cell
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        elem_node_coords = mesh.xy[node_IDs]
        # Compute element reference coordinates
        xi_eta = physical_to_elemref(xy, elem_node_coords)
        # Compute phi from the fit
        phi = evaluate_level_set_fit(data, primal_ID, xi_eta)
        # Compute objective function
        f = .5 * phi**2
    return f
def f_edge_jac(zeta, mesh, data, problem, primal_ID, face_node_coords, t):
    """Compute the Jacobian of the objective function for an edge point."""
    # If the problem has an exact level set
    if problem.has_exact_phi:
        print("This part of the code (exact level set) may be outdated!")
        # Convert reference to physical coordinates
        coords = xi_to_xy(xi, xy1, xy2).reshape(1, -1)
        # Compute d(xy)/d(xi)
        dxy_dxi = xy2 - xy1
        # Compute d(phi)/d(xy) using chain rule
        dphi_dxy = (problem.compute_exact_phi(coords, t)
                * problem.compute_exact_phi_gradient(coords, t))
        # Combine (chain rule) to get d(phi)/d(xi)
        f_jac = np.dot(dphi_dxy[:, 0], dxy_dxi)
    # If using the numerically computed phi
    else:
        # Compute physical coordinates
        xy = faceref_to_physical(zeta, face_node_coords)
        # Get nodes of this primal cell
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        elem_node_coords = mesh.xy[node_IDs]
        # Compute element reference coordinates
        xi_eta = physical_to_elemref(xy, elem_node_coords)
        # Compute phi from the fit
        phi = evaluate_level_set_fit(data, primal_ID, xi_eta)
        # Compute d(phi)/d(xi, eta) from the fit
        d_phi_d_xi_eta = evaluate_level_set_gradient(data, primal_ID, xi_eta)
        # Compute Jacobian
        d_xi_eta_d_zeta = faceref_to_elemref_jacobian(elem_node_coords, face_node_coords)
        # Combine using chain rule to get d(phi)/d(zeta)
        d_phi_d_zeta = d_phi_d_xi_eta @ d_xi_eta_d_zeta
        # Use chain rule to compute d(f)/d(zeta)
        f_jac = phi * d_phi_d_zeta
    return f_jac
factor = 0.1
def f_vol(xi_eta, data, problem, node_coords, primal_ID, edge_point_coords):
    """Compute objective function for a volume point."""
    xy = elemref_to_physical(xi_eta, node_coords)
    # If the problem has an exact level set
    if problem.has_exact_phi:
        coords = get_coords_from_barycentric(bary, node_coords).reshape(1, -1)
        # Compute objective function
        f = .5 * problem.compute_exact_phi(coords, data.t)**2
    # If using the numerically computed phi
    else:
        # Evaluate level set
        phi = evaluate_level_set_fit(data, primal_ID, xi_eta)
        # Compute objective function
        f = .5 * phi**2
        for edge_point in edge_point_coords:
            f += factor * .5 * np.linalg.norm(xy - edge_point)**2
    return f
def f_vol_jac(xi_eta, data, problem, node_coords, primal_ID, edge_point_coords):
    """Compute the Jacobian of the objective function for a volume point."""
    xy = elemref_to_physical(xi_eta, node_coords)
    # If the problem has an exact level set
    if problem.has_exact_phi:
        coords = get_coords_from_barycentric(bary, node_coords).reshape(1, -1)
        xy1 = node_coords[0]
        xy2 = node_coords[1]
        xy3 = node_coords[2]
        # Compute d(f)/d(xy)
        df_dxy = (problem.compute_exact_phi(coords, data.t)
                * problem.compute_exact_phi_gradient(coords, data.t))
        # Compute d(xy)/d(bary)
        dxy_dbary = np.array([xy1 - xy3, xy2 - xy3])
        # Combine with chain rule to get d(f)/d(bary)
        f_jac = df_dxy @ dxy_dbary
    # If using the numerically computed phi
    else:
        # Evaluate level set
        phi = evaluate_level_set_fit(data, primal_ID, xi_eta)
        # Evaluate level set gradient
        d_phi_d_xi_eta = evaluate_level_set_gradient(data, primal_ID, xi_eta)
        # Use chain rule to compute d(f)/d(xi, eta)
        f_jac = phi * d_phi_d_xi_eta
        # Add contribution from edge point terms
        jac = elemref_to_physical_jacobian(node_coords)
        for edge_point in edge_point_coords:
            f_jac += factor * (xy - edge_point) @ jac
    return f_jac
