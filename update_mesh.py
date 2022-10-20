import numpy as np
import scipy.optimize


def update_mesh(mesh, data, problem):
    '''
    Update the dual mesh to fit the interface better.
    '''
    for face_ID in range(mesh.n_faces):
        # TODO: Only works for interior faces
        if mesh.is_boundary(face_ID): continue

        # Get dual mesh neighbors
        i, j = mesh.edge[face_ID]
        # Check for interface
        if data.phi[i] * data.phi[j] < 0:
            # If it's an interface, move the face points towards phi = 0
            for i_point in range(3):
                # -- Edge points -- #
                if i_point == 1:
                    coords = optimize_edge_point(mesh, data, problem, i, j, face_ID)
                # -- Volume points -- #
                else:
                    #TODO
                    pass
                    coords = optimize_vol_point(mesh, data, problem, i_point, face_ID)


def optimize_edge_point(mesh, data, problem, i, j, face_ID):
    coords = mesh.edge_points[mesh.face_points[face_ID, 1]]
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
    # TODO: Figure out what this should be
    tol = 1e-10
    # Loop over both primal cells
    new_coords = np.empty((2, 2))
    for index, primal_ID in enumerate(primal_cells):
        guesses = np.linspace(0, 1, 5)
        success = False
        minimum_phi = 1e99
        for guess in guesses:
            optimization = scipy.optimize.minimize(
                    f_edge, guess,
                    args=(mesh, data, problem, primal_ID, mesh.xy[i], mesh.xy[j], data.t,),
                    jac=f_edge_jac, tol=tol,
                    bounds=((0, 1),), method='slsqp')
            if optimization.success:
                success = True
                best_opt = optimization
                if optimization.fun < minimum_phi:
                    minimum_phi = optimization.fun
                    optimal_xi = optimization.x.copy()
        if success:
            new_coords[index] = xi_to_xy(optimal_xi, mesh.xy[i],
                    mesh.xy[j])
        else:
            print(f'Oh no! Edge point of face {face_ID} failed to optimize!')

    # Use the average of the results from the two primal cells
    coords[:] = .5 * (new_coords[0] + new_coords[1])


def optimize_vol_point(mesh, data, problem, i_point, face_ID):
    def constraint_func(bary, node_coords):
        '''
        All barycentric coordinates need to be positive.
        Since s and t are already positive by the bounds
        given, then only 1 - s - t needs to be constrained
        to be positive.

        Thank you to andreasdr on Stack Overflow:
        https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        '''
        constraint = 1 - bary[0] - bary[1]
        return constraint
    def constraint_jac(bary, node_coords):
        '''
        Compute the Jacobian of the constraint.
        '''
        jac = np.array([-1, -1])
        return jac
    cell_ID = mesh.face_points[face_ID, i_point]
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
    #tol = 1e-12
    tol = 1e-2 * .5 * np.min(data.phi[node_IDs]**2)
    for guess in guesses:
        optimization = scipy.optimize.minimize(
                f_vol, guess,
                args=(mesh, data, problem, node_IDs, node_coords, data.t,),
                jac=f_vol_jac, tol=tol,
                constraints=constraints,
                bounds=((0, None), (0, None)))
        if optimization.success:
            success = True
            best_opt = optimization
            if optimization.fun < minimum_phi:
                minimum_phi = optimization.fun
                optimal_bary = optimization.x.copy()
    if success:
        coords[:] = get_coords_from_barycentric(optimal_bary, node_coords)
    else:
        print(f'Oh no! Volume point of primal cell {cell_ID} failed to optimize!')


def get_coords_from_barycentric(bary, node_coords):
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    return bary[0]*xy1 + bary[1]*xy2 + (1 - bary[0] -
            bary[1])*xy3
# TODO Solve this linear system exactly
def get_barycentric_from_coords(coords, node_coords):
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    A = np.empty((2, 2))
    A[:, 0] = xy1 - xy3
    A[:, 1] = xy2 - xy3
    b = coords.flatten() - xy3
    return np.linalg.solve(A, b)
def xi_to_xy(xi, xy1, xy2):
    return xi*xy2 + (1 - xi) * xy1
def f_edge(xi, mesh, data, problem, primal_ID, xy1, xy2, t):
    """Compute objective function for an edge point."""
    # Convert reference to physical coordinates
    coords = xi_to_xy(xi, xy1, xy2).reshape(1, -1)
    # If the problem has an exact level set
    if problem.has_exact_phi:
        # Compute objective function
        f = .5 * problem.compute_exact_phi(coords, t)**2
    # If using the numerically computed phi
    else:
        # Get nodes of this primal cell
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        node_coords = mesh.xy[node_IDs]
        # Get barycentric coordinates
        bary = get_barycentric_from_coords(coords, node_coords)
        # Use barycentric interpolation to compute phi
        basis = np.array([bary[0], bary[1], 1 - bary[0] - bary[1]])
        phi = np.dot(basis, data.phi[node_IDs])
        # Compute objective function
        f = .5 * phi**2
    return f
def f_edge_jac(xi, mesh, data, problem, primal_ID, xy1, xy2, t):
    """Compute the Jacobian of the objective function for an edge point."""
    # Convert reference to physical coordinates
    coords = xi_to_xy(xi, xy1, xy2).reshape(1, -1)
    # Compute d(xy)/d(xi)
    dxy_dxi = xy2 - xy1
    # If the problem has an exact level set
    if problem.has_exact_phi:
        # Compute d(phi)/d(xy) using chain rule
        dphi_dxy = (problem.compute_exact_phi(coords, t)
                * problem.compute_exact_phi_gradient(coords, t))
        # Combine (chain rule) to get d(phi)/d(xi)
        f_jac = np.dot(dphi_dxy[:, 0], dxy_dxi)
    # If using the numerically computed phi
    else:
        # Get nodes of this primal cell
        node_IDs = mesh.primal_cell_to_nodes[primal_ID]
        node_coords = mesh.xy[node_IDs]
        # Get barycentric coordinates
        bary = get_barycentric_from_coords(coords, node_coords)
        # Use barycentric interpolation to compute phi
        basis = np.array([bary[0], bary[1], 1 - bary[0] - bary[1]])
        phi = np.dot(basis, data.phi[node_IDs])
        # Compute d(phi)/d(bary)
        phi1, phi2, phi3 = data.phi[node_IDs]
        dphi_dbary = np.array([phi1 - phi3, phi2 - phi3])
        # Compute d(bary)/d(xy) as the inverse of d(xy)/d(bary)
        xy1 = node_coords[0]
        xy2 = node_coords[1]
        xy3 = node_coords[2]
        dbary_dxy = np.linalg.inv(np.array([xy1 - xy3, xy2 - xy3]))
        # Combine using chain rule to get d(phi)/d(xi)
        dphi_dxi = dphi_dbary @ dbary_dxy @ dxy_dxi
        # Use chain rule to compute d(f)/d(xi)
        f_jac = phi * dphi_dxi
    return f_jac
def f_vol(bary, mesh, data, problem, node_IDs, node_coords, t):
    """Compute objective function for a volume point."""
    coords = get_coords_from_barycentric(bary, node_coords).reshape(1, -1)
    # If the problem has an exact level set
    if problem.has_exact_phi:
        # Compute objective function
        f = .5 * problem.compute_exact_phi(coords, t)**2
    # If using the numerically computed phi
    else:
        # Use barycentric interpolation to compute phi
        basis = np.array([bary[0], bary[1], 1 - bary[0] - bary[1]])
        phi = np.dot(basis, data.phi[node_IDs])
        # Compute objective function
        f = .5 * phi**2
    return f
def f_vol_jac(bary, mesh, data, problem, node_IDs, node_coords, t):
    """Compute the Jacobian of the objective function for a volume point."""
    coords = get_coords_from_barycentric(bary, node_coords).reshape(1, -1)
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    # If the problem has an exact level set
    if problem.has_exact_phi:
        # Compute d(f)/d(xy)
        df_dxy = (problem.compute_exact_phi(coords, t)
                * problem.compute_exact_phi_gradient(coords, t))
        # Compute d(xy)/d(bary)
        dxy_dbary = np.array([xy1 - xy3, xy2 - xy3])
        # Combine with chain rule to get d(f)/d(bary)
        f_jac = df_dxy @ dxy_dbary
    # If using the numerically computed phi
    else:
        # Use barycentric interpolation to compute phi
        basis = np.array([bary[0], bary[1], 1 - bary[0] - bary[1]])
        phi = np.dot(basis, data.phi[node_IDs])
        # Compute d(phi)/d(bary)
        phi1, phi2, phi3 = data.phi[node_IDs]
        dphi_dbary = np.array([phi1 - phi3, phi2 - phi3])
        # Use chain rule to compute d(f)/d(bary)
        f_jac = phi * dphi_dbary
    return f_jac
