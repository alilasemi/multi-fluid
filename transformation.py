import numpy as np

def elemref_to_physical(xi_eta, node_coords):
    """Convert from element reference space to physical space.

    Inputs:
    xi_eta: The two values of xi and eta, the element reference coordinates
    node_coords: The coordinates in physical space of the element nodes
    """
    xi, eta = xi_eta
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    return (1 - xi - eta) * xy1 + xi * xy2 + eta * xy3

# TODO Solve this linear system exactly
def physical_to_elemref(xy, node_coords):
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    A = np.empty((2, 2))
    A[:, 0] = -xy1 + xy2
    A[:, 1] = -xy1 + xy3
    b = xy - xy1
    return np.linalg.solve(A, b)

def faceref_to_physical(zeta, node_coords):
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    return zeta*xy2 + (1 - zeta) * xy1

def physical_to_faceref(zeta, node_coords):
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    return zeta*xy2 + (1 - zeta) * xy1

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
        # Compute phi from the fit
        phi = evaluate_level_set_fit(data, primal_ID, bary)
        # Compute objective function
        f = .5 * phi**2
        print('xi, f = ', xi, f)
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
        # Compute phi from the fit
        phi = evaluate_level_set_fit(data, primal_ID, bary)
        # Compute d(phi)/d(bary) from the fit
        dphi_dbary = evaluate_level_set_gradient(data, primal_ID, bary)
        print('bary ', bary)
        # Compute d(bary)/d(xy) as the inverse of d(xy)/d(bary)
        xy1 = node_coords[0]
        xy2 = node_coords[1]
        xy3 = node_coords[2]
        dbary_dxy = np.linalg.inv(np.array([xy1 - xy3, xy2 - xy3]).T)
        # Combine using chain rule to get d(phi)/d(xi)
        dphi_dxi = dphi_dbary @ dbary_dxy @ dxy_dxi
        # Use chain rule to compute d(f)/d(xi)
        f_jac = phi * dphi_dxi
        print('f_jac = ', f_jac)
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

if __name__ == "__main__":
    # TODO: Put these in unit tests
    xy1 = np.array([-.067, -.067])
    xy2 = np.array([-.022, -.022])
    class MeshMock:
        primal_cell_to_nodes = np.array([[0, 1, 2]])
        #xy = np.array([[1, 0], [0, 1], [0, 0]])
        xy = np.array([[-0.06666667, -0.06666667],
               [-0.02222222, -0.02222222],
               [-0.06666667, -0.02222222]])
    class DataMock:
        #phi_c = np.array([[1, -1, -1, 0, 0, 0]])
        phi_c = np.array([[ 0.05411418,  0.01338193, -0.02645798, -0.07596969,  0.00268392,
                0.04843283]])
    class ProblemMock:
        has_exact_phi = False
    primal_ID = 0
    fe = f_edge(0.053159, MeshMock(), DataMock(), ProblemMock, primal_ID, xy1, xy2, None)
    #fej = f_edge_jac(0, MeshMock(), DataMock(), ProblemMock, primal_ID, xy1, xy2, None)
    breakpoint()
