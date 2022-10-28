import numpy as np

def elemref_to_physical(xi_eta, node_coords):
    """Convert from element reference space to physical space.

    Inputs:
        xi_eta: Two values of xi and eta, the element reference coordinates
        node_coords: Coordinates in physical space of the element nodes
    Outputs:
        xy: Resulting coordinates in physical space
    """
    xi, eta = xi_eta
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    xy = (1 - xi - eta) * xy1 + xi * xy2 + eta * xy3
    return xy

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

def elemref_to_physical_jacobian(node_coords):
    """Compute Jacobian of physical space with respect to element reference
    space.

    Inputs:
        xi_eta: Two values of xi and eta, the element reference coordinates
        node_coords: Coordinates in physical space of the element nodes
    Outputs:
        jac: Jacobian matrix, d(x, y)/d(xi, eta)
    """
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy3 = node_coords[2]
    jac = np.empty((2, 2))
    jac[:, 0] = -xy1 + xy2
    jac[:, 1] = -xy1 + xy3
    return jac

def faceref_to_physical(zeta, node_coords):
    """Convert from face reference space to physical space.

    Inputs:
        zeta: Value of zeta, the face reference coordinate
        node_coords: Coordinates in physical space of the face nodes
    Outputs:
        xy: Resulting coordinates in physical space
    """
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    xy = zeta*xy2 + (1 - zeta) * xy1
    return xy

def faceref_to_physical_jacobian(node_coords):
    """Compute Jacobian of physical space with respect to face reference
    space.

    Inputs:
        node_coords: The coordinates in physical space of the face nodes
    Outputs:
        jac: Jacobian array, d(x, y)/d(zeta)
    """
    xy1 = node_coords[0]
    xy2 = node_coords[1]
    jac = xy2 - xy1
    return jac

def faceref_to_elemref_jacobian(elem_node_coords, face_node_coords):
    """Compute Jacobian of element reference space with respect to face
    reference space by using the chain rule.

    Inputs:
        elem_node_coords: The coordinates in physical space of the element nodes
        face_node_coords: The coordinates in physical space of the face nodes
    Outputs:
        d_xi_eta_d_zeta: Jacobian array, d(xi, eta)/d(zeta)
    """
    d_xy_d_xi_eta = elemref_to_physical_jacobian(elem_node_coords)
    d_xy_d_zeta = faceref_to_physical_jacobian(face_node_coords)
    d_xi_eta_d_zeta = np.linalg.inv(d_xy_d_xi_eta) @ d_xy_d_zeta
    return d_xi_eta_d_zeta
