import numpy as np


class LagrangeTriangle:
    '''
    Class for performing Langrange interpolation and quadrature on triangles.

    Hardcoded for second order interpolants.
    '''
    # Number of basis functions
    nb = 6

    # The quadrature rule below comes from:
    # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    # This is the STRANG1 rule from this page (3 point, second order accurate).
    # Quadrature points
    quad_pts = np.array([
            [0.66666666666666666667, 0.16666666666666666667],
            [0.16666666666666666667, 0.66666666666666666667],
            [0.16666666666666666667, 0.16666666666666666667],
    ])
    # Quadrature weights
    quad_wts = np.array([
            0.33333333333333333333,
            0.33333333333333333333,
            0.33333333333333333333,
    ])/2

    # Nodes for the interpolation on the reference triangle
    xy_nodes = np.array([
        [0, 0], [1/2, 0], [1, 0],
        [0, 1/2], [1/2, 1/2],
        [0, 1]
    ])

    def __init__(self, coords):
        self.coords = coords
        self.get_area()

    def get_basis_values(self, xy):
        '''
        Compute values of Lagrange basis at a set of points.

        Inputs:
        -------
        xy - array of points (n, 2)

        Outputs:
        --------
        basis_val - array of basis values evaluated at xy (n, nb)
        '''
        n = xy.shape[0]
        basis_val = np.empty((n, self.nb))
        x = xy[:, 0]
        y = xy[:, 1]
        # Evaluate Lagrange basis functions
        basis_val[:, 0] = 2 * (x + y - 1) * (x + y - 1/2)
        basis_val[:, 1] = -4*x * (x + y - 1)
        basis_val[:, 2] = 2*x * (x - 1/2)
        basis_val[:, 3] = -4*y * (x + y - 1)
        basis_val[:, 4] = 4 * x * y
        basis_val[:, 5] = 2*y * (y - 1/2)
        return basis_val

    def get_basis_gradient(self, xy):
        '''
        Compute gradient of Lagrange basis at a set of points.

        Inputs:
        -------
        xy - array of points (n, 2)

        Outputs:
        --------
        basis_grad - array of basis values evaluated at xy (n, nb)
        '''
        n = xy.shape[0]
        basis_grad = np.empty((n, self.nb, 2))
        x = xy[:, 0]
        y = xy[:, 1]
        # Evaluate Lagrange basis gradient
        # x - direction
        basis_grad[:, 0, 0] = 4 * x + 4 * y - 3
        basis_grad[:, 1, 0] = -4 * (2*x + y - 1)
        basis_grad[:, 2, 0] = 4 * x - 1
        basis_grad[:, 3, 0] = -4 * y
        basis_grad[:, 4, 0] = 4 * y
        basis_grad[:, 5, 0] = 0
        # y - direction
        basis_grad[:, 0, 1] = 4 * x + 4 * y - 3
        basis_grad[:, 1, 1] = -4 * x
        basis_grad[:, 2, 1] = 0
        basis_grad[:, 3, 1] = -4 * (x + 2*y - 1)
        basis_grad[:, 4, 1] = 4 * x
        basis_grad[:, 5, 1] = 4 * y - 1
        return basis_grad

    def get_jacobian(self, xy):
        '''
        Compute Jacobian of the transformation at a set of points.

        Inputs:
        -------
        xy - array of points (n, 2)

        Outputs:
        --------
        jac - determinant of Jacobian evaluated at xy (n,)
        '''
        jac_matrix = np.einsum(
                'ijk, jl -> ikl', self.get_basis_gradient(xy), self.coords)
        return np.linalg.det(jac_matrix)

    def get_area(self):
        jac = self.get_jacobian(self.quad_pts)
        self.area = np.sum(self.quad_wts * jac)
