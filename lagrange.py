import numpy as np


class Lagrange:
    '''
    Base class for performing Langrange interpolation and quadrature.
    '''
    def __init__(self, coords):
        self.coords = coords
        self.get_jacobian(self.quad_pts)
        self.get_area()

    def get_area(self):
        self.area = np.sum(self.quad_wts * self.jac)


class LagrangeSegment(Lagrange):
    '''
    Base class for performing Langrange interpolation and quadrature on line segments.
    '''

    # The quadrature rule below comes from:
    # https://en.wikipedia.org/wiki/Gaussian_quadrature
    # This is the 2-point Guass-Legendre quadrature rule from this page.
    # Quadrature points
    quad_pts = np.array([-1/(2*np.sqrt(3)) + 1/2, 1/(2*np.sqrt(3)) + 1/2])
    # Quadrature weights
    quad_wts = np.array([1/2, 1/2])

    def get_jacobian(self, x):
        '''
        Compute Jacobian of the transformation at a set of points.

        Inputs:
        -------
        x - array of points (n,)

        Outputs:
        --------
        self.jac - determinant of Jacobian evaluated at x (n,)
        '''
        self.jac = np.einsum(
                'ij, j -> i', self.get_basis_gradient(x), self.coords)


class LagrangeTriangle(Lagrange):
    '''
    Base class for performing Langrange interpolation and quadrature on triangles.
    '''

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

    def get_jacobian(self, xy):
        '''
        Compute Jacobian of the transformation at a set of points.

        Inputs:
        -------
        xy - array of points (n, 2)

        Outputs:
        --------
        self.jac - determinant of Jacobian evaluated at xy (n,)
        '''
        jac_matrix = np.einsum(
                'ijk, jl -> ikl', self.get_basis_gradient(xy), self.coords)
        self.jac = np.linalg.det(jac_matrix)


class LagrangeSegmentP1(LagrangeSegment):
    '''
    Class for performing Langrange interpolation and quadrature on line segments.

    Hardcoded for first order interpolants.
    '''
    # Number of basis functions
    nb = 2

    # Nodes for the interpolation on the reference segment
    xy_nodes = np.array([ 0, 1 ])

    def get_basis_values(self, x):
        '''
        Compute values of Lagrange basis at a set of points.

        Inputs:
        -------
        x - array of points (n,)

        Outputs:
        --------
        basis_val - array of basis values evaluated at x (n, nb)
        '''
        n = x.shape[0]
        basis_val = np.empty((n, self.nb))
        # Evaluate Lagrange basis functions
        basis_val[:, 0] = 1 - x
        basis_val[:, 1] = x
        return basis_val

    def get_basis_gradient(self, xy):
        '''
        Compute gradient of Lagrange basis at a set of points.

        Inputs:
        -------
        x - array of points (n,)

        Outputs:
        --------
        basis_grad - array of basis values evaluated at xy (n, nb)
        '''
        n = xy.shape[0]
        basis_grad = np.empty((n, self.nb))
        # Evaluate Lagrange basis gradient
        basis_grad[:, 0] = -1
        basis_grad[:, 1] = 1
        return basis_grad


class LagrangeSegmentP2(LagrangeSegment):
    '''
    Class for performing Langrange interpolation and quadrature on line segments.

    Hardcoded for second order interpolants.
    '''
    # Number of basis functions
    nb = 3

    # Nodes for the interpolation on the reference segment
    xy_nodes = np.array([ 0, 1/2, 1 ])

    def get_basis_values(self, x):
        '''
        Compute values of Lagrange basis at a set of points.

        Inputs:
        -------
        x - array of points (n,)

        Outputs:
        --------
        basis_val - array of basis values evaluated at x (n, nb)
        '''
        n = x.shape[0]
        basis_val = np.empty((n, self.nb))
        # Evaluate Lagrange basis functions
        basis_val[:, 0] = 2 * (x - 1/2) * (x - 1)
        basis_val[:, 1] = -4 * x * (x - 1)
        basis_val[:, 2] = 2 * x * (x - 1/2)
        return basis_val

    def get_basis_gradient(self, x):
        '''
        Compute gradient of Lagrange basis at a set of points.

        Inputs:
        -------
        x - array of points (n,)

        Outputs:
        --------
        basis_grad - array of basis values evaluated at xy (n, nb)
        '''
        n = x.shape[0]
        basis_grad = np.empty((n, self.nb))
        # Evaluate Lagrange basis gradient
        basis_grad[:, 0] = 4 * x - 3
        basis_grad[:, 1] = 4 - 8 * x
        basis_grad[:, 2] = 4 * x - 1
        return basis_grad


class LagrangeTriangleP1(LagrangeTriangle):
    '''
    Class for performing Langrange interpolation and quadrature on triangles.

    Hardcoded for first order interpolants.
    '''
    # Number of basis functions
    nb = 3

    # Nodes for the interpolation on the reference triangle
    xy_nodes = np.array([ [0, 0], [1, 0], [0, 1] ])

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
        basis_val[:, 0] = 1 - x - y
        basis_val[:, 1] = x
        basis_val[:, 2] = y
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
        basis_grad[:, 0, 0] = -1
        basis_grad[:, 1, 0] = 1
        basis_grad[:, 2, 0] = 0
        # y - direction
        basis_grad[:, 0, 1] = -1
        basis_grad[:, 1, 1] = 0
        basis_grad[:, 2, 1] = 1
        return basis_grad


class LagrangeTriangleP2(LagrangeTriangle):
    '''
    Class for performing Langrange interpolation and quadrature on triangles.

    Hardcoded for second order interpolants.
    '''
    # Number of basis functions
    nb = 6

    # Nodes for the interpolation on the reference triangle
    xy_nodes = np.array([
        [0, 0], [1/2, 0], [1, 0],
        [0, 1/2], [1/2, 1/2],
        [0, 1]
    ])

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
