import numpy as np


class Problem:
    '''
    Parent class for all problem definitions.
    '''
    def __init__(self, xy):
        self.xy = xy
        self.n = xy.shape[0]


class RiemannProblem(Problem):
    '''
    Class for defining ICs and BCs for a Riemann problem.
    '''

    # In a Riemann problem, the left state is usually referred to as state 4,
    # and the right state is usually state 1. This naming is used below.

    # Left state (rho, p, u, v, phi)
    state_4 = np.array([
            1, 100, 0, 1e5, -1
    ])

    # Right state (rho, p, u, v, phi)
    state_1 = np.array([
            .125, 50, 0, 1e4, 1
    ])

    # Ratio of specific heats
    g = 1.4

    def get_initial_conditions(self):
        # Unpack
        r4, u4, v4, p4, phi4 = self.state_4
        r1, u1, v1, p1, phi1 = self.state_1
        g = self.g

        # Get initial conditions as conservatives
        W4 = primitive_to_conservative(r4, u4, v4, p4, g)
        W1 = primitive_to_conservative(r1, u1, v1, p1, g)

        # Set left and right state with the contact at x = 0
        U = np.empty((self.n, 4))
        U[self.xy[:, 0] <= 0] = W4
        U[self.xy[:, 0] > 0] = W1
        # Phi is set to be zero at the initial contact (x = 0)
        phi = self.xy[:, 0].copy()
        phi /= np.max(phi)
        return U, phi


def primitive_to_conservative(r, u, v, p, g):
    W = np.empty(4)
    W[0] = r
    W[1] = r * u
    W[2] = r * v
    W[3] = p / (g - 1) + .5 * r * (u**2 + v**2)
    return W
