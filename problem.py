import numpy as np

from exact_solution import exact_solution


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

    def __init__(self, xy, t_list):
        super().__init__(xy)
        # Unpack
        r4, u4, v4, p4, phi4 = self.state_4
        r1, u1, v1, p1, phi1 = self.state_1
        g = self.g
        # Get exact solution to this Riemann Problem
        exact_solution(
                r4, p4, u4, v4, r1, p1, u1, v1, g, t_list)

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

    def compute_ghost_state(self, U, U_ghost, bc_type):
        # Unpack
        r4, u4, v4, p4, _ = self.state_4
        r1, u1, v1, p1, _ = self.state_1
        g = self.g
        # Loop over each boundary cell
        for i in range(U_ghost.shape[0]):
            # Get the type of this boundary
            cell_ID, bc = bc_type[i]
            # Get primitives
            V = conservative_to_primitive(*U[cell_ID], g)
            # Compute wall ghost state
            if bc == 1:
                # The density and pressure are kept the same in the ghost state
                r = V[0]
                p = V[3]
                # The x-direction velocity is not changed since the wall is
                # horizontal
                u = V[1]
                # The y-direction velocity is flipped in sign, since the wall is
                # horizontal
                v = -V[2]
            # Compute inflow ghost state
            elif bc == 2:
                # Set state to the left state
                r = r4
                p = p4
                u = u4
                v = v4
            # Compute outflow ghost state
            elif bc == 3:
                # Set state to the right state
                p = p1
                r = r1
                u = u1
                v = v1
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')
            # Compute ghost state
            U_ghost[i] = primitive_to_conservative(r, u, v, p, g)

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        # Unpack
        *_, phi4 = self.state_4
        *_, phi1 = self.state_1
        # Loop over each boundary cell
        for i in range(phi_ghost.shape[0]):
            cell_ID, bc = bc_type[i]
            # Compute wall ghost state
            if bc == 1:
                # The density and pressure are kept the same in the ghost state, so
                # seems reasonable to keep phi the same as well since it's a scalar
                phi_ghost[i] = phi[cell_ID]
            # Compute inflow ghost state
            elif bc == 2:
                # Set state to the left state
                phi_ghost[i] = phi4
            # Compute outflow ghost state
            elif bc == 3:
                # Set state to the right state
                phi_ghost[i] = phi1
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')


def primitive_to_conservative(r, u, v, p, g):
    W = np.empty(4)
    W[0] = r
    W[1] = r * u
    W[2] = r * v
    W[3] = p / (g - 1) + .5 * r * (u**2 + v**2)
    return W

def conservative_to_primitive(r, ru, rv, re, g):
    V = np.empty(4)
    V[0] = r
    V[1] = ru / r
    V[2] = rv / r
    V[3] = (re - .5 * (ru**2 + rv**2) / r) * (g - 1)
    return V

