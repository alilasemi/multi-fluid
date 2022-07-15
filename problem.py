import numpy as np

from exact_solution import exact_solution


class Problem:
    '''
    Parent class for all problem definitions.
    '''
    exact = False
    def __init__(self, xy, t_list):
        self.xy = xy
        self.t_list = t_list
        self.n = xy.shape[0]


class RiemannProblem(Problem):
    '''
    Class for defining ICs and BCs for a Riemann problem.
    '''

    # In a Riemann problem, the left state is usually referred to as state 4,
    # and the right state is usually state 1. This naming is used below.

    # Left state (rho, u, v, p, phi)
    state_4 = np.array([
            1, 100, 0, 1e5, -1
    ])

    # Right state (rho, u, v, p, phi)
    state_1 = np.array([
            .125, 50, 0, 1e4, 1
    ])

    # Ratio of specific heats
    g = 1.4

    # The exact solution is defined
    exact = True

    def __init__(self, xy, t_list):
        super().__init__(xy, t_list)
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


class AdvectedContact(RiemannProblem):
    '''
    Class for a contact wave advected at a constant velocity. This is based off
    of a special case of a Riemann problem.
    '''
    # Left state (rho, u, v, p, phi)
    state_4 = np.array([
            1, 300, 0, 1e5, -1
    ])

    # Right state (rho, u, v, p, phi)
    state_1 = np.array([
            .125, 300, 0, 1e5, 1
    ])

class AdvectedBubble(Problem):
    '''
    Class for a bubble advecting at constant velocity.
    '''

    # bubble state (rho, u, v, p, phi)
    bubble = np.array([
            .125, 50, 0, 1e5, -1
    ])

    # Ambient state (rho, u, v, p, phi)
    ambient = np.array([
            1, 50, 0, 1e5, 1
    ])

    # Radius of bubble
    radius = .25

    # Ratio of specific heats
    g = 1.4

    def get_initial_conditions(self):
        # Unpack
        r0, u0, v0, p0, phi0 = self.ambient
        r1, u1, v1, p1, phi1 = self.bubble
        g = self.g

        # Get initial conditions as conservatives
        W0 = primitive_to_conservative(r0, u0, v0, p0, g)
        W1 = primitive_to_conservative(r1, u1, v1, p1, g)

        # Set bubble in the center of the domain
        U = W0 * np.ones((self.n, 4))
        indices = np.nonzero(self.xy[:, 0]**2 + self.xy[:, 1]**2 <
                self.radius**2)
        U[indices] = W1
        # Phi is set to be zero at the initial bubble
        phi = self.xy[:, 0]**2 + self.xy[:, 1]**2 - self.radius**2
        phi /= np.max(phi)
        return U, phi

    def compute_ghost_state(self, U, U_ghost, bc_type):
        # Unpack
        r0, u0, v0, p0, _ = self.ambient
        r1, u1, v1, p1, _ = self.bubble
        g = self.g
        # Loop over each boundary cell
        for i in range(U_ghost.shape[0]):
            # Get the type of this boundary
            cell_ID, bc = bc_type[i]
            # Get primitives
            V = conservative_to_primitive(*U[cell_ID], g)
            # Compute wall ghost state, regardless of if it's marked as a wall,
            # inflow or outflow
            if bc in [1, 2, 3]:
                # The density and pressure are kept the same in the ghost state
                r = V[0]
                p = V[3]
                # The x-direction velocity is not changed since the wall is
                # horizontal
                u = V[1]
                # The y-direction velocity is flipped in sign, since the wall is
                # horizontal
                v = -V[2]
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')
            # Compute ghost state
            U_ghost[i] = primitive_to_conservative(r, u, v, p, g)

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        # Unpack
        *_, phi0 = self.ambient
        *_, phi1 = self.bubble
        # Loop over each boundary cell
        for i in range(phi_ghost.shape[0]):
            cell_ID, bc = bc_type[i]
            # Compute wall ghost state, regardless of if it's marked as a wall,
            # inflow or outflow
            if bc in [1, 2, 3]:
                # Just use the initial value as the boundary value
                phi_ghost[i] = (self.xy[cell_ID, 0]**2 + self.xy[cell_ID, 1]**2
                        - self.radius**2)
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')


class TaylorGreen(Problem):
    '''
    Class for the Taylor Green vortex problem.
    '''

    # bubble state (rho, u, v, p, phi)
    bubble = np.array([
            .125, 50, 0, 1e5, -1
    ])

    # Ambient state (rho, u, v, p, phi)
    ambient = np.array([
            1, 50, 0, 1e5, 1
    ])

    # Radius of bubble
    radius = .25

    # Ratio of specific heats
    g = 1.4

    def get_initial_conditions(self):
        # Unpack
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        g = self.g
        # Taylor Green initial velocity field
        u =  np.cos(x) * np.sin(y)
        v = -np.sin(x) * np.cos(y)
        # Pick a number for density (it's an "incompressible" problem)
        r = 1
        # Pressure comes from momentum equations
        p = (-r / 4) * (np.cos(2*x) + np.cos(2*y))
        # TODO: What is going on here? TVG is a viscous case :(
        p += 2

        # Get initial conditions as conservatives
        U = np.empty((self.n, 4))
        for i in range(self.n):
            U[i] = primitive_to_conservative(r, u[i], v[i], p[i], g)

        # Phi doesn't really matter for this problem, but I guess a circle is
        # cool
        phi = self.xy[:, 0]**2 + self.xy[:, 1]**2 - self.radius**2
        phi /= np.max(phi)
        return U, phi

    def compute_ghost_state(self, U, U_ghost, bc_type):
        g = self.g
        # Loop over each boundary cell
        for i in range(U_ghost.shape[0]):
            # Get the type of this boundary
            cell_ID, bc = bc_type[i]
            # Get primitives
            V = conservative_to_primitive(*U[cell_ID], g)
            # Compute wall ghost state, regardless of if it's marked as a wall,
            # inflow or outflow
            if bc in [1, 2, 3]:
                # The density and pressure are kept the same in the ghost state
                r = V[0]
                p = V[3]
                # The x-direction velocity is not changed since the wall is
                # horizontal
                u = V[1]
                # The y-direction velocity is flipped in sign, since the wall is
                # horizontal
                v = -V[2]
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')
            # Compute ghost state
            U_ghost[i] = primitive_to_conservative(r, u, v, p, g)

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        # Loop over each boundary cell
        for i in range(phi_ghost.shape[0]):
            cell_ID, bc = bc_type[i]
            # Compute wall ghost state, regardless of if it's marked as a wall,
            # inflow or outflow
            if bc in [1, 2, 3]:
                # Just use the initial value as the boundary value
                phi_ghost[i] = (self.xy[cell_ID, 0]**2 + self.xy[cell_ID, 1]**2
                        - self.radius**2)
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

