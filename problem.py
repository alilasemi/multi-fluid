import numpy as np

from exact_solution import exact_solution
import matplotlib.patches


class Problem:
    '''
    Parent class for all problem definitions.
    '''
    exact = False
    def __init__(self, xy, t_list):
        self.xy = xy
        self.t_list = t_list
        self.n = xy.shape[0]

    def compute_ghost_wall(self, V, bc_area_normal, wall_velocity=None):
        '''
        Compute the ghost state for a wall BC.

        Inputs:
        -------
        V - array of primitive variables (4,)
        bc_area_normal - array of area-weighted normal vector (2,)
        wall_velocity - velocity of wall (2,)

        Outputs:
        --------
        V_ghost - array of primitive ghost state (4,)
        '''
        if wall_velocity is None:
            wall_velocity = np.zeros(2)
        # The density and pressure are kept the same in the ghost state
        r = V[0]
        p = V[3]
        # Compute unit normal vector
        n_hat = bc_area_normal / np.linalg.norm(bc_area_normal)
        # Tangent vector is normal vector, rotated 90 degrees
        t_hat = np.array([-n_hat[1], n_hat[0]])
        # Create rotation matrix
        rotation = np.array([n_hat, t_hat])
        # Rotate velocity into normal - tangential frame
        velocity = np.array([V[1], V[2]])
        velocity_nt = rotation @ velocity
        wall_velocity_nt = rotation @ wall_velocity
        # The normal velocity of the fluid is set so that the mean of the normal
        # velocity of the fluid vs. the ghost will equal the wall velocity.
        # This is represented by: 1/2 (U_fluid + U_ghost) = U_wall. Solving for
        # U_ghost gives:
        velocity_nt[0] = 2 * wall_velocity_nt[0] - velocity_nt[0]
        # Rotate back to original frame
        velocity_new = rotation.T @ velocity_nt
        #if not np.all(np.isclose(wall_velocity, np.zeros(2))): breakpoint()
        V_ghost = np.array([r, *velocity_new, p])
        return V_ghost


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
        # Compute initial phi
        phi = self.compute_exact_phi(self.xy, 0)
        return U, phi

    def compute_ghost_state(self, U, U_ghost, bc_type, bc_area_normal):
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
            # TODO: This requires exact phi!
            # Compute interface ghost state
            if bc == 0:
                r, u, v, p = self.compute_ghost_wall(V, bc_area_normal[i],
                        wall_velocity=np.array([self.u, 0]))
            # Compute wall ghost state
            elif bc == 1:
                r, u, v, p = self.compute_ghost_wall(V, bc_area_normal[i])
            # Compute inflow ghost state
            elif bc == 2:
                # Set state to the left state
                r = r4
                u = u4
                v = v4
                p = p4
            # Compute outflow ghost state
            elif bc == 3:
                # Set state to the right state
                r = r1
                u = u1
                v = v1
                p = p1
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
            if bc in [0, 1]:
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
    # Advection speed
    u = 50
    # Left state (rho, u, v, p, phi)
    state_4 = np.array([
            1, u, 0, 1e5, -1
    ])

    # Right state (rho, u, v, p, phi)
    state_1 = np.array([
            .125, u, 0, 1e5, 1
    ])

    def compute_exact_phi(self, coords, t):
        '''
        Compute the exact phi, which is a plane advecting at constant speed.
        '''
        x = coords[:, 0]
        phi = x - self.u * t
        return phi

    def compute_exact_phi_gradient(self, coords, t):
        '''
        Compute the gradient of the exact phi.
        '''
        x = coords[:, 0]
        gphi = np.array([1, 0])
        return gphi

    def plot_exact_interface(self, axis, mesh, t):
        axis.vlines(self.u * t, mesh.yL, mesh.yR, color='r')

class AdvectedBubble(Problem):
    '''
    Class for a bubble advecting at constant velocity.
    '''
    # Advection speed
    u = 50
    # bubble state (rho, u, v, p, phi)
    bubble = np.array([
            .125, u, 0, 1e5, -1
    ])
    # Ambient state (rho, u, v, p, phi)
    ambient = np.array([
            1, u, 0, 1e5, 1
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
        # Compute initial phi
        phi = self.compute_exact_phi(self.xy, 0)
        return U, phi

    def compute_exact_phi(self, coords, t):
        '''
        Compute the exact phi, which is a paraboloid advecting at constant
        speed.
        '''
        x = coords[:, 0]
        y = coords[:, 1]
        phi = (x - self.u * t)**2 + y**2 - self.radius**2
        return phi

    def compute_exact_phi_gradient(self, coords, t):
        '''
        Compute the gradient of the exact phi.
        '''
        x = coords[:, 0]
        y = coords[:, 1]
        gphi = np.array([
            2 * (x - self.u * t),
            2 * y])
        return gphi

    def plot_exact_interface(self, axis, mesh, t):
        axis.add_patch(matplotlib.patches.Circle((self.u * t, 0), self.radius,
            color='r', fill=False))

    def compute_ghost_state(self, U, U_ghost, bc_type, bc_area_normal):
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
            # TODO: This requires exact phi!
            # Compute interface ghost state
            if bc == 0:
                r, u, v, p = self.compute_ghost_wall(V, bc_area_normal[i],
                        wall_velocity=np.array([self.u, 0]))
            # Compute wall ghost state
            elif bc == 1:
                r, u, v, p = self.compute_ghost_wall(V, bc_area_normal[i])
            # Compute inflow/outflow ghost state
            elif bc in [2, 3]:
                # Set state to ambient
                r, u, v, p, _ = self.ambient
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
            if bc in [0, 1, 2, 3]:
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
