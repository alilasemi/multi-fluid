import numpy as np

from exact_solution import exact_solution
import matplotlib.patches


class Problem:
    '''
    Parent class for all problem definitions.
    '''

    bc_data = np.empty((4, 5))
    exact = False
    def __init__(self, xy, t_list, bc_type):
        self.xy = xy
        self.t_list = t_list
        self.n = xy.shape[0]
        self.bc_type = bc_type
        self.set_bc_data()

    def set_bc(self, bc, bc_name, data=None):
        # All BCs store gamma in the last entry
        self.bc_data[bc, -1] = self.g
        # Nothing special needed for walls
        if bc_name == 'wall':
            pass
        # For interfaces, need to pass some data for computing the interface
        # velocity later on. For full state BCs, store the state.
        elif bc_name == 'interface' or 'full state':
            # Check to make sure data was supplied
            if data is None:
                print(f'No data given for {bc_name} BC! data = {data}')
            self.bc_data[bc, :data.size] = data
        # Otherwise, this BC is not recognized
        else:
            print(f'BC name not recognized! was given bc_name = {bc_name}')
        # Change the BC IDs
        self.change_bc_ID(bc, bc_name)

    #TODO: This is a bit jank. maybe change how this works at some point.
    # Basically, originally the mesh marks the top and bottom as bc = 1, the
    # left as bc = 2, and the right as bc = 3. Later, when Problem sets the BC
    # type on each boundary, this number is changed: 0 for interface, 1 for
    # wall, and 2 for full state.
    def change_bc_ID(self, bc, bc_name):
        if bc_name == 'interface':
            bc_ID = 0
        elif bc_name == 'wall':
            bc_ID = 1
        elif bc_name == 'full state':
            bc_ID = 2
        self.bc_type[np.nonzero(self.bc_type[:, 1] == bc), 1] = bc_ID


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

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        self.set_bc(0, 'interface')
        # Set BC 1 to be the walls
        self.set_bc(1, 'wall')
        # Set BC 2 to be the left state
        self.set_bc(2, 'full state', self.state_4[:-1])
        # Set BC 3 to be the right state
        self.set_bc(3, 'full state', self.state_1[:-1])

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
    v = 0
    # bubble state (rho, u, v, p, phi)
    bubble = np.array([
            .125, u, v, 1e5, -1
    ])
    # Ambient state (rho, u, v, p, phi)
    ambient = np.array([
            1, u, v, 1e5, 1
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

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        interface_velocity = np.array([self.u, self.v])
        self.set_bc(0, 'interface', interface_velocity)
        # Set BC 1 to be the walls
        self.set_bc(1, 'wall')
        # Set BC 2 and 3 to be the ambient state
        self.set_bc(2, 'full state', self.ambient[:-1])
        self.set_bc(3, 'full state', self.ambient[:-1])

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


class CollapsingCylinder(Problem):
    '''
    Class for a collapsing cylinder.
    '''
    # Ambient state (rho, u, v, p)
    ambient = np.array([
            1, 0, 0, 1e5
    ])

    # Radius of cylinder
    R = .25

    # Ratio of specific heats
    g = 1.4

    def get_initial_conditions(self):
        # Unpack
        r, u, v, p = self.ambient
        g = self.g

        # Get initial conditions as conservatives
        W = primitive_to_conservative(r, u, v, p, g)

        # Set whole domain to have this solution
        U = W * np.ones((self.n, 4))
        # Compute initial phi
        phi = self.compute_exact_phi(self.xy, 0)
        return U, phi

    def compute_exact_phi(self, coords, t):
        '''
        Compute the exact phi, which is a deformed paraboloid following the
        interface.
        '''
        x = coords[:, 0]
        y = coords[:, 1]
        theta = np.arctan2(y, x)
        # Compute r
        a = 100 * np.pi
        r = (1/3 * (2 + np.cos(4 * theta))) * np.sin(a*t)**2 + np.cos(a*t)**2
        r *= self.R
        # Compute paraboloid
        phi = (x/r)**2 + (y/r)**2 - 1
        return phi

    def compute_exact_phi_gradient(self, coords, t):
        '''
        Compute the gradient of the exact phi.
        '''
        x = coords[:, 0]
        y = coords[:, 1]
        theta = np.arctan2(y, x)
        # Compute r
        a = 100 * np.pi
        r = (1/3 * (2 + np.cos(4 * theta))) * np.sin(a*t)**2 + np.cos(a*t)**2
        r *= R
        # Compute paraboloid's gradient
        gphi = np.array([
            2 * (x/r) / r,
            2 * (y/r) / r])
        return gphi

    def plot_exact_interface(self, axis, mesh, t):
        # Range of theta
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        # Compute r
        a = 100 * np.pi
        r = (1/3 * (2 + np.cos(4 * theta))) * np.sin(a*t)**2 + np.cos(a*t)**2
        r *= self.R
        # Convert to x and y
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # "loop" the array back onto itself for plotting
        x_loop = np.empty(n_points + 1)
        y_loop = np.empty(n_points + 1)
        x_loop[:-1] = x
        y_loop[:-1] = y
        x_loop[-1] = x[0]
        y_loop[-1] = y[0]
        # Plot
        axis.plot(x_loop, y_loop, 'r')

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        #TODO
        interface_velocity_data = np.array([self.R])
        self.set_bc(0, 'interface', interface_velocity_data)
        # Set BC 1, 2, and 3 to be walls
        self.set_bc(1, 'wall')
        self.set_bc(2, 'wall')
        self.set_bc(3, 'wall')

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        print('compute_ghost_phi not implemented for CollapsingCylinder!')


# TODO: These are marked for removal. Point to the C++ functions instead.
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
