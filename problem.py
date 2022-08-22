import numpy as np

from exact_solution import exact_solution
import matplotlib.patches
import scipy
import scipy.special
import scipy.interpolate


class Problem:
    '''
    Parent class for all problem definitions.
    '''
    bc_data = np.empty((4, 5))
    exact = False
    fluid_solid = True
    def __init__(self, xy, t_list, bc_type):
        self.xy = xy
        self.t_list = t_list
        self.n = xy.shape[0]
        self.bc_type = bc_type
        self.set_bc_data()

    def set_bc(self, bc, bc_name, data=None):
        # All BCs store gamma in the last entry
        self.bc_data[bc, -1] = self.g
        # Nothing special needed for walls or advected interfaces
        if bc_name == 'wall' or bc_name == 'advected interface':
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
        elif bc_name == 'advected interface':
            bc_ID = 3
        self.bc_type[np.nonzero(self.bc_type[:, 1] == bc), 1] = bc_ID


class RiemannProblem(Problem):
    '''
    Class for defining ICs and BCs for a Riemann problem.
    '''
    # Domain
    xL = -10
    xR = 10
    yL = -1
    yR = 1

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
    fluid_solid = False
    # Domain
    xL = -1
    xR = 1
    yL = -1
    yR = 1
    # Advection speed
    u = 50
    v = 0
    # bubble state (rho, u, v, p, phi)
    bubble = np.array([
            1, u, v, 1e5, -1
    ])
    # Ambient state (rho, u, v, p, phi)
    ambient = np.array([
            1000, u, v, 1e5, 1
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

    def plot_exact_interface(self, axis, mesh, t, lw_scale):
        axis.add_patch(matplotlib.patches.Circle((self.u * t, 0), self.radius,
            color='r', fill=False))

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        self.set_bc(0, 'advected interface')
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
            if bc in [0, 1, 2]:
                # Just use the initial value as the boundary value
                phi_ghost[i] = (self.xy[cell_ID, 0]**2 + self.xy[cell_ID, 1]**2
                        - self.radius**2)
            # Compute advected interface ghost state
            elif bc > 2:
                # Just use the ghost fluid value of phi
                ghost_cell_ID = bc - 3
                phi_ghost[i] = phi[ghost_cell_ID]
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')


class CollapsingCylinder(Problem):
    '''
    Class for a collapsing cylinder.
    '''
    # Domain
    xL = -1
    xR = 1
    yL = -1
    yR = 1

    # Ambient state (rho, u, v, p)
    ambient = np.array([
            1, 0, 0, 1e5
    ])

    # Radius of cylinder
    R = .25

    # Total time to collapse and return back to a cylinder
    period = .0005

    # Ratio of specific heats
    g = 1.4

    # Levels to use for contour plots
    levels = [
            np.linspace(.8, 2, 17),
            np.linspace(-400, 400, 17),
            np.linspace(-400, 400, 17),
            np.linspace(1e4, 2e5, 20)]

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
        a = np.pi / self.period
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
        # From Wikipedia: https://en.wikipedia.org/wiki/Atan2#Derivative
        dtheta_dx = -y / (x**2 + y**2)
        dtheta_dy =  x / (x**2 + y**2)
        # Compute r
        a = np.pi / self.period
        r = (1/3 * (2 + np.cos(4 * theta))) * np.sin(a*t)**2 + np.cos(a*t)**2
        r *= self.R
        # Compute derivatives
        dr_dtheta = -self.R * (4/3 * np.sin(a*t)**2) * np.sin(4 * theta)
        dr_dx = dr_dtheta * dtheta_dx
        dr_dy = dr_dtheta * dtheta_dy
        # Compute paraboloid's gradient
        gphi = np.array([
            2 * (x/r) * (r - x*dr_dx)/(r**2),
            2 * (y/r) * (r - y*dr_dy)/(r**2)])
        return gphi

    def plot_exact_interface(self, axis, mesh, t, lw_scale):
        # Range of theta
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        # Compute r
        a = np.pi / self.period
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
        axis.plot(x_loop, y_loop, 'r', 2*lw_scale)

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        #TODO
        interface_velocity_data = np.array([self.R, self.period])
        self.set_bc(0, 'interface', interface_velocity_data)
        # Set BC 1, 2, and 3 to be walls
        self.set_bc(1, 'wall')
        self.set_bc(2, 'wall')
        self.set_bc(3, 'wall')

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        print('compute_ghost_phi not implemented for CollapsingCylinder!')

class Star(Problem):
    '''
    Class for the star problem.
    '''
    # Diameter
    D = 1
    # Domain
    xL = -1*D
    xR = 1*D
    yL = -1*D
    yR = 1*D

    # Ambient state (rho, u, v, p)
    ambient = np.array([
            1, 0, 0, 1e4
    ])

    # Frequency
    f = 100

    # Ratio of specific heats
    g = 1.4

    # Levels to use for contour plots
    #levels = [
    #        np.linspace(.8, 4, 17),
    #        np.linspace(-400, 400, 17),
    #        np.linspace(-400, 400, 17),
    #        np.linspace(1e5, 2e6, 20)]

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
        f = self.f
#        theta = np.arctan2(y, x)
#        # Convert to x and y
#        x_int = self.D/2 * np.cos(theta)
#        y_int = self.D/2 * np.sin(theta)
#        # From slides that Prof. Farhat gave me
#        A = np.abs(x_int) + np.abs(y_int) - np.sqrt(
#                x_int**2 + y_int**2)
#        B = np.cos(2 * np.pi * self.f * t) - 1
#        displacement = .5 * A * B * np.sign([x_int, y_int])
#        x_disp = x_int + displacement[0, :]
#        y_disp = y_int + displacement[1, :]
#        r = np.sqrt(x_disp**2 + y_disp**2)
#        # Compute paraboloid
#        phi = (x/r)**2 + (y/r)**2 - 1
        remainder = (t - 1/(4*f)) % (1/f)
        if remainder < 1/(2*f):
            x = np.array([0, .3, .4, .47, .495, .5])
        else:
            x = np.array([0, .05, .12, .22, .35, .5])
        # Get y at this point for a circle of radius 1/2
        y = np.sqrt(1/4 - x**2)
        # From slides that Prof. Farhat gave me
        A = np.abs(x) + np.abs(y) - np.sqrt(
                x**2 + y**2)
        B = np.cos(2 * np.pi * f * t) - 1
        displacement = .5 * A * B * np.sign([x, y])
        xq = x + displacement[0, :]
        yq = y + displacement[1, :]
        # Basis eval'd at quad points
        nb = nq = 6
        phi = np.empty((nb, nq))
        for i in range(nb):
            phi[i] = scipy.special.eval_legendre(i, xq)
        # Compute coefficients
        Uc = np.linalg.solve(phi.T, yq)
        # Compute solution at given points
        x = coords[:, 0]
        y = coords[:, 1]
        phi = np.abs(y) - np.polynomial.legendre.legval(np.abs(x), Uc)
        return phi

    def compute_exact_phi_gradient(self, coords, t):
        '''
        Compute the gradient of the exact phi.
        '''
        f = self.f
#        x = coords[:, 0]
#        y = coords[:, 1]
#        theta = np.arctan2(y, x)
#        # Convert to x and y
#        x_int = self.D/2 * np.cos(theta)
#        y_int = self.D/2 * np.sin(theta)
#        # From Wikipedia: https://en.wikipedia.org/wiki/Atan2#Derivative
#        dtheta_dx = -y / (x**2 + y**2)
#        dtheta_dy =  x / (x**2 + y**2)
#        # From slides that Prof. Farhat gave me
#        A = np.abs(x_int) + np.abs(y_int) - np.sqrt(
#                x_int**2 + y_int**2)
#        B = np.cos(2 * np.pi * self.f * t) - 1
#        # Compute grad A
#        norm = np.sqrt(x_int**2 + y_int**2)
#        dx_int_dx = self.D/2 * (-np.sin(theta))*dtheta_dx
#        dx_int_dy = self.D/2 * (-np.sin(theta))*dtheta_dy
#        dy_int_dx = self.D/2 * ( np.cos(theta))*dtheta_dx
#        dy_int_dy = self.D/2 * ( np.cos(theta))*dtheta_dy
#        dA_dx_int = np.sign(x_int) - x_int / norm;
#        dA_dy_int = np.sign(y_int) - y_int / norm;
#        dA_dx = dA_dx_int * dx_int_dx + dA_dy_int * dy_int_dx
#        dA_dy = dA_dx_int * dx_int_dy + dA_dy_int * dy_int_dy
#        # Compute r
#        displacement = .5 * A * B * np.sign([x_int, y_int])
#        dd0_dx = .5 * dA_dx * B * np.sign(x_int)
#        dd0_dy = .5 * dA_dy * B * np.sign(x_int)
#        dd1_dx = .5 * dA_dx * B * np.sign(y_int)
#        dd1_dy = .5 * dA_dy * B * np.sign(y_int)
#        x_disp = x_int + displacement[0, :]
#        y_disp = y_int + displacement[1, :]
#        # Compute derivatives
#        r = np.sqrt(x_disp**2 + y_disp**2)
#        dr_dx_disp = x_disp / r
#        dr_dy_disp = y_disp / r
#        dx_disp_dx = dx_int_dx + dd0_dx
#        dx_disp_dy = dx_int_dy + dd0_dy
#        dy_disp_dx = dy_int_dx + dd1_dx
#        dy_disp_dy = dy_int_dy + dd1_dy
#        dr_dx = dr_dx_disp * dx_disp_dx + dr_dy_disp * dy_disp_dx
#        dr_dy = dr_dx_disp * dx_disp_dy + dr_dy_disp * dy_disp_dy
#        # Compute paraboloid's gradient
#        gphi = np.array([
#            2 * (x/r) * (r - x*dr_dx)/(r**2),
#            2 * (y/r) * (r - y*dr_dy)/(r**2)])
        remainder = (t - 1/(4*f)) % (1/f)
        if remainder < 1/(2*f):
            x = np.array([0, .3, .4, .47, .495, .5])
        else:
            x = np.array([0, .05, .12, .22, .35, .5])
        # Get y at this point for a circle of radius 1/2
        y = np.sqrt(1/4 - x**2)
        # From slides that Prof. Farhat gave me
        A = np.abs(x) + np.abs(y) - np.sqrt(
                x**2 + y**2)
        B = np.cos(2 * np.pi * f * t) - 1
        displacement = .5 * A * B * np.sign([x, y])
        xq = x + displacement[0, :]
        yq = y + displacement[1, :]
        # Basis eval'd at quad points
        nb = nq = 6
        phi = np.empty((nb, nq))
        for i in range(nb):
            phi[i] = scipy.special.eval_legendre(i, xq)
        # Compute coefficients
        Uc = np.linalg.solve(phi.T, yq)
        # Compute solution at given points
        x = coords[:, 0]
        y = coords[:, 1]
        Uc_derivative = np.polynomial.legendre.legder(Uc)
        gphi = np.array([
            -np.sign(x) * np.polynomial.legendre.legval(np.abs(x), Uc_derivative),
            np.sign(y)])
        return gphi

    def plot_exact_interface(self, axis, mesh, t, lw_scale):
        # Range of theta
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        # Convert to x and y
        x = self.D/2 * np.cos(theta)
        y = self.D/2 * np.sin(theta)
        # Compute displacements: From slides that Prof. Farhat gave me
        norm = np.sqrt(x**2 + y**2)
        A = np.abs(x) + np.abs(y) - norm
        B = np.cos(2 * np.pi * self.f * t) - 1
        dx = .5 * A * B * np.sign(x)
        dy = .5 * A * B * np.sign(y)
        # Displace
        x += dx
        y += dy
        # "loop" the array back onto itself for plotting
        x_loop = np.empty(n_points + 1)
        y_loop = np.empty(n_points + 1)
        x_loop[:-1] = x
        y_loop[:-1] = y
        x_loop[-1] = x[0]
        y_loop[-1] = y[0]
        # Plot
        axis.plot(x_loop, y_loop, 'r', 2*lw_scale)

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        #TODO
        interface_velocity_data = np.array([self.f])
        self.set_bc(0, 'interface', interface_velocity_data)
        # Set BC 1, 2, and 3 to be walls
        self.set_bc(1, 'wall')
        self.set_bc(2, 'wall')
        self.set_bc(3, 'wall')

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        print('compute_ghost_phi not implemented for Star!')


class Cavitation(Problem):
    '''
    Class for the Rayleigh-Plesset problem.

    https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Plesset_equation
    '''
    fluid_solid = False
    # Domain
    xL = -2e-3
    xR = 2e-3
    yL = -2e-3
    yR = 2e-3

    # Ambient state (rho, u, v, p)
    ambient = np.array([
            1000, 0, 0, 1e5
    ])

    # Bubble state (rho, u, v, p)
    bubble = np.array([
            1, 0, 0, .02e5
    ])

    # Initial radius
    radius = .4e-3

    # Ratio of specific heats
    g = 1.4

    # Levels to use for contour plots
    levels = [
            np.linspace(0, ambient[0], 11),
            None,#np.linspace(-400, 400, 17),
            None,#np.linspace(-400, 400, 17),
            np.linspace(0, 1.5*ambient[3], 16)]

    def get_initial_conditions(self):
        g = self.g

        # Get initial conditions as conservatives
        r, u, v, p = self.ambient
        W_ambient = primitive_to_conservative(r, u, v, p, g)
        r, u, v, p = self.bubble
        W_bubble = primitive_to_conservative(r, u, v, p, g)

        # Set bubble in the center of the domain
        U = W_ambient * np.ones((self.n, 4))
        indices = np.nonzero(self.xy[:, 0]**2 + self.xy[:, 1]**2 <
                self.radius**2)
        U[indices] = W_bubble
        # Compute initial phi
        phi = self.compute_exact_phi(self.xy, 0)
        return U, phi

    def compute_exact_phi(self, coords, t):
        '''
        Compute the exact phi, which is an absolute value function following the
        interface.
        '''
        if np.isclose(t, 0):
            x = coords[:, 0]
            y = coords[:, 1]
            phi = np.sqrt(x**2 + y**2) - self.radius
            return phi
        else:
            raise NotImplementedError('compute_exact_phi not implemented for Cavitation!')

    def compute_exact_phi_gradient(self, coords, t):
        '''
        Compute the gradient of the exact phi.
        '''

    def plot_exact_interface(self, axis, mesh, t, lw_scale):
        # Range of theta
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        # Convert to x and y
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        # "loop" the array back onto itself for plotting
        x_loop = np.empty(n_points + 1)
        y_loop = np.empty(n_points + 1)
        x_loop[:-1] = x
        y_loop[:-1] = y
        x_loop[-1] = x[0]
        y_loop[-1] = y[0]
        # Plot
        axis.plot(x_loop, y_loop, 'r', 2*lw_scale)

    def set_bc_data(self):
        # Set BC 0 to be the interfaces
        self.set_bc(0, 'advected interface')
        # Set BC 1, 2, and 3 to be ambient state
        self.set_bc(1, 'full state', self.ambient)
        self.set_bc(2, 'full state', self.ambient)
        self.set_bc(3, 'full state', self.ambient)

    def compute_ghost_phi(self, phi, phi_ghost, bc_type):
        # Loop over each boundary cell
        for i in range(phi_ghost.shape[0]):
            cell_ID, bc = bc_type[i]
            # Compute ambient ghost state
            if bc == 2:
                # Just use the neighboring value
                phi_ghost[i] = phi[cell_ID]
            # Compute advected interface ghost state
            elif bc > 2:
                # Just use the ghost fluid value of phi
                ghost_cell_ID = bc - 3
                phi_ghost[i] = phi[ghost_cell_ID]
            else:
                print(f'ERROR: Invalid BC type given! bc = {bc}')


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
