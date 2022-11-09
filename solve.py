import numpy as np
from rich.progress import TextColumn, BarColumn, TimeRemainingColumn, Progress

from mesh import Mesh
from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        CollapsingCylinder, Star, Cavitation, conservative_to_primitive,
        primitive_to_conservative)
from residual import get_residual, get_residual_phi
from build.src.libpybind_bindings import compute_gradient, compute_gradient_phi
from lagrange import LagrangeSegmentP2


# Solver inputs
Problem = Cavitation
nx = 101
ny = 101
#n_t = 1
cfl = .5
#t_final = 1e-6
t_final = 2e-2
max_n_t = 99999999999
level_set_reinitialization_rate = 5
adaptive = False
rho_levels = np.linspace(.15, 1.05, 19)
linear_reconstruction = True

# Physical parameters
g = [4.4, 1.4]
psg = [6e5, 0]#[6e8, 0]
#g = [1.4, 1.4]
#psg = [0, 0]

file_name = 'data.npz'
ghost_fluid_interfaces = True
update_ghost_fluid_cells = True
linear_ghost_extrapolation = False
levelset = True

# List of times at which the solution should be written to file
t_list = np.linspace(0, t_final, 11).tolist()

def main(show_progress_bar=True):
    # Create mesh
    mesh = Mesh(nx, ny, Problem.xL, Problem.xR, Problem.yL, Problem.yR,
            adaptive)
    vol_points_copy = mesh.vol_points.copy()
    edge_points_copy = mesh.edge_points.copy()

    # Initial solution
    problem = Problem(mesh.xy, t_list, mesh.bc_type, g, psg)
    U, phi = problem.get_initial_conditions()
    U_ghost = np.empty((mesh.bc_type.shape[0], 4))

    # Store data
    data = SimulationData(mesh.nx, mesh.ny, mesh.n_faces, U, phi, U_ghost,
            t_list, g, psg, file_name)
    del U, phi, U_ghost
    # TODO hack to save initial phi
    initial_phi = data.phi.copy()

    # Set fluid identity based on phi
    data.fluid_ID = (data.phi < 0).astype(int)
    # Save initial condition, if desired
    written_times = -np.ones_like(t_list)
    if 0 in t_list:
        data.save_current_state(mesh)
        written_times[0] = 0
        data.t_list[0] = 0
    # Create initial ghost fluid interfaces
    if ghost_fluid_interfaces:
        mesh.create_interfaces(data, problem.fluid_solid)

    print('---- Solving ----')
    # Set up progress bar
    with Progress(
            TextColumn('Running iterations...', justify='right'),
            BarColumn(finished_style='purple'),
            '[progress.percentage]{task.percentage:>3.0f}%',
            TimeRemainingColumn(),
            '[yellow]{task.fields[iteration]}',
            disable = not show_progress_bar,
            auto_refresh=False) as progress:
        task = progress.add_task('Running iterations...',
                total=t_final, iteration=0)

        # Loop over time
        x_shock = []
        for i in range(max_n_t):
            data.new_iteration()

            # Set fluid identity based on phi
            data.fluid_ID = (data.phi < 0).astype(int)

            # -- Update Mesh -- #
            if adaptive:
                # Store copy of original in case this iteration crashes
                vol_points_copy = mesh.vol_points.copy()
                edge_points_copy = mesh.edge_points.copy()
                # Revert back to original face points
                mesh.vol_points = mesh.original_vol_points.copy()
                mesh.edge_points = mesh.original_edge_points.copy()
                # Update gradient of phi
                compute_gradient_phi(data.phi, mesh.xy, mesh.neighbors,
                        data.grad_phi)
                # Update the mesh
                mesh.update(data, problem)
            # Update the stencil to not include points across the interface
            mesh.update_stencil(data.phi)

            # -- Choose Timestep -- #
            # This should happen after the mesh update, so that the updated cell
            # areas are used in the CFL condition.
            # If told to compute timestep using CFL
            if 'cfl' in globals():
                # Get fluid data for each cell
                g_i = np.array(g)[data.fluid_ID]
                psg_i = np.array(psg)[data.fluid_ID]
                r  = data.U[:, 0]
                ru = data.U[:, 1]
                rv = data.U[:, 2]
                re = data.U[:, 3]
                # Compute pressure
                p = (g_i - 1) * (re - .5 * (ru**2 + rv**2) / r) - g_i * psg_i;
                # Speed of sound
                a = np.sqrt(g_i * (p + psg_i) / r)
                # Norm of velocity
                norm_u = np.sqrt(ru**2 + rv**2) / r
                # Compute timestep from CFL
                max_wave_speed = norm_u + a
                lengths = np.sqrt(mesh.area)
                timesteps = cfl * lengths / max_wave_speed
                dt = np.nanmin(timesteps)
                # logic to ensure final time step yields final time
                if data.t + dt > t_final:
                    dt = t_final - data.t
            # If told to compute timestep using number of timesteps
            elif 'n_t' in globals():
                dt = t_final / n_t
            else:
                raise ValueError('Neither a CFL nor a number of timesteps was '
                        'provided!')
            data.dt = dt

            # Compute gradients
            compute_gradient(data.U, mesh.xy, mesh.stencil,
                    data.gradV.reshape(-1), data.g, data.psg, data.fluid_ID)
            compute_gradient_phi(data.phi, mesh.xy, mesh.neighbors,
                    data.grad_phi)

            # If first order is requested, set the gradients to zero
            if not linear_reconstruction:
                data.gradV = np.zeros_like(data.gradV)
                data.grad_phi = np.zeros_like(data.grad_phi)

            # Create ghost fluid interfaces
            if ghost_fluid_interfaces:
                mesh.create_interfaces(data, problem.fluid_solid)
            # Allocate according to new interfaces
            data.U_L_p1 = np.empty((mesh.interior_face_IDs.size, 4))
            data.U_R_p1 = np.empty_like(data.U_L_p1)
            data.U_L_p2 = np.empty((mesh.interface_IDs.size,
                LagrangeSegmentP2.nq, 4))
            data.U_R_p2 = np.empty_like(data.U_L_p2)

            #print('bubdensity', data.U[4, 0])
            #print(data.phi[0])
            #print(data.phi[8])
            #if True:#data.i in [22, 23, 24]:
            #    print(f'Printing i = {data.i}')
            #    print(data.U[0])
            #    flipped = data.U[8].copy()
            #    flipped[1:3] *= -1
            #    print(flipped)
            #    print(data.U[4])
            #    #if data.i == 24: reakpoint()

            # -- Copy solution, then update -- #
            U_old = data.U.copy()
            try:
                data.U = update(dt, data, mesh, problem)
            # Handle errors from the Riemann solver
            except RuntimeError as err:
                data.U = U_old
                data.save_current_state(mesh)
                data.t_list = [time for time in data.t_list if time <= data.t]
                data.t_list.append(data.t)
                # Save solution
                data.write_to_file()
                raise err
            data.t += dt

            phi_old = data.phi.copy()
            # Reinitialize level set
            # TODO: This stuff is mostly just a hack for now
            if level_set_reinitialization_rate != 0 and i % level_set_reinitialization_rate == 0:
                if adaptive:
                    reinitialize_level_set_higher_order(data, mesh)
                else:
                    reinitialize_level_set(data, mesh)

            # -- Update Phi -- #
            if problem.fluid_solid:
                data.phi = problem.compute_exact_phi(mesh.xy, data.t)
            else:
                if levelset:
                    data.phi = update_phi(dt, data, mesh, problem)
#                    # TODO: This is a hack to try a prescribed phi
#                    theta = np.loadtxt('../../mnt/ind/projects/cavitation/simulations/case1/postpro/radius_theta.txt')
#                    k = 11
#                    X_hat = np.empty_like(theta)
#                    for i in range(k + 1):
#                        X_hat[i] = (data.t / 1e-2)**i
#                    radius = X_hat.T @ theta
#                    x = mesh.xy[:, 0]
#                    y = mesh.xy[:, 1]
#                    data.phi = np.sqrt(x**2 + y**2) - radius

            # -- Update Ghost Fluid Cells -- #
            if ghost_fluid_interfaces and update_ghost_fluid_cells:
                # Find which cells just switched from being in one fluid to being in
                # another
                ghost_IDs = np.argwhere(phi_old * data.phi < 0)[:, 0]
                # Loop over each ghost
                for ghost_ID in ghost_IDs:
                    # Get new sign of phi for this ghost
                    sign = np.sign(data.phi[ghost_ID])
                    # Update fluid ID
                    data.fluid_ID[ghost_ID] = int(sign < 0)
                    # Get neighbors, not including this cell
                    neighbors = np.array(mesh.neighbors[ghost_ID])[1:]
                    # Get neighbors that are in the same fluid
                    fluid_neighbors = neighbors[
                            np.argwhere(data.phi[neighbors] * sign > 0
                            )[:, 0]].tolist()
                    # Make sure other ghosts are not included
                    fluid_neighbors = np.array([n for n in fluid_neighbors if not n in
                            ghost_IDs])
                    # Check for lonely ghosts
                    n_points = fluid_neighbors.size
                    if n_points == 0:
                        print(f'Oh no, a lone ghost fluid cell! ID = {ghost_ID}')
                        continue

                    # Perform linear extrapolation of the ghost fluid state
                    if linear_ghost_extrapolation:
                        #TODO
                        print("Linear ghost extrapolation with primitive variables not implemented!")
                        print("The code in this loop is wrong!")
                        # Loop over state variables
                        for k in range(4):
                            # Construct A matrix:
                            # [x_i, y_i, 1]
                            # [1,   0,   0]
                            # [0,   1,   0]
                            A = np.zeros((3 * n_points, 3))
                            gradients = data.gradU[fluid_neighbors, k]
                            A[:n_points, :-1] = mesh.xy[fluid_neighbors]
                            A[:n_points, -1] = 1
                            A[n_points:2*n_points, 0] = 1
                            A[2*n_points:, 1] = 1
                            b = np.empty(3 * n_points)
                            b[:n_points] = data.U[fluid_neighbors, k]
                            b[n_points:2*n_points] = gradients[:, 0]
                            b[2*n_points:] = gradients[:, 1]
                            # We desired [x_i, y_i, 1] @ [c0, c1, c2] = U[i], therefore Ax=b.
                            # However, there are more equations than unknowns (for most points)
                            # so instead, solve the normal equations: A.T @ A x = A.T @ b
                            c = np.linalg.solve(A.T @ A, A.T @ b)
                            # Evaluate extrapolant, U = c0 x + c1 y + c2.
                            data.U[ghost_ID, k] = np.dot(c[:-1], mesh.xy[ghost_ID]) + c[-1]
                    # Use constant extrapolation
                    else:
                        # Neighbor conservative state vectors
                        neighbor_U = data.U[fluid_neighbors]
                        # For each neighbor
                        neighbor_V = np.empty_like(neighbor_U)
                        for j, cell_ID in enumerate(fluid_neighbors):
                            # Get fluid data
                            g_j = data.g[data.fluid_ID[cell_ID]]
                            psg_j = data.psg[data.fluid_ID[cell_ID]]
                            # Convert to primitive
                            neighbor_V[j] = conservative_to_primitive(
                                    *neighbor_U[j], g_j, psg_j)
                        # Average neighbor primitive variables
                        mean_V = np.mean(neighbor_V, axis=0)
                        # Convert back to conservative
                        g_ghost = data.g[data.fluid_ID[ghost_ID]]
                        psg_ghost = data.psg[data.fluid_ID[ghost_ID]]
                        data.U[ghost_ID] = primitive_to_conservative(
                                *mean_V, g_ghost, psg_ghost)

            #if data.t > .008:
            #    V = np.empty_like(data.U)
            #    for j in range(mesh.n):
            #        g_j = data.g[data.fluid_ID[j]]
            #        psg_j = data.psg[data.fluid_ID[j]]
            #        V[j] = conservative_to_primitive(*data.U[j], g_j, psg_j)
            #    max_p = np.max(V[:, 3])
            #    if max_p > 140000:
            #        reakpoint()

            # If the solution NaN's, then store the current solution for plotting
            # and stop. It is important to do this after the ghost fluid update,
            # since in in a ghost fluid method, the cells that have seemingly
            # NaN'd their residuals might actually be replaced by ghost fluid.
            # This is one of the advantages of ghost fluid methods, giving
            # robustness (for example, in large density jump regions, this can
            # really save you).
            nan_IDs = np.unique(np.argwhere(np.isnan(data.U))[:, 0])
            if nan_IDs.size != 0:
                data.U = U_old
                data.phi = phi_old
                data.save_current_state(mesh)
                data.t_list = [time for time in data.t_list if time <= data.t]
                data.t_list.append(data.t)
                # Save solution
                data.write_to_file()
                # Raise error
                message = f'Oh no! NaN detected in the solution! Iteration = {data.i}\n'
                message += f'The following {nan_IDs.size} cells are all NaN\'d out:\n'
                message += f'{nan_IDs}'
                raise FloatingPointError(message)

            # Figure out which index to write to
            last_write_indices = np.argwhere(written_times > -1)
            if last_write_indices.size == 0:
                write_index = 0
            else:
                write_index = np.max(last_write_indices) + 1
            # Store data once the time has come
            if np.isclose(data.t, t_final, atol=0) or data.t >= data.t_list[write_index]:
                # TODO: I do a mesh update before writing, to sync up phi with the
                # mesh. Is this the best solution?
                if adaptive:
                    # Revert back to original face points
                    mesh.vol_points = mesh.original_vol_points.copy()
                    mesh.edge_points = mesh.original_edge_points.copy()
                    # Update the mesh
                    mesh.update(data, problem)
                # Update the chosen save time with the real one
                written_times[write_index] = data.t
                data.t_list[write_index] = data.t
                data.save_current_state(mesh)
            # Find shock
            if Problem == RiemannProblem:
                for j in range(nx):
                    # Jump in x-velocity
                    # TODO Cleaner indices
                    line = ny // 2
                    u4 = problem.state_4[1]
                    delta_u = data.U[line*nx + nx - 1 - j, 1] / data.U[line*nx + nx - 1 - j, 0] - u4
                    if delta_u > .01 * u4:
                        x_shock.append(mesh.xy[nx - 1 - j, 0])
                        break
            # Update progress bar
            progress.update(task, advance=dt, iteration=i, refresh=True)
            # If it's hit the final time, then stop iterating
            if np.isclose(data.t, t_final, atol=0): break

    # Fit a line to the shock location
    if Problem == RiemannProblem:
        try:
            fit_shock = np.polyfit(np.linspace(dt, t_final, n_t), x_shock, 1)
            shock_speed = fit_shock[0]
            print(f'The shock speed is {shock_speed} m/s.')
        except np.linalg.LinAlgError:
            print('-- Shock speed calculation failed! --')

    # Save solution
    data.write_to_file()

def update(dt, data, mesh, problem):
    U_new = data.U.copy()
    # Compute residual
    R = get_residual(data, mesh, problem)
    # Forward euler
    U_new += dt * R
    return U_new

def update_phi(dt, data, mesh, problem):
    phi_new = data.phi.copy()
    # Compute residual
    R = get_residual_phi(data, mesh, problem)
    # Forward euler
    phi_new += dt * R
    return phi_new

def reinitialize_level_set(data, mesh):
    # Check to see if there are any interfaces
    if mesh.interface_IDs.size == 0:
        print('Level set reinitialization requested, but there are no '
                'interfaces!')
        return
    # Array of points on the interface
    interface_points = np.empty((mesh.interface_IDs.size, 2))
    # For each interface
    for i, interface_ID in enumerate(mesh.interface_IDs):
        # Get cells on either side
        L, R = mesh.edge[interface_ID]
        phiL, phiR = data.phi[[L, R]]
        xL, xR = mesh.xy[[L, R], 0]
        yL, yR = mesh.xy[[L, R], 1]
        # Use a 1D Lagrange fit in both x and y to find the location of phi = 0
        if np.isclose(xL, xR):
            x = np.mean([xL, xR])
        else:
            x = xR * phiL / (xL - xR) + xL * phiR / (xR - xL)
            x /= phiL / (xL - xR) + phiR / (xR - xL)
        if np.isclose(yL, yR):
            y = np.mean([yL, yR])
        else:
            y = yR * phiL / (yL - yR) + yL * phiR / (yR - yL)
            y /= phiL / (yL - yR) + phiR / (yR - yL)
        # Store
        interface_points[i] = [x, y]

    # Set phi to be the signed distance to the nearest point on the interface
    data.phi = np.sign(data.phi)
    for i in range(mesh.n):
        distances = np.linalg.norm(interface_points - mesh.xy[i], axis=1)
        data.phi[i] *= np.min(distances)

    # Update gradient
    compute_gradient_phi(data.phi, mesh.xy, mesh.neighbors, data.grad_phi)

# TODO: This will fail if you have an interface on a boundary
def reinitialize_level_set_higher_order(data, mesh):
    # Check to see if there are any interfaces
    if mesh.interface_IDs.size == 0:
        print('Level set reinitialization requested, but there are no '
                'interfaces!')
        return
    # Array of points on the interface
    n_points_per_face = 10
    ref_coords = np.linspace(0, 1, n_points_per_face)
    interface_points = np.empty((n_points_per_face * mesh.interface_IDs.size, 2))
    # For each interface
    for i, interface_ID in enumerate(mesh.interface_IDs):
        # Get face points
        face_point_coords = mesh.get_face_point_coords(interface_ID)
        x = face_point_coords[:, 0]
        y = face_point_coords[:, 1]
        # Create a Lagrange segment
        x_seg = LagrangeSegmentP2(x)
        y_seg = LagrangeSegmentP2(y)
        # Evaluate location of points in physical space
        physical_x = np.matmul(
                x_seg.get_basis_values(ref_coords), x_seg.coords)
        physical_y = np.matmul(
                y_seg.get_basis_values(ref_coords), y_seg.coords)
        # Store
        interface_points[i*n_points_per_face : (i + 1)*n_points_per_face, 0] = (
                physical_x)
        interface_points[i*n_points_per_face : (i + 1)*n_points_per_face, 1] = (
                physical_y)

    eps = 1e-15
    # Set phi to be the signed distance to the nearest point on the interface
    data.phi = np.sign(data.phi)
    for i in range(mesh.n):
        distances = np.linalg.norm(interface_points - mesh.xy[i], axis=1)
        min_distance = np.min(distances)
        # Prevent this from reaching exactly zero
        min_distance = np.max([min_distance, eps])
        # Update phi
        data.phi[i] *= min_distance

    # Update gradient
    compute_gradient_phi(data.phi, mesh.xy, mesh.neighbors, data.grad_phi)

class SimulationData:
    '''
    Class for initializing and storing objects and data.

    Members:
    --------
    i - int, current iteration counter
    t - float, simulation time
    nx - int, number of grid points in x
    ny - int, number of grid points in y
    n_faces - int, number of faces
    U_list - list, solution at each stored timestep
    phi_list - list, level set at each stored timestep
    t_list - list, time at each stored timestep
    edge_points_list - list, coordinates of edge points at each stored timestep
    vol_points_list - list, coordinates of volume points at each stored timestep
    g - float, ratio of specific heats
    U - np.array, solution
    phi - np.array, level set
    U_ghost - np.array, ghost state
    gradV - np.array, gradient of solution
    grad_phi - np.array, gradient of level set
    U_L - np.array, solution evaluated at the left side of each face
    U_R - np.array, solution evaluated at the right side of each face
    fluid_ID - np.array, the fluid identity of each cell
    fluid_ID_list - list, fluid identity at each stored timestep
    file_name - string, name of file for reading/writing data
    '''
    # Iteration counter
    i = 0
    # Simulation time
    t = 0
    # Current timestep
    dt = 0

    def __init__(self, nx, ny, n_faces, U, phi, U_ghost, t_list, g, psg, file_name):
        # Save mesh sizing
        self.nx = nx
        self.ny = ny
        self.n_faces = n_faces
        # Set the ratio of specific heats
        self.g = g
        self.psg = psg
        # Temporary buffers
        self.U = U
        self.phi = phi
        self.U_ghost = U_ghost
        self.gradV = np.empty((nx*ny, 4, 2))
        self.grad_phi = np.empty((nx*ny, 2))
        self.U_L_p1 = None
        self.U_R_p1 = None
        self.U_L_p2 = None
        self.U_R_p2 = None
        self.U_L = np.empty((n_faces, 4))
        self.U_R = np.empty((n_faces, 4))
        self.fluid_ID = np.empty(nx*ny, dtype=int)
        self.fluid_ID_list = []
        # Lists of data for each stored timestep
        self.U_list = []
        self.phi_list = []
        self.t_list = t_list.copy()
        self.edge_points_list = []
        self.vol_points_list = []
        # Name of file to write to
        self.file_name = file_name

    def new_iteration(self):
        # Update iteration counter
        self.i += 1

    def save_current_state(self, mesh):
        '''
        Save the state of the simulation.
        '''
        self.U_list.append(self.U.copy())
        self.phi_list.append(self.phi.copy())
        self.fluid_ID_list.append(self.fluid_ID.copy())
        self.edge_points_list.append(mesh.edge_points.copy())
        self.vol_points_list.append(mesh.vol_points.copy())

    def write_to_file(self):
        with open(file_name, 'wb') as f:
            np.savez(f, nx=self.nx, ny=self.ny, n_faces=self.n_faces, g=self.g,
                    psg=self.psg, fluid_ID_list=self.fluid_ID_list, U_list=self.U_list,
                    phi_list=self.phi_list, t_list=self.t_list,
                    edge_points_list=self.edge_points_list,
                    vol_points_list=self.vol_points_list, allow_pickle=True)

    @classmethod
    def read_from_file(cls, file_name):
        with open(file_name, 'rb') as f:
            data = np.load(f)
            sim_data = cls(data['nx'], data['ny'], data['n_faces'], None, None,
                    None, data['t_list'], data['g'], data['psg'], file_name)
            sim_data.U_list = data['U_list']
            sim_data.phi_list = data['phi_list']
            sim_data.edge_points_list = data['edge_points_list']
            sim_data.vol_points_list = data['vol_points_list']
            sim_data.fluid_ID_list = data['fluid_ID_list']
        return sim_data

if __name__ == '__main__':
    main()
