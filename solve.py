import numpy as np
from rich.progress import TextColumn, BarColumn, TimeRemainingColumn, Progress

from mesh import Mesh
from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        CollapsingCylinder, Star, Cavitation, conservative_to_primitive,
        primitive_to_conservative)
from residual import get_residual, get_residual_phi
from build.src.libpybind_bindings import compute_gradient


# Solver inputs
Problem = AdvectedBubble
nx = 21
ny = 21
#n_t = 100
cfl = .1
t_final = .01#1e-4#3.7e-5
max_n_t = 99999999999
adaptive = False
rho_levels = np.linspace(.15, 1.05, 19)

file_name = 'data.npz'
ghost_fluid_interfaces = True
update_ghost_fluid_cells = True
linear_ghost_extrapolation = False
levelset = True

#t_list = [dt, .025, .05, .075, .1]
#t_list = [dt, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
#t_list = [dt, .0025, .005, .0075, .01]
#t_list = [dt, .0025, .005, .0075, .01, .0125, .015, .0175, .02]
t_list = np.linspace(0, t_final, 11).tolist()
#t_list = [dt, .00125, .0025, .00325, .005]
#t_list = [dt, .00025, .0005, .00075, .001]
#t_list = [dt, .000125, .00025, .000375, .0005]
#t_list = [dt, .00025]
#t_list = [dt, .000025, .00005, .000075, .0001, .000125]
#t_list = [dt, .000125, .00025, .000375, .0005,
#        .000625, .00075, .000825, .001]
#t_list = [.01]
#t_list = [dt, .004, .008]
#t_list = [dt, 4, 8]
#t_list = [dt, 8*dt, 16*dt, 24*dt, 32*dt, 40*dt]
#t_list = [dt, 4*dt, 8*dt, 12*dt, 16*dt, 20*dt]
#t_list = [dt, 2*dt, 3*dt, 4*dt, 5*dt, 6*dt, 7*dt, 8*dt, 9*dt, 10*dt, 11*dt,
#        12*dt, 13*dt, 14*dt, 15*dt]
#t_list = [dt, 2*dt, 3*dt, 4*dt]
#t_list = [0, dt,]
#t_list = [dt,]
#t_list = [0, 1, 2, 3, 4, 5]

def main(show_progress_bar=True):
    # Create mesh
    mesh = Mesh(nx, ny, Problem.xL, Problem.xR, Problem.yL, Problem.yR)
    vol_points_copy = mesh.vol_points.copy()
    edge_points_copy = mesh.edge_points.copy()

    # Initial solution
    problem = Problem(mesh.xy, t_list, mesh.bc_type)
    U, phi = problem.get_initial_conditions()
    U_ghost = np.empty((mesh.bc_type.shape[0], 4))

    # Store data
    data = SimulationData(mesh.nx, mesh.ny, U, phi, U_ghost, t_list, problem.g, file_name)
    del U, phi, U_ghost

    # Save initial condition, if desired
    written_times = -np.ones_like(t_list)
    if 0 in t_list:
        data.save_current_state(mesh)
        written_times[0] = 0
        data.t_list[0] = 0

    print('---- Solving ----')
    # Set up progress bar
    with Progress(
            TextColumn('Running iterations...', justify='right'),
            BarColumn(finished_style='purple'),
            '[progress.percentage]{task.percentage:>3.0f}%',
            TimeRemainingColumn(),
            disable = not show_progress_bar) as progress:
        task = progress.add_task('Running iterations...',
                total=t_final)

        # Loop over time
        x_shock = []
        for i in range(max_n_t):
            data.new_iteration()

            # If told to compute timestep using CFL
            if 'cfl' in globals():
                r  = data.U[:, 0]
                ru = data.U[:, 1]
                rv = data.U[:, 2]
                re = data.U[:, 3]
                # Compute pressure
                p = (re - .5 * (ru**2 + rv**2) / r) * (problem.g - 1)
                # Speed of sound
                a = np.sqrt(problem.g * p / r)
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


            # -- Update Mesh -- #
            if adaptive:
                # Store copy of original in case this iteration crashes
                vol_points_copy = mesh.vol_points.copy()
                edge_points_copy = mesh.edge_points.copy()
                # Revert back to original face points
                mesh.vol_points = mesh.original_vol_points.copy()
                mesh.edge_points = mesh.original_edge_points.copy()
                # Update the mesh
                mesh.update(data, problem)
            # Update the stencil to not include points across the interface
            mesh.update_stencil(data.phi)

            # Compute gradients
            compute_gradient(data.U, mesh.xy, mesh.stencil,
                    data.gradU.reshape(-1))
            # Create ghost fluid interfaces
            if ghost_fluid_interfaces:
                mesh.create_interfaces(data, problem.fluid_solid)

            # -- Copy solution, then update -- #
            U_old = data.U.copy()
            data.U = update(dt, data, mesh, problem)
            data.t += dt

            # -- Copy phi, Then Update -- #
            phi_old = data.phi.copy()
            if problem.fluid_solid:
                data.phi = problem.compute_exact_phi(mesh.xy, data.t)
            else:
                if levelset:
                    data.phi = update_phi(dt, data, mesh, problem)

            # -- Update Ghost Fluid Cells -- #
            if ghost_fluid_interfaces and update_ghost_fluid_cells:
                # Find which cells just switched from being in one fluid to being in
                # another
                ghost_IDs = np.argwhere(phi_old * data.phi < 0)[:, 0]
                # Loop over each ghost
                for ghost_ID in ghost_IDs:
                    # Get new sign of phi for this ghost
                    sign = np.sign(data.phi[ghost_ID])
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

                    # Perform extrapolation of the ghost fluid state
                    if linear_ghost_extrapolation:
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
                    else:
                        # Use constant extrapolation
                        data.U[ghost_ID] = np.mean(data.U[fluid_neighbors], axis=0)

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
            if data.t >= data.t_list[write_index]:
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
            progress.update(task, advance=dt)
            # If it's hit the final time, then stop iterating
            if np.isclose(data.t, t_final): break

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

class SimulationData:
    '''
    Class for initializing and storing objects and data.

    Members:
    --------
    i
    t
    U_list
    phi_list
    t_list
    '''
    # Iteration counter
    i = 0
    # Simulation time
    t = 0
    # Current timestep
    dt = 0

    def __init__(self, nx, ny, U, phi, U_ghost, t_list, g, file_name):
        # Save mesh sizing
        self.nx = nx
        self.ny = ny
        # Set the ratio of specific heats
        self.g = 1.4
        # Temporary buffers
        self.U = U
        self.phi = phi
        self.U_ghost = U_ghost
        self.gradU = np.empty((nx*ny, 4, 2))
        # Lists of data for each stored timestep
        self.U_list = []
        self.phi_list = []
        self.t_list = t_list.copy()
        self.iter_list = []
        self.coords_list = []
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
        self.edge_points_list.append(mesh.edge_points.copy())
        self.vol_points_list.append(mesh.vol_points.copy())

    def write_to_file(self):
        with open(file_name, 'wb') as f:
            np.savez(f, nx=self.nx, ny=self.ny, g=self.g, U_list=self.U_list,
                    phi_list=self.phi_list, t_list = self.t_list,
                    edge_points_list=self.edge_points_list,
                    vol_points_list=self.vol_points_list,
                    allow_pickle=True)

    @classmethod
    def read_from_file(cls, file_name):
        with open(file_name, 'rb') as f:
            data = np.load(f)
            sim_data = cls(data['nx'], data['ny'], None, None, None, data['t_list'],
                    data['g'], file_name)
            sim_data.U_list = data['U_list']
            sim_data.phi_list = data['phi_list']
            sim_data.edge_points_list = data['edge_points_list']
            sim_data.vol_points_list = data['vol_points_list']
        return sim_data

if __name__ == '__main__':
    main()
