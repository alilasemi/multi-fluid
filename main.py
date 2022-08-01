import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import rich.traceback
rich.traceback.install()
from rich.progress import track, Progress

from mesh import Mesh
from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        CollapsingCylinder, conservative_to_primitive)
from residual import get_residual, get_residual_phi, Upwind


# Solver inputs
Problem = CollapsingCylinder
nx = 20
ny = 20
n_t = 4
t_final = .0005 / 50
dt = t_final / n_t
adaptive = False
rho_levels = np.linspace(.15, 1.05, 19)

# Domain
xL = -1
xR = 1
yL = -1
yR = 1

show_progress_bar = False
ghost_fluid_interfaces = True
update_ghost_fluid_cells = True
linear_ghost_extrapolation = True
hardcoded_phi = True
levelset = True
plot_profile = False
plot_mesh = True
plot_contour = True
only_rho = False
plot_ICs = False
equal_aspect_ratio = True
filetype = 'pdf'

#t_list = [dt, .025, .05, .075, .1]
#t_list = [dt, .0025, .005, .0075, .01]
#t_list = [dt, .00025 ,.0005, .00075, .001]
#t_list = [dt, .000125 ,.00025, .000375, .0005]
#t_list = [dt, .000125 ,.00025, .000375, .0005,
#        .000625, .00075, .000825, .001]
#t_list = [.01]
#t_list = [dt, .004, .008]
#t_list = [dt, 4, 8]
#t_list = [dt, 8*dt, 16*dt, 24*dt, 32*dt, 40*dt]
#t_list = [dt, 4*dt, 8*dt, 12*dt, 16*dt, 20*dt]
#t_list = [dt, 2*dt, 3*dt, 4*dt, 5*dt, 6*dt, 7*dt]
t_list = [dt, 2*dt, 3*dt, 4*dt]
#t_list = [dt]

def main():
    compute_solution()

def compute_solution():
    # Create mesh
    mesh = Mesh(nx, ny, xL, xR, yL, yR)
    original_vol_points = mesh.vol_points.copy()
    original_edge_points = mesh.edge_points.copy()

    # Initial solution
    problem = Problem(mesh.xy, t_list, mesh.bc_type)
    U, phi = problem.get_initial_conditions()

    # Store data
    data = SimulationData(U, phi, problem.g)
    data.U_ghost = np.empty((mesh.bc_type.shape[0], 4))
    #TODO
    data.coords_list = []

    # Loop over time
    U_list = []
    phi_list = []
    x_shock = np.empty(n_t)
    print('---- Solving ----')
    for i in track(range(n_t), description="Running iterations...",
            finished_style='purple', disable=not show_progress_bar):
        if plot_ICs:
            U_list.append(U)
            phi_list.append(phi)
            break
        data.new_iteration()

        # -- Update Mesh -- #
        if adaptive:
            # Revert back to original face points
            mesh.vol_points = original_vol_points.copy()
            mesh.edge_points = original_edge_points.copy()
            # Update the mesh
            mesh.update(data, problem)
        # Update the stencil to not include points across the interface
        mesh.update_stencil(data.phi)

        # -- Update Solution -- #
        # Compute gradients
        data.gradU = compute_gradient(data.U, mesh)
        # Create ghost fluid interfaces
        if ghost_fluid_interfaces:
            mesh.create_interfaces(data)
        # Update solution
        data.U = update(dt, data, mesh, problem)
        data.t = (i + 1) * dt

        # -- Copy phi, Then Update -- #
        phi_old = data.phi.copy()
        if hardcoded_phi:
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

        # Store data
        if np.any(np.isclose(t_list, data.t)):
            U_list.append(data.U.copy())
            phi_list.append(data.phi.copy())
            # Store the new face points, for plotting later
            data.coords_list.append([mesh.vol_points.copy(),
                    mesh.edge_points.copy()])
        # Find shock
        if Problem == RiemannProblem:
            for j in range(nx):
                # Jump in x-velocity
                # TODO Cleaner indices
                line = ny // 2
                u4 = problem.state_4[1]
                delta_u = data.U[line*nx + nx - 1 - j, 1] / data.U[line*nx + nx - 1 - j, 0] - u4
                if delta_u > .01 * u4:
                    x_shock[i] = mesh.xy[nx - 1 - j, 0]
                    break

    # Copy the original edge back. This is to make plotting not skip interfaces
    mesh.edge = mesh.original_edge.copy()

    # Fit a line to the shock location
    if Problem == RiemannProblem:
        try:
            fit_shock = np.polyfit(np.linspace(dt, t_final, n_t), x_shock, 1)
            shock_speed = fit_shock[0]
            print(f'The shock speed is {shock_speed} m/s.')
        except np.linalg.LinAlgError:
            print('-- Shock speed calculation failed! --')

    # Final primitives
    V_list = []
    for U in U_list:
        V = np.empty_like(U)
        for i in range(mesh.n):
            V[i] = conservative_to_primitive(
                    U[i, 0], U[i, 1], U[i, 2], U[i, 3], problem.g)
        V_list.append(V)

    # Save solution
    with open('U.npy', 'wb') as f:
        np.save(f, data.U)

    # Plot
    print('---- Plotting ----')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Density, velocity, and pressure profiles
    if plot_profile:
        print('Plotting profiles...')
        if only_rho:
            fig, axes = plt.subplots(len(t_list), 1, figsize=(10, 3), squeeze=False)
            num_vars = 1
        else:
            fig, axes = plt.subplots(len(t_list), 3, figsize=(6.5, 6.5), squeeze=False)
            num_vars = 3
        for i in range(len(t_list)):
            V = V_list[i]
            phi = phi_list[i]
            t = t_list[i]

            r = V[:, 0]
            u = V[:, 1]
            v = V[:, 2]
            p = V[:, 3]

            # Exact solution
            if problem.exact:
                with open(f'data/r_exact_t_{t}.npy', 'rb') as f:
                    r_exact = np.load(f)
                with open(f'data/u_exact_t_{t}.npy', 'rb') as f:
                    u_exact = np.load(f)
                with open(f'data/p_exact_t_{t}.npy', 'rb') as f:
                    p_exact = np.load(f)
                with open(f'data/x_exact_t_{t}.npy', 'rb') as f:
                    x_exact = np.load(f)
                f_exact = [r_exact, u_exact, p_exact]

            # Index of y to slice for plotting
            j = ny // 2
            # Plotting rho, u, and p
            f = [r, u, p]
            time = f'(t={t} \\textrm{{ s}})'
            ylabels = [f'$\\rho{time}$ (kg/m$^3$)', f'$u{time}$ (m/s)', f'$p{time}$ (N/m$^2$)']
            # Loop over rho, u, p
            for idx in range(num_vars):
                ax = axes[i, idx]
                ax.plot(mesh.xy[j*nx:(j+1)*nx, 0], f[idx][j*nx:(j+1)*nx], 'k', linewidth=2)
                # Plot phi on top of rho
                if idx == 0:
                    # Plot phi
                    ax.plot(mesh.xy[j*nx:(j+1)*nx, 0], phi[j*nx:(j+1)*nx], 'k', linewidth=1)
                    # Find interface and plot as a vertical line
                    i_interface = np.argmin(np.abs(phi[j*nx:(j+1)*nx]))
                    # Make sure no scaling happens with this tall vertical line
                    ax.autoscale(False)
                    ax.vlines(mesh.xy[j*nx:(j+1)*nx, 0][i_interface], mesh.yL,
                            mesh.yR, color='r', ls='--')
                if problem.exact:
                    ax.plot(x_exact, f_exact[idx], '--k', linewidth=1)
                if only_rho:
                    axes[i, idx].set_xlabel('x (m)', fontsize=10)
                ax.set_ylabel(ylabels[idx], fontsize=10)
                ax.tick_params(labelsize=10)
                ax.grid(linestyle='--')
                ax.set_xlim([mesh.xL, mesh.xR])
        for idx in range(num_vars):
            if not only_rho:
                axes[-1, idx].set_xlabel('x (m)', fontsize=10)
        # Save
        print('Saving profile plot...', end='', flush=True)
        plt.tight_layout()
        plt.savefig(f'figs/result_{mesh.nx}x{mesh.ny}.{filetype}', bbox_inches='tight')
        print('Done')

    # Mesh plots
    if plot_mesh:
        fig, axes = plt.subplots(len(t_list), 1, figsize=(6.5, 4*len(t_list)),
                squeeze=False)
        # Progress bar setup
        with Progress() as progress:
            task1 = progress.add_task('Plotting primal cells...',
                    total=len(t_list) * mesh.n_primal_cells)
            task2 = progress.add_task('Plotting dual faces...',
                    total=len(t_list) * mesh.n_faces)

            for i_iter in range(len(t_list)):
                phi = phi_list[i_iter]
                coords = data.coords_list[i_iter]

                ax = axes[i_iter, 0]
                if equal_aspect_ratio:
                    ax.set_aspect('equal', adjustable='box')
                ax.set_xlim([mesh.xL, mesh.xR])
                ax.set_ylim([mesh.yL, mesh.yR])
                if hardcoded_phi:
                    problem.plot_exact_interface(ax, mesh, t_list[i_iter])
                # Loop over primal cells
                for cell_ID in range(mesh.n_primal_cells):
                    points = mesh.get_plot_points_primal_cell(cell_ID)
                    ax.plot(points[:, 0], points[:, 1], 'k', lw=.5)
                    progress.update(task1, advance=1)
                # Loop over dual faces
                for face_ID in range(mesh.n_faces):
                    points = mesh.get_face_point_coords(face_ID, *coords)
                    # Get dual mesh neighbors
                    i, j = mesh.edge[face_ID]
                    # Check if this is a surrogate boundary
                    is_surrogate = phi[i] * phi[j] < 0
                    if is_surrogate:
                        options = {'color' : 'k', 'lw' : 2}
                    else:
                        options = {'color' : 'k', 'ls' : '--', 'lw' : 1}
                    ax.plot(points[:, 0], points[:, 1], **options)
                    progress.update(task2, advance=1)

        # Save
        print('Saving mesh plot...', end='', flush=True)
        plt.tight_layout()
        plt.savefig(f'figs/mesh_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')
        print('Done')

    # Density, velocity, and pressure contour plots
    if plot_contour:
        if only_rho:
            num_vars = 1
        else: num_vars = 4
        fig, axes = plt.subplots(len(t_list), num_vars, figsize=(5 * num_vars, 4*len(t_list)), squeeze=False)
        # Progress bar setup
        with Progress() as progress:
            task1 = progress.add_task('Plotting contours...',
                    total=len(t_list) * num_vars)
            for i_iter in range(len(t_list)):
                V = V_list[i_iter]
                phi = phi_list[i_iter]
                t = t_list[i_iter]
                coords = data.coords_list[i_iter]

                r = V[:, 0]
                u = V[:, 1]
                v = V[:, 2]
                p = V[:, 3]

                # Plotting rho, u, and p
                f = [r, u, v, p]
                time = f'(t={t} \\textrm{{ s}})'
                ylabels = [f'$\\rho{time}$ (kg/m$^3$)', f'$u{time}$ (m/s)',
                        f'$v{time}$ (m/s)', f'$p{time}$ (N/m$^2$)']
                # Loop over variables
                for idx in range(num_vars):
                    ax = axes[i_iter, idx]
                    if equal_aspect_ratio:
                        ax.set_aspect('equal', adjustable='box')
                    #TODO Make levels less jank
                    if only_rho:
                        levels = rho_levels
                    else:
                        # Try using the problem's defined levels
                        try:
                            levels = problem.levels[idx]
                        # If there aren't any defined, let the plotter choose
                        # its own
                        except AttributeError:
                            levels = None
                    contourf = ax.tricontourf(mesh.xy[:, 0], mesh.xy[:, 1], f[idx],
                            levels=levels, extend='both')
                    plt.colorbar(mappable=contourf, ax=ax)
                    ax.set_title(ylabels[idx], fontsize=10)
                    ax.tick_params(labelsize=10)
                    if hardcoded_phi:
                        problem.plot_exact_interface(ax, mesh, t_list[i_iter])

                    # Loop over dual faces
                    for face_ID in range(mesh.n_faces):
                        points = mesh.get_face_point_coords(face_ID, *coords)
                        # Get dual mesh neighbors
                        i, j = mesh.edge[face_ID]
                        # Check if this is a surrogate boundary
                        is_surrogate = phi[i] * phi[j] < 0
                        if is_surrogate:
                            ax.plot(points[:, 0], points[:, 1], 'k', lw=2)
                    progress.update(task1, advance=1)

        for idx in range(num_vars):
            axes[-1, idx].set_xlabel('x (m)', fontsize=10)
        for idx in range(len(t_list)):
            axes[idx, 0].set_ylabel('y (m)', fontsize=10)
        # Save
        print('Saving contour plot...', end='', flush=True)
        plt.tight_layout()
        plt.savefig(f'figs/contour_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')
        print('Done')

    print(f'Plots written to files ({mesh.nx}x{mesh.ny}).')

def compute_gradient(U, mesh):
    gradU = np.empty((mesh.n, 4, 2))
    # Loop over all cells
    for i in range(mesh.n):
        n_points = len(mesh.stencil[i])
        # If there are no other points in the stencil, then set the gradient to
        # zero
        if n_points == 1:
            gradU[i] = 0
        # Otherwise, solve with least squares
        else:
            # Construct A matrix: [x_i, y_i, 1]
            A = np.ones((n_points, 3))
            A[:, :-1] = mesh.xy[mesh.stencil[i]]
            # We desired [x_i, y_i, 1] @ [c0, c1, c2] = U[i], therefore Ax=b.
            # However, there are more equations than unknowns (for most points)
            # so instead, solve the normal equations: A.T @ A x = A.T @ b
            try:
                c = np.linalg.solve(A.T @ A, A.T @ U[mesh.stencil[i]])
                # Since U = c0 x + c1 y + c2, then dU/dx = c0 and dU/dy = c1.
                gradU[i] = c[:-1].T
            except:
                print(f'Gradient calculation failed! Stencil = {mesh.stencil[i]}')
                gradU[i] = 0
    return gradU

def update(dt, data, mesh, problem):
    U_new = data.U.copy()
    # Compute residual
    R = get_residual(data, mesh, problem)
    # Check for NaNs
    nan_IDs = np.unique(np.argwhere(np.isnan(R))[:, 0])
    if nan_IDs.size != 0:
        message = 'Oh no! NaN detected in the residual!\n'
        message += f'The following {nan_IDs.size} cell residuals are all NaN\'d out:\n'
        message += f'{nan_IDs}'
        raise FloatingPointError(message)
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
    U
    U_ghost
    gradU
    phi
    flux_phi
    i
    t
    '''
    # Iteration counter
    i = 0
    # Simulation time
    t = 0

    def __init__(self, U, phi, g):
        self.U = U
        self.phi = phi
        # Set the flux of phi to be upwind
        self.flux_phi = Upwind()
        # Set the ratio of specific heats
        self.g = 1.4

    def new_iteration(self):
        # Update iteration counter
        self.i += 1

if __name__ == '__main__':
    main()
