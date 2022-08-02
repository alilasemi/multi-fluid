import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
from rich.progress import track, Progress

from mesh import Mesh
from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        CollapsingCylinder, conservative_to_primitive)
from solve import SimulationData

Problem = CollapsingCylinder
file_name = 'data.npz'
show_progress_bar = True
plot_profile = True
plot_mesh = True
plot_contour = True
only_rho = False
equal_aspect_ratio = True
filetype = 'pdf'


def post_process():
    # Get data from file
    data = SimulationData.read_from_file(file_name)
    n_times = len(data.t_list)
    # Create mesh
    mesh = Mesh(data.nx, data.ny, Problem.xL, Problem.xR, Problem.yL, Problem.yR)
    # Create problem
    problem = Problem(mesh.xy, data.t_list, mesh.bc_type)
    has_exact_phi = hasattr(problem, 'plot_exact_interface') and callable(
            problem.plot_exact_interface)

    # Final primitives
    V_list = []
    for U in data.U_list:
        V = np.empty_like(U)
        for i in range(mesh.n):
            V[i] = conservative_to_primitive(
                    U[i, 0], U[i, 1], U[i, 2], U[i, 3], data.g)
        V_list.append(V)

    # Plot
    print('---- Plotting ----')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Density, velocity, and pressure profiles
    if plot_profile:
        print('Plotting profiles...')
        if only_rho:
            fig, axes = plt.subplots(n_times, 1, figsize=(10, 3), squeeze=False)
            num_vars = 1
        else:
            fig, axes = plt.subplots(n_times, 3, figsize=(6.5, 6.5), squeeze=False)
            num_vars = 3
        for i in range(n_times):
            V = V_list[i]
            phi = data.phi_list[i]
            t = data.t_list[i]

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
            nx, ny = data.nx, data.ny
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
        save_plot('profile', mesh)

    # Mesh plots
    if plot_mesh:
        fig, axes = plt.subplots(n_times, 1, figsize=(6.5, 4*n_times),
                squeeze=False)
        # Progress bar setup
        with Progress() as progress:
            task1 = progress.add_task('Plotting primal cells...',
                    total=n_times * mesh.n_primal_cells)
            task2 = progress.add_task('Plotting dual faces...',
                    total=n_times * mesh.n_faces)

            for i_iter in range(n_times):
                phi = data.phi_list[i_iter]
                edge_points_list = data.edge_points_list[i_iter]
                vol_points_list = data.vol_points_list[i_iter]

                ax = axes[i_iter, 0]
                if equal_aspect_ratio:
                    ax.set_aspect('equal', adjustable='box')
                ax.set_xlim([mesh.xL, mesh.xR])
                ax.set_ylim([mesh.yL, mesh.yR])
                if has_exact_phi:
                    problem.plot_exact_interface(ax, mesh, data.t_list[i_iter])
                # Loop over primal cells
                for cell_ID in range(mesh.n_primal_cells):
                    points = mesh.get_plot_points_primal_cell(cell_ID)
                    ax.plot(points[:, 0], points[:, 1], 'k', lw=.5)
                    progress.update(task1, advance=1)
                # Loop over dual faces
                for face_ID in range(mesh.n_faces):
                    points = mesh.get_face_point_coords(face_ID,
                            edge_points_list, vol_points_list)
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
        save_plot('mesh', mesh)

    # Density, velocity, and pressure contour plots
    if plot_contour:
        if only_rho:
            num_vars = 1
        else: num_vars = 4
        fig, axes = plt.subplots(n_times, num_vars, figsize=(5 * num_vars, 4*n_times), squeeze=False)
        # Progress bar setup
        with Progress() as progress:
            task1 = progress.add_task('Plotting contours...',
                    total=n_times * num_vars)
            for i_iter in range(n_times):
                V = V_list[i_iter]
                phi = data.phi_list[i_iter]
                t = data.t_list[i_iter]
                edge_points_list = data.edge_points_list[i_iter]
                vol_points_list = data.vol_points_list[i_iter]

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
                    if has_exact_phi:
                        problem.plot_exact_interface(ax, mesh, data.t_list[i_iter])

                    # Loop over dual faces
                    for face_ID in range(mesh.n_faces):
                        points = mesh.get_face_point_coords(face_ID,
                                edge_points_list, vol_points_list)
                        # Get dual mesh neighbors
                        i, j = mesh.edge[face_ID]
                        # Check if this is a surrogate boundary
                        is_surrogate = phi[i] * phi[j] < 0
                        if is_surrogate:
                            ax.plot(points[:, 0], points[:, 1], 'k', lw=2)
                    progress.update(task1, advance=1)
                    # TODO: Quick hack for NaNs
                    if i_iter == n_times - 1:
                        nan_IDs = np.array([662, 663, 896, 936], dtype=int)
                        points = mesh.xy[nan_IDs]
                        ax.plot(points[:, 0], points[:, 1], 'ow', mfc='None', ms=6)

        for idx in range(num_vars):
            axes[-1, idx].set_xlabel('x (m)', fontsize=10)
        for idx in range(n_times):
            axes[idx, 0].set_ylabel('y (m)', fontsize=10)
        # Save
        save_plot('contour', mesh)

    print(f'Plots written to files ({mesh.nx}x{mesh.ny}).')

def save_plot(plot_name, mesh):
    print(f'Saving {plot_name} plot...', end='', flush=True)
    plt.tight_layout()
    plot_file = f'{plot_name}_{mesh.nx}x{mesh.ny}.{filetype}'
    # Write figure to file
    plt.savefig(f'figs/{plot_file}', bbox_inches='tight')
    # Create symlink
    os.symlink(plot_file, f'figs/new_result.{filetype}')
    os.replace(f'figs/new_result.{filetype}', f'figs/result.{filetype}')
    print('Done')

if __name__ == '__main__':
    post_process()
