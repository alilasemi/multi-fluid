import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
from rich.progress import track, Progress

from mesh import Mesh
from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        CollapsingCylinder, Star, Cavitation, conservative_to_primitive)
from solve import SimulationData, adaptive

Problem = Cavitation
file_name = 'data.npz'
show_progress_bar = True
plot_profile = False
plot_mesh = True
plot_contour = True
mark_volume_points = False
plot_phi_contours = True
only_phi = False
only_rho = False
equal_aspect_ratio = True
mesh_legend = False
filetype = 'pdf'


def post_process():
    # Get data from file
    data = SimulationData.read_from_file(file_name)
    n_times = len(data.t_list)
    # Create mesh
    mesh = Mesh(data.nx, data.ny, Problem.xL, Problem.xR, Problem.yL,
            Problem.yR, adaptive)
    # Create problem
    problem = Problem(mesh.xy, data.t_list, mesh.bc_type, data.g, data.psg)
    has_exact_phi = hasattr(problem, 'plot_exact_interface') and callable(
            problem.plot_exact_interface)
    # TODO: This is disabled for now
    has_exact_phi = False

    # Final primitives
    V_list = []
    for i_iter, U in enumerate(data.U_list):
        V = np.empty_like(U)
        for i in range(mesh.n):
            # Get fluid data for each cell
            fluid_ID = data.fluid_ID_list[i_iter]
            g_i = data.g[fluid_ID[i]]
            psg_i = data.psg[fluid_ID[i]]
            V[i] = conservative_to_primitive(
                    U[i, 0], U[i, 1], U[i, 2], U[i, 3], g_i, psg_i)
        V_list.append(V)

    # Plot
    print('---- Plotting ----')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

#    # TODO: This is basically a hack to plot mass conservation error
#    bubble_mass = np.empty(n_times)
#    for i_iter in range(n_times):
#        # Extract  data
#        V = V_list[i_iter]
#        r = V[:, 0]
#        phi = data.phi_list[i_iter]
#        t = data.t_list[i_iter]
#        edge_points = data.edge_points_list[i_iter]
#        vol_points = data.vol_points_list[i_iter]
#        # Get mesh areas
#        mesh.edge_points = edge_points
#        mesh.vol_points = vol_points
#        mesh.compute_cell_areas()
#
#        # Compute total mass of bubble
#        bubble_IDs = np.argwhere(phi < 0)[:, 0]
#        bubble_mass[i_iter] = np.sum(r[bubble_IDs] * mesh.area[bubble_IDs])
#    # Save data
#    from solve import adaptive
#    bubble_data = np.empty((n_times, 2))
#    bubble_data[:, 0] = data.t_list
#    bubble_data[:, 1] = bubble_mass
#    np.savetxt(f'bubble_data/bubble_data_{mesh.nx}_{adaptive}.txt', bubble_data)
#    quit()
#    # Plot
#    plt.plot(data.t_list, bubble_mass, '-k', lw=2)
#    plt.tight_layout()
#    plt.savefig(f'bubble_mass.pdf', bbox_inches='tight')
#    #plt.show()

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
                    axes[i, idx].set_xlabel('$x$ (m)', fontsize=10)
                ax.set_ylabel(ylabels[idx], fontsize=10)
                ax.tick_params(labelsize=10)
                ax.grid(linestyle='--')
                ax.set_xlim([mesh.xL, mesh.xR])
        for idx in range(num_vars):
            if not only_rho:
                axes[-1, idx].set_xlabel('$x$ (m)', fontsize=10)
        # Save
        save_plot('profile', mesh)

    # Mesh plots
    seperate_files = False
    lw_scale = .5
    if plot_mesh:
        if not seperate_files:
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

                if seperate_files:
                    fig = plt.figure(figsize=(6.5, 4))
                    ax = plt.gca()
                else:
                    ax = axes[i_iter, 0]

                if equal_aspect_ratio:
                    ax.set_aspect('equal', adjustable='box')
                #ax.set_xlim([mesh.xL, mesh.xR])
                #ax.set_ylim([mesh.yL, mesh.yR])
                lim = .048
                #ax.set_xlim([-lim, lim])
                #ax.set_ylim([-lim, lim])
                ax.set_xlabel('$x$ (m)', fontsize=10)
                ax.set_ylabel('$y$ (m)', fontsize=10)
                if has_exact_phi:
                    problem.plot_exact_interface(ax, mesh, data.t_list[i_iter],
                            lw_scale)
                # Loop over primal cells
                for cell_ID in range(mesh.n_primal_cells):
                    points = mesh.get_plot_points_primal_cell(cell_ID)
                    ax.plot(points[:, 0], points[:, 1], 'k', lw=.5*lw_scale)
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
                        options = {'color': 'k', 'lw': 3*lw_scale}
                    else:
                        options = {'color': 'k', 'ls': '--', 'lw': 1*lw_scale}
                    ax.plot(points[:, 0], points[:, 1], **options)
                    if mark_volume_points:
                        options = {'color': 'purple', 'marker': 'x',
                                'ls': 'None', 'ms': 10*lw_scale,
                                'mew': 2*lw_scale}
                        if points.shape[0] == 3:
                            indices = [0, 2]
                        else:
                            indices = [0,]
                        ax.plot(points[indices, 0], points[indices, 1], **options)
                    progress.update(task2, advance=1)

                # Write each frame seperately if desired
                if seperate_files:
                    if mesh_legend:
                        plt.plot(0, 0, 'k', lw=.5*lw_scale, label='Primal Mesh')
                        plt.plot(0, 0, '--k', lw=1*lw_scale, label='Dual Mesh')
                        plt.plot(0, 0, 'k', lw=3*lw_scale, label='Surrogate Interface')
                        plt.plot(0, 0, 'r', lw=3*lw_scale, label='True Interface')
                        plt.legend(fontsize=10, loc='lower right', framealpha=.9)
                    save_plot(f'meshes/mesh_{i_iter}', mesh, seperate_files)
                    plt.close(fig)

        # Save
        if not seperate_files:
            if mesh_legend:
                plt.plot(0, 0, 'k', lw=.5*lw_scale, label='Primal Mesh')
                plt.plot(0, 0, '--k', lw=1*lw_scale, label='Dual Mesh')
                plt.plot(0, 0, 'k', lw=3*lw_scale, label='Surrogate Interface')
                plt.plot(0, 0, 'r', lw=3*lw_scale, label='True Interface')
                plt.legend(fontsize=10, loc='lower right', framealpha=.9)
            save_plot('mesh', mesh)

    # Density, velocity, and pressure contour plots
    radius = np.zeros(n_times)
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
                phi_label = f'$\\phi{time}$'
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
                    if (idx != 0) or (not only_phi):
                        contourf = ax.tricontourf(mesh.xy[:, 0], mesh.xy[:, 1], f[idx],
                                levels=levels, extend='both')
                        plt.colorbar(mappable=contourf, ax=ax)
                        ax.set_title(ylabels[idx], fontsize=10)
                    elif (idx == 0) and only_phi:
                        ax.set_title(phi_label, fontsize=10)
                    ax.tick_params(labelsize=10)
                    ax.set_xlim([mesh.xL, mesh.xR])
                    ax.set_ylim([mesh.yL, mesh.yR])
                    # Plot exact interface location
                    if has_exact_phi:
                        problem.plot_exact_interface(ax, mesh,
                                data.t_list[i_iter], .5)

                    # Plot phi contours on top of density plot
                    if plot_phi_contours and idx == 0:
                        if np.any(np.isnan(phi)):
                            print('Warning: NaNs detected in phi!')
                        phi[np.argwhere(np.isnan(phi))[:,0]] = -10
                        contour = ax.tricontour(mesh.xy[:, 0], mesh.xy[:, 1],
                                phi, colors='k', extend='both',
                                linewidths=1*lw_scale)
                        zero_contour = ax.tricontour(mesh.xy[:, 0], mesh.xy[:, 1],
                                phi, levels = [0,], colors='purple',
                                linewidths=2*lw_scale, linestyles='dashed')

                        # Plot nodes near interface
                        # TODO: This method is probably most accurate, but
                        # results in some wierd (non-smooth) plots for a low
                        # order interface.
                        if False:#adaptive:
                            is_surrogate = phi[mesh.edge[:, 0]] * phi[mesh.edge[:, 1]] < 0
                            interface_IDs = np.argwhere(is_surrogate)[:, 0]
                            points = [mesh.get_face_point_coords(face_ID,
                                    edge_points_list, vol_points_list) for face_ID
                                    in interface_IDs]
                            radii = [np.mean(np.linalg.norm(p, axis=1)) for p in
                                    points]
                            radius[i_iter] = np.mean(radii)
                        else:
                            cutoff = .01
                            near_interface = np.abs(phi) < cutoff
                            ax.plot(mesh.xy[near_interface, 0],
                                    mesh.xy[near_interface, 1], 'r.', ms=5)
                            # Update radius based on a weighted average of these
                            # points
                            weights = 1 - np.abs(phi[near_interface]) / cutoff
                            radii = np.linalg.norm(mesh.xy[near_interface], axis=1)
                            radius[i_iter] = (weights @ radii) / np.sum(weights)

#                        # Update radius based on the area of the bubble
#                        areas = data.area_list[i_iter][data.fluid_ID_list[i_iter].astype(bool)]
#                        radius[i_iter] = np.sqrt(areas.sum() / np.pi)

                    # Update progress bar
                    progress.update(task1, advance=1)

        for idx in range(num_vars):
            axes[-1, idx].set_xlabel('$x$ (m)', fontsize=10)
        for idx in range(n_times):
            axes[idx, 0].set_ylabel('$y$ (m)', fontsize=10)
        # Save
        save_plot('contour', mesh)
        # Write radius to file for separate processing
        data = np.append(data.t_list.reshape(-1, 1), radius.reshape(-1, 1), axis=1)
        np.savetxt('radius.txt', data, delimiter=",")

    print(f'Plots written to files ({mesh.nx}x{mesh.ny}).')

def save_plot(plot_name, mesh, seperate_files=False):
    print(f'Saving {plot_name} plot...', end='', flush=True)
    plt.tight_layout()
    plot_file = f'{plot_name}_{mesh.nx}x{mesh.ny}.{filetype}'
    # Write figure to file
    plt.savefig(f'figs/{plot_file}', bbox_inches='tight')
    # Create symlink
    if not seperate_files:
        os.symlink(plot_file, f'figs/new_{plot_name}.{filetype}')
        os.replace(f'figs/new_{plot_name}.{filetype}', f'figs/{plot_name}.{filetype}')
    print('done')

if __name__ == '__main__':
    post_process()
