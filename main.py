import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from mesh import Mesh
from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        TaylorGreen, conservative_to_primitive)
from residual import get_residual, get_residual_phi, Roe, Upwind


# Solver inputs
Problem = TaylorGreen
nx = 10
ny = 10
n_t = 200
t_final = 10#1e2 * .01 / 400
dt = t_final / n_t
adaptive = False

# Domain
xL = 0
xR = 2*np.pi
yL = 0
yR = 2*np.pi

plot_mesh = False#True
plot_contour = True
only_rho = False
plot_ICs = False
filetype = 'pdf'

#t_list = [dt, .004, .008]
t_list = [dt, 4, 8]
#t_list = [dt]

def main():
    compute_solution()

def compute_solution():
    # Create mesh
    mesh = Mesh(nx, ny, xL, xR, yL, yR)

    # Initial solution
    problem = Problem(mesh.xy, t_list)
    U, phi = problem.get_initial_conditions()
    U_ghost = np.empty((mesh.bc_type.shape[0], 4))

    # Store data
    data = SimulationData(U, U_ghost, phi, problem.g)

    # Loop over time
    U_list = []
    phi_list = []
    x_shock = np.empty(n_t)
    for i in range(n_t):
        if plot_ICs:
            U_list.append(U)
            phi_list.append(phi)
            break
        data.new_iteration()
        data.gradU = compute_gradient(data.U, mesh)
        # Update mesh
        if adaptive:
            mesh.update(data)
        # Update solution
        data.U = update(dt, data, mesh, problem)
        data.phi = update_phi(dt, data, mesh, problem)
        data.t = (i + 1) * dt
        if np.any(np.isclose(t_list, data.t)):
            U_list.append(data.U)
            phi_list.append(data.phi)
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

    # Fit a line to the shock location
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
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Density, velocity, and pressure profiles
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
    plt.tight_layout()
    plt.savefig(f'figs/result_{mesh.nx}x{mesh.ny}.{filetype}', bbox_inches='tight')

    # Mesh plots
    if plot_mesh:
        fig, axes = plt.subplots(len(t_list), 1, figsize=(6.5, 8), squeeze=False)
        for i in range(len(t_list)):
            phi = phi_list[i]

            ax = axes[i, 0]
            ax.set_xlim([mesh.xL, mesh.xR])
            ax.set_ylim([mesh.yL, mesh.yR])
            # Loop over primal cells
            for cell_ID in range(mesh.n_primal_cells):
                points = mesh.get_plot_points_primal_cell(cell_ID)
                ax.plot(points[:, 0], points[:, 1], 'k', lw=.5)
            # Loop over dual faces
            for face_ID in range(mesh.n_faces):
                points = mesh.get_face_point_coords(face_ID)
                # Get dual mesh neighbors
                i, j = mesh.edge[face_ID]
                # Check if this is a surrogate boundary
                is_surrogate = phi[i] * phi[j] < 0
                if is_surrogate:
                    options = {'color' : 'k', 'lw' : 2}
                else:
                    options = {'color' : 'k', 'ls' : '--', 'lw' : 1}
                ax.plot(points[:, 0], points[:, 1], **options)

        # Save
        plt.tight_layout()
        plt.savefig(f'figs/mesh_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')

    # Density, velocity, and pressure contour plots
    if plot_contour:
        if only_rho:
            num_vars = 1
        else: num_vars = 3
        fig, axes = plt.subplots(len(t_list), num_vars, figsize=(6.5, 8), squeeze=False)
        for i_iter in range(len(t_list)):
            V = V_list[i_iter]
            phi = phi_list[i_iter]
            t = t_list[i_iter]

            r = V[:, 0]
            u = V[:, 1]
            v = V[:, 2]
            p = V[:, 3]

            # Plotting rho, u, and p
            f = [r, u, p]
            time = f'(t={t} \\textrm{{ s}})'
            ylabels = [f'$\\rho{time}$ (kg/m$^3$)', f'$u{time}$ (m/s)', f'$p{time}$ (N/m$^2$)']
            # Loop over variables
            for idx in range(num_vars):
                ax = axes[i_iter, idx]
                contourf = ax.tricontourf(mesh.xy[:, 0], mesh.xy[:, 1], f[idx])
                plt.colorbar(mappable=contourf, ax=ax)
                ax.set_title(ylabels[idx], fontsize=10)
                ax.tick_params(labelsize=10)

                # Loop over dual faces
                for face_ID in range(mesh.n_faces):
                    points = mesh.get_face_point_coords(face_ID)
                    # Get dual mesh neighbors
                    i, j = mesh.edge[face_ID]
                    # Check if this is a surrogate boundary
                    is_surrogate = phi[i] * phi[j] < 0
                    if is_surrogate:
                        ax.plot(points[:, 0], points[:, 1], 'k', lw=2)

        for idx in range(num_vars):
            axes[-1, idx].set_xlabel('x (m)', fontsize=10)
        for idx in range(len(t_list)):
            axes[idx, 0].set_ylabel('y (m)', fontsize=10)
        # Save
        plt.tight_layout()
        plt.savefig(f'figs/contour_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')

    print(f'Plots written to files ({mesh.nx}x{mesh.ny}).')

def compute_gradient(U, mesh):
    gradU = np.empty((mesh.n, 4, 2))
    # Loop over all cells
    for i in range(mesh.n):
        n_points = len(mesh.stencil[i])
        # Construct A matrix: [x_i, y_i, 1]
        A = np.ones((n_points, 3))
        A[:, :-1] = mesh.xy[mesh.stencil[i]]
        # We desired [x_i, y_i, 1] @ [c0, c1, c2] = U[i], therefore Ax=b.
        # However, there are more equations than unknowns (for most points)
        # so instead, solve the normal equations: A.T @ A x = A.T @ b
        c = np.linalg.solve(A.T @ A, A.T @ U[mesh.stencil[i]])
        # Since U = c0 x + c1 u + c2, then dU/dx = c0 and dU/dy = c1.
        gradU[i] = c[:-1].T
    return gradU

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
    U
    U_ghost
    gradU
    phi
    flux
    flux_phi
    i
    '''
    # Iteration counter
    i = 0
    # Simulation time
    t = 0

    def __init__(self, U, U_ghost, phi, g):
        self.U = U
        self.U_ghost = U_ghost
        self.phi = phi
        # Set flux of flow variables to be Roe
        self.flux = Roe(g)
        # Set the flux of phi to be upwind
        self.flux_phi = Upwind()

    def new_iteration(self):
        # Print
        print(self.i)
        # Update iteration counter
        self.i += 1

if __name__ == '__main__':
    main()
