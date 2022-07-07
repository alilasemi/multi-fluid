import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sympy as sp
import pathlib
import pickle

from problem import (RiemannProblem, AdvectedContact, AdvectedBubble,
        conservative_to_primitive)
from mesh import Mesh


# Solver inputs
Problem = AdvectedBubble
nx = 40
ny = 40
n_t = 400
t_final = .01
dt = t_final / n_t
include_phi_source = False

# Domain
xL = -1
xR = 1
yL = -1
yR = 1

plot_mesh = False
plot_contour = True
only_rho = True
filetype = 'pdf'

t_list = [dt, .004, .008]

def main():
    compute_solution()

def compute_solution():
    # Create mesh
    mesh = Mesh(nx, ny, xL, xR, yL, yR)

    # Initial solution
    problem = Problem(mesh.xy, t_list)
    U, phi = problem.get_initial_conditions()

    # Set flux of flow variables to be Roe
    flux = Roe(problem.g)
    # Set the flux of phi to be upwind
    flux_phi = Upwind()

    # Loop over time
    U_list = []
    U_ghost = np.empty((mesh.bc_type.shape[0], 4))
    phi_list = []
    x_shock = np.empty(n_t)
    for i in range(n_t):
        print(i)
        gradU = compute_gradient(U, mesh)
        # Update mesh
        #mesh = update_mesh(i, mesh)
        # Update solution
        U = update(i, U, U_ghost, gradU, dt, mesh, problem, flux.compute_flux)
        phi = update_phi(i, U, U_ghost, gradU, phi, dt, mesh, problem, flux_phi.compute_flux)
        t = (i + 1) * dt
        if np.any(np.isclose(t_list, t)):
            U_list.append(U)
            phi_list.append(phi)
        # Find shock
        if Problem == RiemannProblem:
            for j in range(nx):
                # Jump in x-velocity
                # TODO Cleaner indices
                line = ny // 2
                u4 = problem.state_4[1]
                delta_u = U[line*nx + nx - 1 - j, 1] / U[line*nx + nx - 1 - j, 0] - u4
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

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Density, velocity, and pressure profiles
    if only_rho:
        fig, axes = plt.subplots(1, len(t_list), figsize=(10, 3))
        num_vars = 1
    else:
        fig, axes = plt.subplots(len(t_list), 3, figsize=(6.5, 6.5))
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
        with open(f'data/r_exact_t_{t}.npy', 'rb') as f:
            r_exact = np.load(f)
        with open(f'data/u_exact_t_{t}.npy', 'rb') as f:
            u_exact = np.load(f)
        with open(f'data/p_exact_t_{t}.npy', 'rb') as f:
            p_exact = np.load(f)
        with open(f'data/x_exact_t_{t}.npy', 'rb') as f:
            x_exact = np.load(f)

        # Index of y to slice for plotting
        j = ny // 2
        # Plotting rho, u, and p
        f = [r, u, p]
        f_exact = [r_exact, u_exact, p_exact]
        time = f'(t={t} \\textrm{{ s}})'
        ylabels = [f'$\\rho{time}$ (kg/m$^3$)', f'$u{time}$ (m/s)', f'$p{time}$ (N/m$^2$)']
        # Loop over rho, u, p
        for idx in range(num_vars):
            if only_rho:
                ax = axes[i]
            else:
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
            ax.plot(x_exact, f_exact[idx], '--k', linewidth=1)
            if only_rho:
                axes[i].set_xlabel('x (m)', fontsize=10)
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
        fig, axes = plt.subplots(len(t_list), 1, figsize=(6.5, 8))
        for i in range(len(t_list)):
            phi = phi_list[i]

            ax = axes[i]
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
            f_exact = [r_exact, u_exact, p_exact]
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

def update(i_iter, U, U_ghost, gradU, dt, mesh, problem, flux_function):
    U_new = U.copy()
    # L and R cell IDs
    L = mesh.edge[:, 0]
    R = mesh.edge[:, 1]
    # Evaluate solution at faces on left and right
    # -- First order component -- #
    U_L = U[mesh.edge[:, 0]]
    U_R = U[mesh.edge[:, 1]]

    # -- Second order component -- #
    # Get edge midpoints
    edge_midpoint = .5 * (mesh.xy[L] + mesh.xy[R])

    # Compute the limiter value
    # This uses the multidimensional Barth-Jesperson limiter from:
    # https://arc.aiaa.org/doi/pdf/10.2514/6.1989-366
    # TODO Why is this damping needed?
    damping = .8
    limiter = np.empty((mesh.n, 4))
    # Loop state vars
    for k in range(4):
        # Compute extrapolated  value
        midpoint = .5 * (mesh.xy.reshape(-1, 1, 2) + mesh.xy[mesh.limiter_stencil])
        U_face = U[:, k].reshape(-1, 1) + np.einsum('id, ijd -> ij', gradU[:, k], midpoint - mesh.xy[mesh.limiter_stencil])
        u_A_min = np.min(U[mesh.limiter_stencil, k], axis=1)
        u_A_max = np.max(U[mesh.limiter_stencil, k], axis=1)
        # Limiter value for all neighbors of cell i
        limiter_j = np.empty((mesh.n, mesh.max_limiter_stencil_size))
        # Condition 1
        index = np.nonzero((U_face - U[:, k].reshape(-1, 1)) > 0)
        limiter_j[index] = (u_A_max[index[0]] - U[index[0], k]) / (U_face[index] - U[index[0], k])
        # Condition 2
        index = np.nonzero((U_face - U[:, k].reshape(-1, 1)) < 0)
        limiter_j[index] = (u_A_min[index[0]] - U[index[0], k]) / (U_face[index] - U[index[0], k])
        # Condition 3
        index = np.nonzero((U_face - U[:, k].reshape(-1, 1)) == 0)
        limiter_j[index] = 1
        # Take the minimum across each face point
        limiter[:, k] = damping * np.min(limiter_j, axis=1)

    U_L += limiter[L] * np.einsum('ijk, ik -> ij', gradU[L], edge_midpoint - mesh.xy[L])
    U_R += limiter[R] * np.einsum('ijk, ik -> ij', gradU[R], edge_midpoint - mesh.xy[R])

    # Evalute interior fluxes
    F = flux_function(U_L, U_R, mesh.edge_area_normal)

    # Compute ghost state
    problem.compute_ghost_state(U, U_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    F_bc = flux_function(U[mesh.bc_type[:, 0]], U_ghost, mesh.bc_area_normal)

    # Update cells on the left and right sides, for interior faces
    cellL_ID = mesh.edge[:, 0]
    cellR_ID = mesh.edge[:, 1]
    np.add.at(U_new, cellL_ID, -dt / mesh.area[cellL_ID].reshape(-1, 1) * F)
    np.add.at(U_new, cellR_ID,  dt / mesh.area[cellR_ID].reshape(-1, 1) * F)
    # Incorporate boundary faces
    cellL_ID = mesh.bc_type[:, 0]
    np.add.at(U_new, cellL_ID, -dt / mesh.area[cellL_ID].reshape(-1, 1) * F_bc)
    return U_new

def update_phi(i_iter, U, U_ghost, gradU, phi, dt, mesh, problem, flux_function):
    phi_new = phi.copy()
    # Evaluate solution at faces on left and right
    U_L = U[mesh.edge[:, 0]]
    U_R = U[mesh.edge[:, 1]]
    phi_L = phi[mesh.edge[:, 0]]
    phi_R = phi[mesh.edge[:, 1]]
    # Evalute interior fluxes
    F = flux_function(U_L, U_R, phi_L, phi_R, mesh.edge_area_normal)

    # Compute ghost phi
    phi_ghost = np.empty((mesh.bc_type.shape[0]))
    problem.compute_ghost_phi(phi, phi_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    F_bc = flux_function(U[mesh.bc_type[:, 0]], U_ghost,
            phi[mesh.bc_type[:, 0]], phi_ghost, mesh.bc_area_normal)

    # Update cells on the left and right sides, for interior faces
    cellL_ID = mesh.edge[:, 0]
    cellR_ID = mesh.edge[:, 1]
    np.add.at(phi_new, cellL_ID, -dt / mesh.area[cellL_ID] * F)
    np.add.at(phi_new, cellR_ID,  dt / mesh.area[cellR_ID] * F)
    # Incorporate boundary faces
    cellL_ID = mesh.bc_type[:, 0]
    np.add.at(phi_new, cellL_ID, -dt / mesh.area[cellL_ID] * F_bc)

    if include_phi_source:
        # Compute velocity gradient. We have the momentum gradient, and using chain
        # rule:
        # du/dx = d(r * u)/dx * du/d(r * u) = d(r * u)/dx * (d(r * u)/du)^-1
        # = d(r * u)/dx * 1/r
        # Therefore div(u) = div(r * u) / r
        div_u = np.trace(gradU[:, 1:3], axis1=1, axis2=2) / U[:, 0]
        # Add source term
        phi_new -= phi * div_u
    return phi_new

class Upwind:
    '''
    Class for computing a fully upwind flux for the level set equation.
    '''
    name = 'upwind'

    def compute_flux(self, U_L, U_R, phi_L, phi_R, area_normal):
        n_faces = U_L.shape[0]
        # Unit normals
        length = np.linalg.norm(area_normal, axis=1, keepdims=True)
        unit_normals = area_normal / length
        # The copy here is needed, since the slice is not c-contiguous, which
        # causes the wrong data to be passed to Pybind.
        nx = unit_normals[:, 0].copy()
        ny = unit_normals[:, 1].copy()

        # Convert to primitives
        rL, rR = U_L[:, 0], U_R[:, 0]
        uL = U_L[:, 1] / rL
        uR = U_R[:, 1] / rR
        vL = U_L[:, 2] / rL
        vR = U_R[:, 2] / rR

        # Velocity vector
        vel_L = np.stack((uL, vL), axis=1)
        vel_R = np.stack((uR, vR), axis=1)
        # Check if velocity points from left to right
        vel_dot_normal_L = np.einsum('ij, ij -> i', vel_L, unit_normals)
        vel_dot_normal_R = np.einsum('ij, ij -> i', vel_R, unit_normals)
        # TODO vectorize
        # Loop
        F = np.empty(n_faces)
        # If velocity points left to right, then the left state is upwind.
        # Otherwise, the right state is upwind
        upwindL = vel_dot_normal_L >= 0
        upwindR = vel_dot_normal_L < 0
        # Compute the upwind flux in both cases
        F[upwindL] = length[upwindL, 0] * phi_L[upwindL] * vel_dot_normal_L[upwindL]
        F[upwindR] = length[upwindR, 0] * phi_R[upwindR] * vel_dot_normal_R[upwindR]
        return F

class Roe:
    name = 'roe'

    def __init__(self, g):
        self.g = g
        # Diagonalize A_RL
        self.A_RL_func, self.Lambda_func, self.Q_inv_func, self.Q_func = \
                self.get_diagonalization()

    def compute_flux(self, U_L, U_R, area_normal):
        g = self.g
        n_faces = U_L.shape[0]
        # Unit normals
        length = np.linalg.norm(area_normal, axis=1, keepdims=True)
        unit_normals = area_normal / length
        # The copy here is needed, since the slice is not c-contiguous, which
        # causes the wrong data to be passed to Pybind.
        nx = unit_normals[:, 0].copy()
        ny = unit_normals[:, 1].copy()

        # Convert to primitives
        rL, rR = U_L[:, 0], U_R[:, 0]
        uL = U_L[:, 1] / rL
        uR = U_R[:, 1] / rR
        vL = U_L[:, 2] / rL
        vR = U_R[:, 2] / rR
        hL = (U_L[:, 3] - (1/(2*g))*(g - 1)*rL*(uL**2 + vL**2)) * g / rL
        hR = (U_R[:, 3] - (1/(2*g))*(g - 1)*rR*(uR**2 + vR**2)) * g / rR

        # The RL state
        uRL = (np.sqrt(rR) * uR + np.sqrt(rL) * uL) / (np.sqrt(rR) + np.sqrt(rL))
        vRL = (np.sqrt(rR) * vR + np.sqrt(rL) * vL) / (np.sqrt(rR) + np.sqrt(rL))
        hRL = (np.sqrt(rR) * hR + np.sqrt(rL) * hL) / (np.sqrt(rR) + np.sqrt(rL))

        # Compute A_RL
        A_RL = self.A_RL_func(uRL, vRL, hRL, nx, ny, g)
        # Compute eigendecomp
        Lambda = self.Lambda_func(uRL, vRL, hRL, nx, ny, g)
        Q_inv = self.Q_inv_func(uRL, vRL, hRL, nx, ny, g)
        Q = self.Q_func(uRL, vRL, hRL, nx, ny, g)

        Lambda_m = (Lambda - np.abs(Lambda))/2
        Lambda_p = Lambda - Lambda_m
        A_RL_m = Q_inv @ Lambda_m @ Q
        A_RL_p = Q_inv @ Lambda_p @ Q
        abs_A_RL = A_RL_p - A_RL_m
        # Compute flux
        F = length * (.5*np.einsum('ijk, ik -> ij',
            convective_fluxes(U_L, g) + convective_fluxes(U_R, g), unit_normals)
            - .5 * (abs_A_RL @ ((U_R - U_L)[:, :, np.newaxis]))[:, :, 0])
        return F

    def get_diagonalization(self):
        equations_filename = 'equations_file.pkl'

        # Sympy symbols
        gamma = sp.symbols('gamma', real=True, positive=True)
        uRL = sp.symbols('u_RL', real=True)
        vRL = sp.symbols('v_RL', real=True)
        hRL = sp.symbols('h_RL', real=True, positive=True, nonzero=True)
        nx = sp.symbols('n_x', real=True)
        ny = sp.symbols('n_y', real=True)

        # Read the eigendecomposition from the saved file
        with open(equations_filename, 'rb') as equations_file:
            loaded_equations = pickle.load(equations_file)
            A_RL, Q_inv, Lambda, Q = loaded_equations

        # Function for generating C code
        def generate_code(expression, var_name):
            code = ''
            tab = '    '
            darray = 'py::array_t<double>'
            args = f'double u_RL, double v_RL, double h_RL, double n_x, double n_y, double gamma, double* {var_name}'
            # TODO fix the i*4*4
            args_i = f'u_RL_ptr[i], v_RL_ptr[i], h_RL_ptr[i], n_x_ptr[i], n_y_ptr[i], gamma, {var_name}_ptr + i*4*4'
            all_args = f'{darray} u_RL, {darray} v_RL, {darray} h_RL, {darray} n_x, {darray} n_y, double gamma'
            # Includes
            code += '#include <math.h>\n'
            code += '#include <pybind11/numpy.h>\n'
            code += '#include <pybind11/pybind11.h>\n'
            code += 'namespace py = pybind11;\n\n'

            # Function signature
            # TODO: fix the *4
            code += f'void compute_{var_name}({args}){{\n'
            for i in range(expression.shape[0]):
                for j in range(expression.shape[1]):
                    code += tab + (var_name + f'[{i} * 4 + {j}] = '
                            + sp.ccode(expression[i, j]) + ';\n')
            code += '}\n\n'

            # Function for computing across all elements
            code += f'{darray} compute_all_{var_name}({all_args}){{\n'
            # Get Pybind buffers
            code += f'py::buffer_info u_RL_buf = u_RL.request();\n'
            code += f'py::buffer_info v_RL_buf = v_RL.request();\n'
            code += f'py::buffer_info h_RL_buf = h_RL.request();\n'
            code += f'py::buffer_info n_x_buf = n_x.request();\n'
            code += f'py::buffer_info n_y_buf = n_y.request();\n'
            # Allocate the return buffer
            code += f'int n = u_RL.size();\n'
            code += f'py::array_t<double> {var_name} = py::array_t<double>(n * 4 * 4);\n'
            code += f'py::buffer_info {var_name}_buf = {var_name}.request();\n'
            # Set pointers
            code += 'double* u_RL_ptr = (double*) u_RL_buf.ptr;\n'
            code += 'double* v_RL_ptr = (double*) v_RL_buf.ptr;\n'
            code += 'double* h_RL_ptr = (double*) h_RL_buf.ptr;\n'
            code += 'double* n_x_ptr = (double*) n_x_buf.ptr;\n'
            code += 'double* n_y_ptr = (double*) n_y_buf.ptr;\n'
            code += f'double* {var_name}_ptr = (double*) {var_name}_buf.ptr;\n'
            # Compute
            code += tab + 'for (int i = 0; i < n; i++) {\n'
            code += tab + tab + f'compute_{var_name}({args_i});\n'
            code += tab + '}\n'
            # Reshape
            code += f'{var_name}.resize({{n, 4, 4}});\n'
            code += tab + f'return {var_name};\n'
            code += '}\n\n'

            # Pybind code
            code += f'PYBIND11_MODULE(compute_{var_name}, m) {{\n'
            code += tab + f'm.doc() = "Generated code"; // optional module docstring;\n'
            code += tab + f'm.def("compute_{var_name}", &compute_all_{var_name}, "A function that computes {var_name}");\n'
            code += '}'

            path = pathlib.Path("cache/") # Create Path object
            path.mkdir(exist_ok=True)
            with open(f'cache/compute_{var_name}.cpp', 'w') as f:
                f.write(code)

        generate_code(A_RL, 'A_RL')
        generate_code(Lambda, 'Lambda')
        generate_code(Q_inv, 'Q_inv')
        generate_code(Q, 'Q')

        # Get python functions
        import cache.build.compute_A_RL
        import cache.build.compute_Lambda
        import cache.build.compute_Q_inv
        import cache.build.compute_Q
        A_RL_func   = cache.build.compute_A_RL.compute_A_RL
        Lambda_func = cache.build.compute_Lambda.compute_Lambda
        Q_inv_func  = cache.build.compute_Q_inv.compute_Q_inv
        Q_func      = cache.build.compute_Q.compute_Q
        return A_RL_func, Lambda_func, Q_inv_func, Q_func

def convective_fluxes(U, g):
    # Unpack
    r =  U[:, 0]
    ru = U[:, 1]
    rv = U[:, 2]
    re = U[:, 3]
    p = (re - .5 * (ru**2 + rv**2) / r) * (g - 1)
    # Compute flux
    F = np.empty(U.shape + (2,))
    F[:, 0, 0] = ru
    F[:, 1, 0] = ru**2 / r + p
    F[:, 2, 0] = ru*rv / r
    F[:, 3, 0] = (re + p) * ru / r
    F[:, 0, 1] = rv
    F[:, 1, 1] = ru*rv / r
    F[:, 2, 1] = rv**2 / r + p
    F[:, 3, 1] = (re + p) * rv / r
    return F


if __name__ == '__main__':
    main()
