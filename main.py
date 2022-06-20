import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap, ufuncify
import pickle

from exact_solution import exact_solution


# Inputs
r4 = 1    # left
p4 = 1e5  # left
u4 = 100  # left
v4 = 0    # left
r1 = .125 # right
p1 = 1e4  # right
u1 = 50   # right
v1 = 0    # right
g = 1.4

def main():
    #exact_solution(
    #        r4, p4, u4, v4, r1, p1, u1, v1, g)
    compute_solution(Roe())

class Mesh:
    # Domain
    xL = -10
    xR = 10
    yL = -1
    yR = 1

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        # Number of faces (formula given in assignment)
        self.n_faces = int( (nx - 1)*ny + (ny - 1)*nx + (nx - 1)*(ny - 1) )
        # Grid spacing
        dx = (self.xR - self.xL) / (nx - 1)
        dy = (self.yR - self.yL) / (ny - 1)
        # Compute nodes
        x = np.linspace(self.xL, self.xR, nx)
        y = np.linspace(self.yL, self.yR, ny)
        grid = np.meshgrid(x, y)
        self.xy = np.empty((self.n, 2))
        self.xy[:, 0] = grid[0].flatten()
        self.xy[:, 1] = grid[1].flatten()
        # Compute areas. There are three types of cells:
        # 1. Interior hexagons
        # 2. Boundary quads
        # 3. Corner quads
        triangle_area = (.5 * np.sqrt( (4/3 * dx)**2 + (2/3 * dy)**2 )
                * np.sqrt( (1/3 * dx)**2 + (1/3 * dy)**2 ))
        small_triangle_area = .5 * (dx/2) * (dy/2)
        self.area = 2*triangle_area * np.ones(nx * ny)
        # Find boundaries and set their area
        boundary_area = triangle_area
        idx_xR = np.where(np.isclose(self.xy[:, 0], self.xR))[0]
        idx_xL = np.where(np.isclose(self.xy[:, 0], self.xL))[0]
        idx_yR = np.where(np.isclose(self.xy[:, 1], self.yR))[0]
        idx_yL = np.where(np.isclose(self.xy[:, 1], self.yL))[0]
        self.area[idx_xR] = boundary_area
        self.area[idx_xL] = boundary_area
        self.area[idx_yR] = boundary_area
        self.area[idx_yL] = boundary_area
        # Find corners and set their area
        self.area[np.intersect1d(idx_xR, idx_yR)] = triangle_area - small_triangle_area
        self.area[np.intersect1d(idx_xR, idx_yL)] = small_triangle_area
        self.area[np.intersect1d(idx_xL, idx_yR)] = small_triangle_area
        self.area[np.intersect1d(idx_xL, idx_yL)] = triangle_area - small_triangle_area
        # -- Faces -- #
        self.edge = np.empty((self.n_faces, 2), dtype=int)
        self.edge_area_normal = np.empty((self.n_faces, 2))
        self.bc_type = np.empty((2*nx + 2*ny, 2), dtype=int)
        self.bc_area_normal = np.empty((2*nx + 2*ny, 2))
        face_ID = 0
        BC_ID = 0
        # Loop over indices
        rotation90 = np.array([[0, -1], [1, 0]])
        for i in range(nx):
            for j in range(ny):
                # Get unstructured index
                cell_ID = j * nx + i

                # Make face above
                if (j < ny - 1):
                    above_ID = (j + 1)*nx + i
                    self.edge[face_ID] = [cell_ID, above_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([2*dx/3, dy/3])
                    # If this is a left/right boundary, cut it in half
                    if i == 0 or i == nx - 1: self.edge_area_normal[face_ID] /= 2
                    face_ID += 1

                # Make face to the right
                if (i < nx - 1):
                    right_ID = j*nx + i + 1
                    self.edge[face_ID] = [cell_ID, right_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([-dx/3, -2*dy/3])
                    # If this is a top/bottom boundary, cut it in half
                    if j == 0 or j == ny - 1: self.edge_area_normal[face_ID] /= 2
                    face_ID += 1

                # Make face diagonally above and to the right
                if (i < nx - 1 and j < ny - 1):
                    diag_ID = (j + 1)*nx + i + 1
                    self.edge[face_ID] = [cell_ID, diag_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([dx/3, -dy/3])
                    face_ID += 1

                # If it's a left/right BC
                if i == 0 or i == nx - 1:
                    self.bc_type[BC_ID] = [cell_ID, 2]
                    self.bc_area_normal[BC_ID] = [dy, 0]
                    # Inflow is negative
                    if i == 0:
                        self.bc_area_normal[BC_ID] *= -1
                    # If it's a corner, cut the area in half
                    if j == 0 or j == ny - 1:
                        self.bc_area_normal[BC_ID] /= 2
                    BC_ID += 1
                # If it's a bottom/top BC
                if j == 0 or j == ny - 1:
                    self.bc_type[BC_ID] = [cell_ID, 1]
                    self.bc_area_normal[BC_ID] = [0, dx]
                    # Bottom wall is negative
                    if j == 0:
                        self.bc_area_normal[BC_ID] *= -1
                    # If it's a corner, cut the area in half
                    if i == 0 or i == nx - 1:
                        self.bc_area_normal[BC_ID] /= 2
                    BC_ID += 1


def compute_solution(flux):
    # Get initial conditions as conservatives
    W4 = primitive_to_conservative(r4, u4, v4, p4, g)
    W1 = primitive_to_conservative(r1, u1, v1, p1, g)

    # Solver inputs
    n_t = 400
    t_final = .01
    dt = t_final / n_t
    t_list = [.004, .008]#[.002, .004, .006, .008]

    nx = 100
    ny = 20

    # Create mesh
    mesh = Mesh(nx, ny)
    # Initial solution
    U = np.empty((mesh.n, 4))
    U[mesh.xy[:, 0] <= 0] = W4
    U[mesh.xy[:, 0] > 0] = W1

    # Loop over time
    U_list = []
    r_probe = np.empty((n_t, 4))
    p_probe = np.empty((n_t, 4))
    x_shock = np.empty(n_t)
    idx_A = 4*nx  + (nx // 2) - 1
    idx_B = 4*nx  + (nx // 2)
    idx_C = 6*nx  + (nx // 2)
    idx_D = 11*nx + (nx // 2)
    probe_list = [idx_A, idx_B, idx_C, idx_D]
    for i in range(n_t):
        print(i)
        # Update solution
        U = update(U, dt, mesh, flux.compute_flux)
        t = (i + 1) * dt
        if t in t_list:
            U_list.append(U)
        # Store probe values
        for j in range(4):
            idx = probe_list[j]
            V = conservative_to_primitive(
                    U[idx, 0], U[idx, 1], U[idx, 2], U[idx, 3], g)
            r_probe[i, j] = V[0]
            p_probe[i, j] = V[3]
        # Find shock
        for j in range(nx):
            # Jump in x-velocity
            delta_u = U[6*nx + nx - 1 - j, 1] / U[6*nx + nx - 1 - j, 0] - u4
            if delta_u > .01 * u4:
                x_shock[i] = mesh.xy[nx - 1 - j, 0]
                break

    # Fit a line to the shock location
    fit_shock = np.polyfit(np.linspace(dt, t_final, n_t), x_shock, 1)
    shock_speed = fit_shock[0]
    print(f'The shock speed is {shock_speed} m/s.')

    # Final primitives
    V_list = []
    for U in U_list:
        V = np.empty_like(U)
        for i in range(mesh.n):
            V[i] = conservative_to_primitive(
                    U[i, 0], U[i, 1], U[i, 2], U[i, 3], g)
        V_list.append(V)

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Density, velocity, and pressure profiles
    fig, axes = plt.subplots(len(t_list), 3, figsize=(6.5, 8))
    for i in range(len(t_list)):
        V = V_list[i]
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
        j = 10
        # Plotting rho, u, and p
        f = [r, u, p]
        f_exact = [r_exact, u_exact, p_exact]
        time = f'(t={t} \\textrm{{ s}})'
        ylabels = [f'$\\rho{time}$ (kg/m$^3$)', f'$u{time}$ (m/s)', f'$p{time}$ (N/m$^2$)']
        # Loop over rho, u, p
        for idx in range(3):
            ax = axes[i, idx]
            ax.plot(mesh.xy[j*nx:(j+1)*nx, 0], f[idx][j*nx:(j+1)*nx], 'k', linewidth=2)
            ax.plot(x_exact, f_exact[idx], '--k', linewidth=1)
            ax.set_ylabel(ylabels[idx], fontsize=10)
            ax.tick_params(labelsize=10)
            ax.grid(linestyle='--')
    for idx in range(3):
        axes[-1, idx].set_xlabel('x (m)', fontsize=10)
    # Save
    plt.tight_layout()
    plt.savefig(f'result_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')

    # Probe values
    fig, axes = plt.subplots(len(probe_list), 2, figsize=(6.5, 8))
    probe_name_list = ['A', 'B', 'C', 'D']
    for i in range(len(probe_list)):
        probe_name = probe_name_list[i]
        # Plotting rho and p
        f = [r_probe[:, i], p_probe[:, i]]
        ylabels = [f'$\\rho_{probe_name}$ (kg/m$^3$)', f'$p_{probe_name}$ (N/m$^2$)']
        # Loop over rho and p
        for idx in range(2):
            ax = axes[i, idx]
            ax.plot(np.linspace(dt, t_final, n_t), f[idx], 'k', linewidth=2)
            ax.set_ylabel(ylabels[idx], fontsize=10)
            ax.tick_params(labelsize=10)
            ax.grid(linestyle='--')
    for idx in range(2):
        axes[-1, idx].set_xlabel('t (s)', fontsize=10)
    # Save
    plt.tight_layout()
    plt.savefig(f'probe_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')

    # Density, velocity, and pressure contour plots
    fig, axes = plt.subplots(len(t_list), 3, figsize=(6.5, 8))
    for i in range(len(t_list)):
        V = V_list[i]
        t = t_list[i]

        r = V[:, 0]
        u = V[:, 1]
        v = V[:, 2]
        p = V[:, 3]

        # Plotting rho, u, and p
        f = [r, u, p]
        f_exact = [r_exact, u_exact, p_exact]
        time = f'(t={t} \\textrm{{ s}})'
        ylabels = [f'$\\rho{time}$ (kg/m$^3$)', f'$u{time}$ (m/s)', f'$p{time}$ (N/m$^2$)']
        # Loop over rho, u, p
        for idx in range(3):
            ax = axes[i, idx]
            contourf = ax.tricontourf(mesh.xy[:, 0], mesh.xy[:, 1], f[idx])
            plt.colorbar(mappable=contourf, ax=ax)
            ax.set_title(ylabels[idx], fontsize=10)
            ax.tick_params(labelsize=10)
    for idx in range(3):
        axes[-1, idx].set_xlabel('x (m)', fontsize=10)
    for idx in range(len(t_list)):
        axes[idx, 0].set_ylabel('y (m)', fontsize=10)
    # Save
    plt.tight_layout()
    plt.savefig(f'contour_{mesh.nx}x{mesh.ny}.pdf', bbox_inches='tight')

    print('Plots written to file.')

def update(U, dt, mesh, flux_function):
    U_new = U.copy()
    # Evaluate solution at faces on left and right
    U_L = U[mesh.edge[:, 0]]
    U_R = U[mesh.edge[:, 1]]
    # Evalute interior fluxes
    F = flux_function(U_L, U_R, mesh)

    # Compute boundary fluxes
    U_ghost = np.empty((mesh.bc_type.shape[0], 4))
    for i in range(mesh.bc_type.shape[0]):
        cell_ID, bc = mesh.bc_type[i]
        # Get primitives
        V = conservative_to_primitive(*U[cell_ID], g)
        # Compute wall ghost state
        if bc == 1:
            # The density and pressure are kept the same in the ghost state
            r = V[0]
            p = V[3]
            # The x-direction velocity is not changed since the wall is
            # horizontal
            u = V[1]
            # The y-direction velocity is flipped in sign, since the wall is
            # horizontal
            v = -V[2]
        # Compute farfield ghost state
        if bc == 2:
            #TODO
            # If it's the inflow
            if mesh.bc_area_normal[i, 0] < 0:
                # Set state to the left state
                r = r4
                p = p4
                u = u4
                v = v4
            # If it's the outflow
            if mesh.bc_area_normal[i, 0] > 0:
                # Set state to the right state
                p = p1
                r = r1
                u = u1
                v = v1
        # Compute ghost state
        U_ghost[i] = primitive_to_conservative(r, u, v, p, g)
    # Evalute boundary fluxes
    F_bc = flux_function(U[mesh.bc_type[:, 0]], U_ghost, mesh)

    # Update cells by looping over faces and updating left and right sides
    for i in range(mesh.n_faces):
        cellL_ID = mesh.edge[i, 0]
        cellR_ID = mesh.edge[i, 1]
        U_new[cellL_ID] -= dt / mesh.area[cellL_ID] * F[i]
        U_new[cellR_ID] += dt / mesh.area[cellR_ID] * F[i]
    for i in range(mesh.bc_type.shape[0]):
        cellL_ID = mesh.bc_type[i, 0]
        U_new[cellL_ID] -= dt / mesh.area[cellL_ID] * F_bc[i]

    return U_new

class Roe:
    name = 'roe'

    def __init__(self):
        # Diagonalize A_RL
        self.A_RL_func, self.Lambda_func, self.Q_inv_func, self.Q_func = \
                self.get_diagonalization()

    def compute_flux(self, U_L, U_R, mesh):
        n_faces = U_L.shape[0]
        # Unit normals
        if n_faces == mesh.n_faces:
            edge_area_normal = mesh.edge_area_normal
        else:
            edge_area_normal = mesh.bc_area_normal
        length = np.linalg.norm(edge_area_normal, axis=1, keepdims=True)
        unit_normals = edge_area_normal / length
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
            convective_fluxes(U_L) + convective_fluxes(U_R), unit_normals)
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

            with open(f'cache/compute_{var_name}.cpp', 'w') as f:
                f.write(code)

        generate_code(A_RL, 'A_RL')
        #generate_code(Lambda, 'Lambda')
        generate_code(Q_inv, 'Q_inv')
        generate_code(Q, 'Q')

        # Get python functions
        import cache.compute_A_RL
        import cache.compute_Lambda
        import cache.compute_Q_inv
        import cache.compute_Q
        A_RL_func   = cache.compute_A_RL.compute_A_RL
        Lambda_func = cache.compute_Lambda.compute_Lambda
        Q_inv_func  = cache.compute_Q_inv.compute_Q_inv
        Q_func      = cache.compute_Q.compute_Q
        return A_RL_func, Lambda_func, Q_inv_func, Q_func

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

def convective_flux(U):
    # Unpack
    r = U[0]
    ru = U[1]
    rv = U[2]
    re = U[3]
    p = (re - .5 * (ru**2 + rv**2) / r) * (g - 1)
    # Compute flux
    F = np.empty(U.shape + (2,))
    F[0, 0] = ru
    F[1, 0] = ru**2 / r + p
    F[2, 0] = ru*rv / r
    F[3, 0] = (re + p) * ru / r
    F[0, 1] = rv
    F[1, 1] = ru*rv / r
    F[2, 1] = rv**2 / r + p
    F[3, 1] = (re + p) * rv / r
    return F

def convective_fluxes(U):
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
