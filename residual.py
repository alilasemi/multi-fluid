import numpy as np
import pathlib
import pickle
import sympy as sp

from cache.build.interior_face_residual import (
        compute_interior_face_residual, compute_boundary_face_residual)
from lagrange import LagrangeSegment


def get_residual(data, mesh, problem):
    # Unpack
    U = data.U
    U_ghost = data.U_ghost
    gradU = data.gradU

    residual = np.zeros_like(U)

    # Compute the limiter value
    # This uses the multidimensional Barth-Jesperson limiter from:
    # https://arc.aiaa.org/doi/pdf/10.2514/6.1989-366
    # TODO Why is this damping needed?
    damping = .7
    limiter = np.empty((mesh.n, 4))
    # Loop state vars
    for k in range(4):
        # Compute extrapolated value
        # In the paper, this is the value of u_i - u_A
        U_face_diff = np.einsum('id, ijd -> ij', gradU[:, k],
                mesh.cell_point_coords - mesh.xy.reshape((mesh.n, 1, 2)))
        u_A_min = np.min(U[mesh.limiter_stencil, k], axis=1)
        u_A_max = np.max(U[mesh.limiter_stencil, k], axis=1)
        # Limiter value for all face points of cell i
        limiter_j = np.empty((mesh.n, mesh.max_num_face_points))
        # Condition 1
        index = np.nonzero(U_face_diff > 0)
        limiter_j[index] = (u_A_max[index[0]] - U[index[0], k]) / U_face_diff[index]
        # Condition 2
        index = np.nonzero(U_face_diff < 0)
        limiter_j[index] = (u_A_min[index[0]] - U[index[0], k]) / U_face_diff[index]
        # Condition 3
        index = np.nonzero(U_face_diff == 0)
        limiter_j[index] = 1
        # Take the minimum across each face point
        limiter[:, k] = damping * np.min(limiter_j, axis=1)

    # Compute the interior face residual
    # TODO: is Pybind OOP a thing? Seems to not be...
    # TODO: Ditch the whole area_normals_p2 vs regular normals thing (actually
    # p1 wouldn't even work)
    compute_interior_face_residual(U, mesh.edge, LagrangeSegment.quad_wts,
            mesh.quad_pts_phys, limiter, gradU, mesh.xy, mesh.area_normals_p2,
            mesh.area, data.flux.g, residual)

    # Compute the boundary face residual
    compute_boundary_face_residual(U, mesh.bc_type, LagrangeSegment.quad_wts,
            mesh.bc_quad_pts_phys, limiter, gradU, mesh.xy, mesh.bc_area_normals_p2,
            mesh.area, data.flux.g, mesh.num_boundaries, problem.bc_data,
            problem.__class__.__name__, residual)

    return residual

def get_residual_phi(data, mesh, problem):
    # Unpack
    U = data.U
    U_ghost = data.U_ghost
    gradU = data.gradU
    phi = data.phi

    residual_phi = np.zeros_like(phi)
    # Evaluate solution at faces on left and right
    U_L = U[mesh.edge[:, 0]]
    U_R = U[mesh.edge[:, 1]]
    phi_L = phi[mesh.edge[:, 0]]
    phi_R = phi[mesh.edge[:, 1]]
    # Evalute interior fluxes
    F = data.flux_phi.compute_flux(U_L, U_R, phi_L, phi_R, mesh.edge_area_normal)

    # Compute ghost phi
    phi_ghost = np.empty((mesh.bc_type.shape[0]))
    problem.compute_ghost_phi(phi, phi_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    F_bc = data.flux_phi.compute_flux(U[mesh.bc_type[:, 0]], U_ghost,
            phi[mesh.bc_type[:, 0]], phi_ghost, mesh.bc_area_normal)

    # Update cells on the left and right sides, for interior faces
    cellL_ID = mesh.edge[:, 0]
    cellR_ID = mesh.edge[:, 1]
    np.add.at(residual_phi, cellL_ID, -1 / mesh.area[cellL_ID] * F)
    np.add.at(residual_phi, cellR_ID,  1 / mesh.area[cellR_ID] * F)
    # Incorporate boundary faces
    cellL_ID = mesh.bc_type[:, 0]
    np.add.at(residual_phi, cellL_ID, -1 / mesh.area[cellL_ID] * F_bc)
    return residual_phi

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

    def compute_flux(self, U_L, U_R, area_normal, cpp=None):
        if cpp is None:
            import cache.build.roe
            return cache.build.roe.compute_flux(U_L[0].copy(), U_R[0].copy(), area_normal, self.g)
        area_normal = area_normal.reshape((1, -1))

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
