import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sympy as sp
import pathlib
import pickle
import time

def get_eigendecomposition():
    # The eigendecomposition is taken from:
    # https://www.researchgate.net/publication/264869943_Eigenvalues_and_eigenvectors_of_the_Euler_equations_in_general_geometries

    equations_filename = 'equations_file.pkl'
    generate_equations = False
    show_plots = False

    # Sympy symbols
    gamma = sp.symbols('gamma', real=True, positive=True)
    uRL = sp.symbols('u_RL', real=True)
    vRL = sp.symbols('v_RL', real=True)
    hRL = sp.symbols('h_RL', real=True, positive=True, nonzero=True)
    nx = sp.symbols('n_x', real=True)
    ny = sp.symbols('n_y', real=True)
    one = sp.sympify(1)

    # If generating the Sympy expressions
    if generate_equations:
        # Given in homework assignment
        ek = (uRL**2 + vRL**2) / 2
        vn = uRL * nx + vRL * ny
        a = sp.sqrt((gamma - 1) * (hRL - ek))
        A_RL = sp.Matrix([
            [0, nx, ny, 0],
            [(gamma - 1)*ek*nx - uRL*vn, vn - (gamma-2)*uRL*nx, uRL*ny - (gamma - 1)*vRL*nx, (gamma - 1)*nx],
            [(gamma - 1)*ek*ny - vRL*vn, vRL*nx - (gamma - 1)*uRL*ny, vn - (gamma-2)*vRL*ny, (gamma - 1)*ny],
            [((gamma - 1)*ek - hRL)*vn, hRL*nx - (gamma - 1)*uRL*vn, hRL*ny - (gamma - 1)*vRL*vn, gamma*vn],
        ])

        Lambda = sp.Matrix([
            [vn - a, 0, 0, 0],
            [0, vn, 0, 0],
            [0, 0, vn + a, 0],
            [0, 0, 0, vn],
        ])
        Q_inv = sp.Matrix([
            [1, 1, 1, 0],
            [uRL - a*nx, uRL, uRL + a*nx, ny],
            [vRL - a*ny, vRL, vRL + a*ny, -nx],
            [hRL - a*vn, ek, hRL + a*vn, uRL*ny - vRL*nx],
        ])
        Q = Q_inv.inv()

        # Write to file for later use, since this takes a
        # long time to generate
        with open(equations_filename, 'wb') as equations_file:
            eqs_to_write = A_RL, Q_inv, Lambda, Q
            pickle.dump(eqs_to_write, equations_file,
                    protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(equations_filename, 'rb') as equations_file:
            loaded_equations = pickle.load(equations_file)
            A_RL, Q_inv, Lambda, Q = loaded_equations


def generate_all_code():
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
        args = f'double u_RL, double v_RL, double h_RL, double n_x, double n_y, double gamma, double* {var_name}'
        # TODO fix the i*4*4
        args_i = f'u_RL_ptr[i], v_RL_ptr[i], h_RL_ptr[i], n_x_ptr[i], n_y_ptr[i], gamma, {var_name}_ptr + i*4*4'
        # Includes
        code += '#include <math.h>\n\n'

        # Function signature
        # TODO: fix the *4
        code += f'void compute_{var_name}({args}){{\n'
        for i in range(expression.shape[0]):
            for j in range(expression.shape[1]):
                code += tab + (var_name + f'[{i} * 4 + {j}] = '
                        + sp.ccode(expression[i, j]) + ';\n')
        code += '}\n'

        path = pathlib.Path("cache/") # Create Path object
        path.mkdir(exist_ok=True)
        with open(f'cache/compute_{var_name}.cpp', 'w') as f:
            f.write(code)

    generate_code(A_RL, 'A_RL')
    generate_code(Lambda, 'Lambda')
    generate_code(Q_inv, 'Q_inv')
    generate_code(Q, 'Q')

if __name__ == "__main__":
    get_eigendecomposition()
    generate_all_code()
