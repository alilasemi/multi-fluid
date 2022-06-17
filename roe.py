import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sympy as sp
import pickle
import time

# The eigendecomposition is taken from:
# https://www.researchgate.net/publication/264869943_Eigenvalues_and_eigenvectors_of_the_Euler_equations_in_general_geometries

equations_filename = 'equations_file.pkl'
generate_equations = True
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
    breakpoint()

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

# Clear variables
del vRL, hRL, gamma, one
