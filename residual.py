import numpy as np
import pathlib
import pickle
import sympy as sp
import scipy.optimize

from build.src.libpybind_bindings import (
        compute_interior_face_residual, compute_fluid_fluid_face_residual,
        compute_boundary_face_residual)
from lagrange import LagrangeSegment


def get_residual(data, mesh, problem):
    # Unpack
    U = data.U
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
    # TODO: Passing 3D numpy arrays is kinda ugly right now...
    compute_interior_face_residual(U, mesh.interior_face_IDs, mesh.edge,
            limiter, gradU.flatten().data, mesh.xy,
            mesh.area_normals_p1, mesh.area, data.g, residual)

    # Compute the boundary face residual
    compute_boundary_face_residual(U, mesh.bc_type, LagrangeSegment.quad_wts,
            mesh.bc_quad_pts_phys.flatten().data, limiter, gradU.flatten().data, mesh.xy,
            mesh.bc_area_normals_p2.flatten().data, mesh.area, data.g,
            mesh.num_boundaries, problem.bc_data, problem.__class__.__name__,
            data.t, residual)

    # Compute the residual from interfaces
    if problem.fluid_solid:
        # Compute the fluid-solid interface residual
        #TODO
        pass
    else:
        # Compute the fluid-fluid interface residual
        compute_fluid_fluid_face_residual(U, mesh.interface_IDs, mesh.edge,
                LagrangeSegment.quad_wts, mesh.quad_pts_phys.flatten().data,
                limiter, gradU.flatten().data, mesh.xy,
                mesh.area_normals_p2.flatten().data, mesh.area, data.g, data.dt,
                residual)

    return residual

def get_residual_phi(data, mesh, problem):
    # Unpack
    U = data.U
    phi = data.phi
    # Flux function
    flux_phi = Upwind()

    residual_phi = np.zeros_like(phi)
    # Get interior faces only, ignoring "deactivated" interfaces
    interior_face_IDs = mesh.edge[:, 0] != -1
    edge_interior = mesh.edge[interior_face_IDs]

    # Evaluate solution at faces on left and right: first order component
    U_L = U[edge_interior[:, 0]]
    U_R = U[edge_interior[:, 1]]
    phi_L = phi[edge_interior[:, 0]]
    phi_R = phi[edge_interior[:, 1]]

    # Evalute interior fluxes
    F = flux_phi.compute_flux(U_L, U_R, phi_L, phi_R,
            mesh.area_normals_p1[interior_face_IDs])

    # Compute ghost phi
    phi_ghost = np.empty((mesh.bc_type.shape[0]))
    problem.compute_ghost_phi(phi, phi_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    # TODO: Is U_ghost really needed here? For now, setting it equal to U.
    U_ghost = U[mesh.bc_type[:, 0]]
    F_bc = flux_phi.compute_flux(U[mesh.bc_type[:, 0]], U_ghost,
            phi[mesh.bc_type[:, 0]], phi_ghost, mesh.bc_area_normal)

    # Update cells on the left and right sides, for interior faces
    cellL_ID = edge_interior[:, 0]
    cellR_ID = edge_interior[:, 1]
    np.add.at(residual_phi, cellL_ID, -1 / mesh.area[cellL_ID] * F)
    np.add.at(residual_phi, cellR_ID,  1 / mesh.area[cellR_ID] * F)
    # Incorporate boundary faces
    cellL_ID = mesh.bc_type[:, 0]
    np.add.at(residual_phi, cellL_ID, -1 / mesh.area[cellL_ID] * F_bc)
    return residual_phi

def compute_gradient_phi(phi, mesh):
    '''
    Warning: This function is completely untested so far!
    '''
    grad_phi = np.empty((mesh.n, 2))
    # Loop over all cells
    for i in range(mesh.n):
        n_points = len(mesh.neighbors[i])
        # If there are no other points in the stencil, then set the gradient to
        # zero
        if n_points == 1:
            grad_phi[i] = 0
        # Otherwise, solve with least squares
        else:
            # Construct A matrix: [x_i, y_i, 1]
            A = np.ones((n_points, 3))
            A[:, :-1] = mesh.xy[mesh.stencil[i]]
            # We desired [x_i, y_i, 1] @ [c0, c1, c2] = U[i], therefore Ax=b.
            # However, there are more equations than unknowns (for most points)
            # so instead, solve the normal equations: A.T @ A x = A.T @ b
            try:
                c = np.linalg.solve(A.T @ A, A.T @ phi[mesh.stencil[i]])
                # Since U = c0 x + c1 y + c2, then dU/dx = c0 and dU/dy = c1.
                grad_phi[i] = c[:-1].T
            except:
                print(f'phi gradient calculation failed! Stencil = {mesh.stencil[i]}')
                grad_phi[i] = 0
    return grad_phi


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
        # If velocity points left to right, then the left state is upwind.
        # Otherwise, the right state is upwind
        upwindL = vel_dot_normal_L >= 0
        upwindR = vel_dot_normal_L < 0
        # Compute the upwind flux in both cases
        F = np.empty(n_faces)
        F[upwindL] = length[upwindL, 0] * phi_L[upwindL] * vel_dot_normal_L[upwindL]
        F[upwindR] = length[upwindR, 0] * phi_R[upwindR] * vel_dot_normal_R[upwindR]
        return F


def exact_riemann_problem(
        r4, p4, u4, r1, p1, u1, g, xL, xR, t):
    # Points at which to evaluate Riemann problem
    x = np.array([xL, xR])

    # Compute speed of sound
    def compute_c(g, p, r): return np.sqrt(g * p / r)
    c1 = compute_c(g, p1, r1)
    c4 = compute_c(g, p4, r4)

    # Compute the pressure ratio
    def p_rhs(p2p1, u1, u4, c1, c4, p1, p4, g):
        return p2p1 * (
            1 + (g - 1) / (2 * c4) * (
                u4 - u1 - (c1/g) * (
                    (p2p1 - 1) /
                    np.sqrt(((g+1) / (2 * g)) * (p2p1 - 1) + 1)
                )
            )
        )**(-(2 * g) / (g - 1)) - p4/p1
    p2p1 = scipy.optimize.fsolve(p_rhs, p4/p1, args=(
            u1, u4, c1, c4, p1, p4, g))[0]
    p2 = p2p1 * p1

    # Compute u2
    u2 = u1 + (c1 / g) * (p2p1-1) / (np.sqrt( ((g+1)/(2*g)) * (p2p1-1) + 1))
    # Compute V
    V = u1 + c1 * np.sqrt( ((g+1)/(2*g)) * (p2p1-1) + 1)
    # Compute c2
    c2 = c1 * np.sqrt(
            p2p1 * (
                (((g+1)/(g-1)) + p2p1
                ) / (
                1 + ((g+1)/(g-1)) * p2p1)
                )
    )
    # Compute r2
    def compute_r(g, p, c): return g * p / (c**2)
    r2 = compute_r(g, p2, c2)

    # p and u same across contact
    u3 = u2
    p3 = p2
    # Compute c3
    c3 = .5 * (g - 1) * (u4 + ((2*c4)/(g-1)) - u3)
    # Compute r3
    r3 = compute_r(g, p3, c3)

    # Flow inside expansion
    u_exp = (2/(g+1)) * (x/t + ((g-1)/2) * u4 + c4)
    c_exp = (2/(g+1)) * (x/t + ((g-1)/2) * u4 + c4) - x/t
    # Clip the speed of sound to be positive. This is not entirely necessary
    # (the spurious negative speed of sound is only outside the expansion,
    # so in the expansion everything is okay) but not doing this makes Numpy
    # give warnings when computing pressure.
    c_exp[c_exp < 0] = 1e-16
    p_exp = p4 * (c_exp/c4)**(2*g/(g-1))
    r_exp = compute_r(g, p_exp, c_exp)

    # Figure out which flow region each point is in
    r = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)
    for i in range(x.size):
        xt = x[i] / t
        # Left of expansion
        if xt < (u4 - c4):
            r[i] = r4
            u[i] = u4
            p[i] = p4
        # Inside expansion
        elif xt < (u3 - c3):
            r[i] = r_exp[i]
            u[i] = u_exp[i]
            p[i] = p_exp[i]
        # Right of expansion
        elif xt < u3:
            r[i] = r3
            u[i] = u3
            p[i] = p3
        # Left of shock
        elif xt < V:
            r[i] = r2
            u[i] = u2
            p[i] = p2
        # Right of shock
        elif xt > V:
            r[i] = r1
            u[i] = u1
            p[i] = p1
    return r, u, p


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
