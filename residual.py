import numpy as np
import pathlib
import pickle
import sympy as sp
import scipy.optimize

from build.src.libpybind_bindings import (
        compute_interior_face_residual, compute_fluid_fluid_face_residual,
        compute_boundary_face_residual, compute_exact_riemann_problem,
        compute_flux, compute_flux_roe)
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
    compute_interior_face_residual(U, data.U_L, data.U_R,
            mesh.interior_face_IDs, mesh.edge, limiter, gradU.flatten().data,
            mesh.xy, mesh.area_normals_p1, mesh.area, data.fluid_ID, data.g,
            data.psg, residual)

    # Compute the boundary face residual
    compute_boundary_face_residual(U, mesh.bc_type, LagrangeSegment.quad_wts,
            mesh.bc_quad_pts_phys.flatten().data, limiter, gradU.flatten().data,
            mesh.xy, mesh.bc_area_normals_p2.flatten().data, mesh.area,
            data.fluid_ID, data.g, data.psg, mesh.num_boundaries,
            problem.bc_data, problem.__class__.__name__, data.t, residual)

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
                mesh.area_normals_p2.flatten().data, mesh.area, data.fluid_ID,
                data.g, data.psg, residual)

    return residual

def get_residual_phi(data, mesh, problem):
    # Unpack
    U = data.U
    phi = data.phi
    U_L = data.U_L
    U_R = data.U_R
    # Left and right cell IDs
    L = mesh.edge[:, 0]
    R = mesh.edge[:, 1]
    # Flux function
    flux_phi = Upwind()

    residual_phi = np.zeros_like(phi)
    # Evaluate phi at faces on left and right: first order component
    phi_L = phi[L]
    phi_R = phi[R]
    # Second order component
    quad_pts = .5 * (mesh.xy[L] + mesh.xy[R])
    phi_L += np.einsum('ik, ik -> i', data.grad_phi[L], quad_pts - mesh.xy[L])
    phi_R += np.einsum('ik, ik -> i', data.grad_phi[R], quad_pts - mesh.xy[R])

    # Evalute interior fluxes
    F = flux_phi.compute_flux(U_L, U_R, phi_L, phi_R,
            mesh.area_normals_p1)

    # Compute ghost phi
    phi_ghost = np.empty((mesh.bc_type.shape[0]))
    problem.compute_ghost_phi(phi, phi_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    # TODO: Is U_ghost really needed here? For now, setting it equal to U.
    U_ghost = U[mesh.bc_type[:, 0]]
    F_bc = flux_phi.compute_flux(U[mesh.bc_type[:, 0]], U_ghost,
            phi[mesh.bc_type[:, 0]], phi_ghost, mesh.bc_area_normal)

    # Update cells on the left and right sides, for interior faces
    np.add.at(residual_phi, L, -1 / mesh.area[L] * F)
    np.add.at(residual_phi, R,  1 / mesh.area[R] * F)
    # Incorporate boundary faces
    L = mesh.bc_type[:, 0]
    np.add.at(residual_phi, L, -1 / mesh.area[L] * F_bc)
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
        # If velocity points left to right, then the left state is upwind.
        # Otherwise, the right state is upwind
        upwindL = vel_dot_normal_L >= 0
        upwindR = vel_dot_normal_L < 0
        # Compute the upwind flux in both cases
        F = np.empty(n_faces)
        F[upwindL] = length[upwindL, 0] * phi_L[upwindL] * vel_dot_normal_L[upwindL]
        F[upwindR] = length[upwindR, 0] * phi_R[upwindR] * vel_dot_normal_R[upwindR]
        return F


if __name__ == '__main__':
    # These tests come from Toro's Riemann solvers book
    # TODO: Add as unit tests
    # TODO: These will only test psg = 0
    output = np.empty(5)
#    g = 1.4
#    psg = 0
#    # Test 1
#    rtol = 1e-4
#    rL, uL, pL, rR, uR, pR = 1, 0, 1, .125, 0, .1
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, g, g, psg, psg, output)
#    test1 = np.isclose([.30313, .92745, .42632, .26557], output, rtol=rtol)
#    print(f'Test 1: {test1}')
#    # Test 2
#    rtol = 1e-2
#    rL, uL, pL, rR, uR, pR = 1, -2, .4, 1, 2, .4
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, g, g, psg, psg, output)
#    test2 = np.isclose([.00189, 0, .02185, .02185], output, rtol=rtol)
#    print(f'Test 2: {test2}')
#    # Test 3
#    rtol = 1e-5
#    rL, uL, pL, rR, uR, pR = 1, 0, 1000, 1, 0, .01
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, g, g, psg, psg, output)
#    test3 = np.isclose([460.894, 19.5975, .57506, 5.99924], output, rtol=rtol)
#    print(f'Test 3: {test3}')
#    # Test 4
#    rtol = 1e-5
#    rL, uL, pL, rR, uR, pR = 1, 0, .01, 1, 0, 100
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, g, g, psg, psg, output)
#    test4 = np.isclose([46.0950, -6.19633, 5.99242, .57511], output, rtol=rtol)
#    print(f'Test 4: {test4}')
#    # Test 5
#    rtol = 1e-5
#    rL, uL, pL, rR, uR, pR = 5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, g, g, psg, psg, output)
#    test5 = np.isclose([1691.64, 8.68975, 14.2823, 31.0426], output, rtol=rtol)
#    print(f'Test 5: {test5}')

    # Testing that the exact flux gives about the same thing as Roe
    #area_normal = np.array([np.cos(.8), np.sin(.8)]).reshape(2, 1)
    area_normal = np.array([np.sqrt(2)/2, np.sqrt(2)/2]).reshape(2, 1)
    #area_normal = np.array([1, 0]).reshape(2, 1)
    #area_normal = np.array([0, 1]).reshape(2, 1)
    gL = 1.4
    gR = 1.4
    psgL = 0
    psgR = 0
    F = np.empty(4)
    F_roe = np.empty(4)
    vL = 2000
    vR = 0
    U_LR = [(np.array([1, vL, 0, 2.5e5 + .5*vL**2]), np.array([1, vR, 0, 2.5e5 +
            .5*vR**2])),
            ]
    for U_L, U_R in U_LR:
        compute_flux(U_L, U_R, area_normal, gL, gR, psgL, psgR, F)
        compute_flux_roe(U_L, U_R, area_normal, gL, F_roe)
        print(F - F_roe)
