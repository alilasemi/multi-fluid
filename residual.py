import numpy as np
import pathlib
import pickle
import sympy as sp

from build.src.libpybind_bindings import (
        compute_interior_face_residual, compute_boundary_face_residual)
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
    compute_interior_face_residual(U, mesh.edge, LagrangeSegment.quad_wts,
            mesh.quad_pts_phys.flatten().data, limiter, gradU.flatten().data, mesh.xy,
            mesh.area_normals_p2.flatten().data, mesh.area, data.g, residual)

    # Compute the boundary face residual
    compute_boundary_face_residual(U, mesh.bc_type, LagrangeSegment.quad_wts,
            mesh.bc_quad_pts_phys.flatten().data, limiter, gradU.flatten().data, mesh.xy,
            mesh.bc_area_normals_p2.flatten().data, mesh.area, data.g,
            mesh.num_boundaries, problem.bc_data, problem.__class__.__name__,
            data.t, residual)

    return residual

def get_residual_phi(data, mesh, problem):
    # Unpack
    U = data.U
    gradU = data.gradU
    phi = data.phi
    # Flux function
    flux_phi = Upwind()

    residual_phi = np.zeros_like(phi)
    # Evaluate solution at faces on left and right
    U_L = U[mesh.edge[:, 0]]
    U_R = U[mesh.edge[:, 1]]
    phi_L = phi[mesh.edge[:, 0]]
    phi_R = phi[mesh.edge[:, 1]]
    # Evalute interior fluxes
    F = flux_phi.compute_flux(U_L, U_R, phi_L, phi_R, mesh.edge_area_normal)

    # Compute ghost phi
    phi_ghost = np.empty((mesh.bc_type.shape[0]))
    problem.compute_ghost_phi(phi, phi_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    # TODO: Is U_ghost really needed here? For now, setting it equal to U.
    U_ghost = U[mesh.bc_type[:, 0]]
    F_bc = flux_phi.compute_flux(U[mesh.bc_type[:, 0]], U_ghost,
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
