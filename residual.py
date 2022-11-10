import numpy as np
import pathlib
import pickle
import sympy as sp
import scipy.optimize

from build.src.libpybind_bindings import (
        compute_interior_face_residual, compute_fluid_fluid_face_residual,
        compute_boundary_face_residual,
        evaluate_solution_at_interior_faces, evaluate_solution_at_interfaces,
        compute_exact_riemann_problem, compute_flux, compute_flux_roe,
        compute_gradient)
from problem import conservative_to_primitive, primitive_to_conservative
from lagrange import LagrangeSegment


def compute_limiter(data, mesh):
    """Compute the limiter value for each cell.

    This uses the multidimensional Barth-Jesperson limiter from:
    https://arc.aiaa.org/doi/pdf/10.2514/6.1989-366
    """
    # TODO Why is this damping needed?
    damping = .7
    limiter = np.empty((mesh.n, 4))
    # Compute primitives
    V = np.empty_like(data.U)
    for i in range(mesh.n):
        # Get fluid data
        g_i = data.g[data.fluid_ID[i]]
        psg_i = data.psg[data.fluid_ID[i]]
        V[i] = conservative_to_primitive(*data.U[i], g_i, psg_i)
    # Loop state vars
    for k in range(4):
        # Compute extrapolated value
        # In the paper, this is the value of u_i - u_A
        V_face_diff = np.einsum('id, ijd -> ij', data.gradV[:, k],
                mesh.cell_point_coords - mesh.xy.reshape((mesh.n, 1, 2)))
        u_A_min = np.min(V[mesh.limiter_stencil, k], axis=1)
        u_A_max = np.max(V[mesh.limiter_stencil, k], axis=1)
        # Limiter value for all face points of cell i
        limiter_j = np.empty((mesh.n, mesh.max_num_face_points))
        # Condition 1
        index = np.nonzero(V_face_diff > 0)
        limiter_j[index] = (u_A_max[index[0]] - V[index[0], k]) / V_face_diff[index]
        # Condition 2
        index = np.nonzero(V_face_diff < 0)
        limiter_j[index] = (u_A_min[index[0]] - V[index[0], k]) / V_face_diff[index]
        # Condition 3
        index = np.nonzero(V_face_diff == 0)
        limiter_j[index] = 1
        # Take the minimum across each face point
        limiter[:, k] = damping * np.min(limiter_j, axis=1)
    return limiter


def get_residual(data, mesh, problem):
    # Unpack
    U = data.U
    gradV = data.gradV

    # Recompute gradient of solution
    compute_gradient(data.U, mesh.xy, mesh.stencil,
            data.gradV.reshape(-1), data.g, data.psg, data.fluid_ID)

    residual = np.zeros_like(U)

    # Compute limiter
    limiter = compute_limiter(data, mesh)

    # Compute the interior face residual
    # TODO: is Pybind OOP a thing? Seems to not be...
    # TODO: Ditch the whole area_normals_p2 vs regular normals thing (actually
    # p1 wouldn't even work)
    # TODO: Passing 3D numpy arrays is kinda ugly right now...
    compute_interior_face_residual(U, data.U_L_p1, data.U_R_p1,
            mesh.interior_face_IDs, mesh.edge, limiter, gradV.flatten().data,
            mesh.xy, mesh.area_normals_p1, mesh.area, data.fluid_ID, data.g,
            data.psg, residual)

    # Compute the boundary face residual
    compute_boundary_face_residual(U, mesh.bc_type, LagrangeSegment.quad_wts,
            mesh.bc_quad_pts_phys.flatten().data, limiter, gradV.flatten().data,
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
        U_L_p2 = np.empty_like(data.U_L_p2).flatten()
        U_R_p2 = np.empty_like(data.U_R_p2).flatten()
        compute_fluid_fluid_face_residual(U, U_L_p2,
                U_R_p2, mesh.interface_IDs, mesh.edge,
                LagrangeSegment.quad_wts, mesh.quad_pts_phys.flatten().data,
                limiter, gradV.flatten().data, mesh.xy,
                mesh.area_normals_p2.flatten().data, mesh.area, data.fluid_ID,
                data.g, data.psg, residual)
        # TODO: Look into cleaner ways to pass multidimensional arrays back
        data.U_L_p2 = U_L_p2.reshape(data.U_L_p2.shape)
        data.U_R_p2 = U_R_p2.reshape(data.U_R_p2.shape)

    return residual

def get_residual_phi(data, mesh, problem):
    # Unpack
    U = data.U
    phi = data.phi
    interior_face_IDs = mesh.interior_face_IDs
    interface_IDs = mesh.interface_IDs

    # Flux function
    flux_phi = Upwind()

    residual_phi = np.zeros_like(phi)

    # Compute limiter
    limiter = compute_limiter(data, mesh)

    # -- Interior Faces -- #
    # Evaluate solution at left and right
    U_L_p1 = data.U_L_p1
    U_R_p1 = data.U_R_p1
    evaluate_solution_at_interior_faces(U, mesh.interior_face_IDs, mesh.edge,
            limiter, data.gradV.flatten().data, mesh.xy, data.fluid_ID,
            data.g, data.psg, U_L_p1, U_R_p1)
    # Left and right cell IDs
    L = mesh.edge[interior_face_IDs, 0]
    R = mesh.edge[interior_face_IDs, 1]
    # Evaluate phi at faces on left and right: first order component
    phi_L = phi[L]
    phi_R = phi[R]
    # Second order component
    quad_pts = .5 * (mesh.xy[L] + mesh.xy[R])
    phi_L += np.einsum('ik, ik -> i', data.grad_phi[L], quad_pts - mesh.xy[L])
    phi_R += np.einsum('ik, ik -> i', data.grad_phi[R], quad_pts - mesh.xy[R])
    # Evalute interior fluxes
    F = flux_phi.compute_flux(U_L_p1, U_R_p1, phi_L, phi_R,
            mesh.area_normals_p1[interior_face_IDs])

    # Interior face contribution to residual
    np.add.at(residual_phi, L, -1 / mesh.area[L] * F)
    np.add.at(residual_phi, R,  1 / mesh.area[R] * F)

    # -- Interfaces -- #
    # Evaluate solution at left and right
    U_L_p2 = np.empty_like(data.U_L_p2).flatten()
    U_R_p2 = np.empty_like(data.U_R_p2).flatten()
    evaluate_solution_at_interfaces(U, mesh.interface_IDs, mesh.edge,
            LagrangeSegment.quad_wts, mesh.quad_pts_phys.flatten().data,
            limiter, data.gradV.flatten().data, mesh.xy, data.fluid_ID,
            data.g, data.psg, U_L_p2, U_R_p2)
    data.U_L_p2 = U_L_p2.reshape(data.U_L_p2.shape)
    data.U_R_p2 = U_R_p2.reshape(data.U_R_p2.shape)
    U_L_p2 = data.U_L_p2
    U_R_p2 = data.U_R_p2
    nq = U_L_p2.shape[1]
    # Left and right cell IDs
    L = mesh.edge[interface_IDs, 0]
    R = mesh.edge[interface_IDs, 1]
    # Evaluate phi at faces on left and right: first order component
    phi_L = phi[L]
    phi_R = phi[R]
    # Copy for each quadrature point
    phi_L = np.tile(phi_L, [4, 1]).T
    phi_R = np.tile(phi_R, [4, 1]).T
    # Second order component
    quad_pts = mesh.quad_pts_phys[interface_IDs]
    phi_L += np.einsum('ik, ijk -> ij', data.grad_phi[L],
            quad_pts - mesh.xy[L].reshape(-1, 1, 2))
    phi_R += np.einsum('ik, ijk -> ij', data.grad_phi[R],
            quad_pts - mesh.xy[R].reshape(-1, 1, 2))
    # Evaluate interior fluxes
    F = np.empty((interface_IDs.size, nq))
    for i in range(nq):
        F[:, i] = flux_phi.compute_flux(U_L_p2[:, i], U_R_p2[:, i], phi_L[:, i],
                phi_R[:, i], mesh.area_normals_p2[interface_IDs, i])
    # Quadrature rule
    F = F @ LagrangeSegment.quad_wts

    # Interface contribution to residual
    np.add.at(residual_phi, L, -1 / mesh.area[L] * F)
    np.add.at(residual_phi, R,  1 / mesh.area[R] * F)

    # -- Boundary Faces -- #
    # Compute ghost phi
    phi_ghost = np.empty((mesh.bc_type.shape[0]))
    problem.compute_ghost_phi(phi, phi_ghost, mesh.bc_type)
    # Evalute boundary fluxes
    # TODO: Is U_ghost really needed here? For now, setting it equal to U.
    U_ghost = U[mesh.bc_type[:, 0]]
    F_bc = flux_phi.compute_flux(U[mesh.bc_type[:, 0]], U_ghost,
            phi[mesh.bc_type[:, 0]], phi_ghost, mesh.bc_area_normal)

    # Boundary face contribution to residual
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
        vel_dot_normal = .5 * (vel_dot_normal_L + vel_dot_normal_R)
        upwindL = vel_dot_normal >= 0
        upwindR = vel_dot_normal < 0
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

    # Testing Riemann problem between the same state
    result = np.empty(9)
#    rL, pL, uL, rR, pR, uR = 1e3, 1e5, 0, 1e3, 1e5, 0
#    gL, gR, psgL, psgR = 4.4, 4.4, 6e5, 6e5
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, gL, gR, psgL, psgR,
#            result)

    # Now with slightly different state
    rL, pL, uL, rR, pR, uR = 1e3, 100000, 0, 1e0, 2000, 0
#    rL, pL, uL, rR, pR, uR = 1e3, 100000, -1, 1e3, 100000, 1
#    rL, pL, uL, rR, pR, uR = 1e3, 150000, .3, 1e3, 140000, .3
#    rL, pL, uL, rR, pR, uR = 1, .4, -2, 1, .4, 2
    gL, gR, psgL, psgR = 4.4, 1.4, 6e5, 0
#    #gL, gR, psgL, psgR = 1.4, 1.4, 0, 0
#    gL, gR, psgL, psgR = 4.4, 4.4, 6e5, 6e5
    n_points = 200
    f = np.empty(n_points)
    p = np.linspace(2000, 1e5, n_points)
    for i in range(n_points):
        result[1] = p[i]
        compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, gL, gR, psgL, psgR,
                result)
        f[i] = result[0]
    import matplotlib.pyplot as plt
    plt.plot(p, f, '-k', lw=3)
    plt.show()
#    compute_exact_riemann_problem(rL, pL, uL, rR, pR, uR, gL, gR, psgL, psgR,
#            result)
    print('r, un, p, g, psg = ', result[:5])
    print('r_starL, r_starR, u_star, p_star = ', result[5:])
    #reakpoint()

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

    print("test")
    U_L = np.array([  1033.20818198,   1864.46042583,   1864.46042651, 789732.18893363])
    U_R = np.array([  1033.20818221,  -1864.46042674,  -1864.46042742, 789732.18889649])
    #U_b = np.array([ 1.02082322e+03, -3.68422265e+02, -3.68422265e+02, 7.84533017e+05])
    U_b = np.array([ 2.04441635e+00, -1.48672743e-08, -1.48674070e-08, 1.40086659e+04])
    compute_flux(U_L, U_b, area_normal, 4.4, 1.4, 6e5, 0, F)
    print(F)
    compute_flux(U_b, U_R, area_normal, 1.4, 4.4, 0, 6e5, F)
    print(F)
