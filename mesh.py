import numpy as np


class Mesh:
    '''
    Class for generating and storing the mesh.

    The mesh is a rectangular domain with triangular primal cells, where the
    diagonals of the triangles go from bottom-left to top-right. The mesh used
    for the actual computation is the dual mesh.
    '''
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
        self.corner_NE = np.intersect1d(idx_xR, idx_yR)[0]
        self.corner_SE = np.intersect1d(idx_xR, idx_yL)[0]
        self.corner_NW = np.intersect1d(idx_xL, idx_yR)[0]
        self.corner_SW = np.intersect1d(idx_xL, idx_yL)[0]
        self.area[self.corner_NE] = triangle_area - small_triangle_area
        self.area[self.corner_SE] = small_triangle_area
        self.area[self.corner_NW] = small_triangle_area
        self.area[self.corner_SW] = triangle_area - small_triangle_area
        # -- Faces -- #
        self.stencil = np.empty(self.n, dtype=object)
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

                # List containing stencil points, including the current cell
                stencil = [cell_ID]

                # Make face above
                if (j < ny - 1):
                    above_ID = (j + 1)*nx + i
                    stencil.append(above_ID)
                    self.edge[face_ID] = [cell_ID, above_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([2*dx/3, dy/3])
                    # If this is a left/right boundary, cut it in half
                    if i == 0 or i == nx - 1: self.edge_area_normal[face_ID] /= 2
                    face_ID += 1

                # Make face to the right
                if (i < nx - 1):
                    right_ID = j*nx + i + 1
                    stencil.append(right_ID)
                    self.edge[face_ID] = [cell_ID, right_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([-dx/3, -2*dy/3])
                    # If this is a top/bottom boundary, cut it in half
                    if j == 0 or j == ny - 1: self.edge_area_normal[face_ID] /= 2
                    face_ID += 1

                # Make face diagonally above and to the right
                if (i < nx - 1 and j < ny - 1):
                    diag_ID = (j + 1)*nx + i + 1
                    stencil.append(diag_ID)
                    self.edge[face_ID] = [cell_ID, diag_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([dx/3, -dy/3])
                    face_ID += 1

                # Also check left and bottom side (not needed for faces since
                # it's redundant, but useful for finishing the stencil)
                if (i > 0):
                    left_ID = j*nx + i - 1
                    stencil.append(left_ID)
                if (j > 0):
                    below_ID = (j - 1)*nx + i
                    stencil.append(below_ID)

                # Store stencil
                self.stencil[cell_ID] = stencil

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
