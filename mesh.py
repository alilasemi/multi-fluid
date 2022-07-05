import numpy as np


class Mesh:
    '''
    Class for generating and storing the mesh.

    The mesh is a rectangular domain with triangular primal cells, where the
    diagonals of the triangles go from bottom-left to top-right. The mesh used
    for the actual computation is the dual mesh.
    '''
    # Domain
    xL = -9
    xR = 9
    yL = -1
    yR = 1

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        # Number of faces (formula given in assignment)
        self.n_faces = int( (nx - 1)*ny + (ny - 1)*nx + (nx - 1)*(ny - 1) )
        # Number of primal mesh cells
        self.n_primal_cells = 2 * (nx - 1) * (ny - 1)
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
        self.edge_points = np.empty((self.n_faces, 2))
        self.vol_points = np.empty((self.n_primal_cells, 2))
        self.stencil = np.empty(self.n, dtype=object)
        self.max_limiter_stencil_size = 6
        self.limiter_stencil = np.empty((self.n, self.max_limiter_stencil_size),
                dtype=int)
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

                # Store a vectorized-friendly version of stencil, with a
                # hardcoded maximum stencil size to avoid the jagged array.
                # Also, don't include the current cell in this one. Instead, the
                # extra "padded" array elements are  the current cell.
                # This is used for the limiter.
                self.limiter_stencil[cell_ID, :len(stencil)] = stencil
                self.limiter_stencil[cell_ID, len(stencil):] = cell_ID

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

        # Loop over primal cells
        self.primal_cell_to_nodes = np.empty((self.n_primal_cells, 3), dtype=int)
        self.primal_cell_neighbors = np.empty((self.n_primal_cells, 3), dtype=int)
        self.vol_points = np.empty((self.n_primal_cells, 2))
        for idx in range(0, self.n_primal_cells, 2):
            # Get node indices
            i = (idx // 2) % (nx - 1)
            j = (idx // 2) // (nx - 1)

            # top-left primal triangle
            self.primal_cell_to_nodes[idx] = np.array([
                j * nx + i, (j+1) * nx + i+1, (j+1) * nx + i])
            # bottom-right primal triangle
            self.primal_cell_to_nodes[idx + 1] = np.array([
                j * nx + i, j * nx + i+1, (j+1) * nx + i+1])

            # Neighbors
            self.primal_cell_neighbors[idx] = np.array([
                idx - 1, idx + 1, idx + 2 * nx - 1])
            self.primal_cell_neighbors[idx + 1] = np.array([
                idx, idx - 2 * nx + 2, idx + 2])
            # Account for boundaries by setting them to -1
            self.primal_cell_neighbors[
                    np.nonzero(self.primal_cell_neighbors < 0)] = -1
            self.primal_cell_neighbors[
                    np.nonzero(self.primal_cell_neighbors >= self.n_primal_cells)] = -1

            # Centroids
            self.vol_points[idx] = np.mean(
                    self.xy[self.primal_cell_to_nodes[idx]], axis=0)
            self.vol_points[idx + 1] = np.mean(
                    self.xy[self.primal_cell_to_nodes[idx + 1]], axis=0)

        # Loop over nodes
        self.edge_points = np.empty((self.n_faces, 2))
        edge_dict = {i : {} for i in range(self.n)}
        for i in range(self.n):
            # For each neighbor, add edge point to dict
            # TODO: This assumes stencil = neighbors
            for j in self.stencil[i]:
                if i != j:
                    edge_dict[i][j] = .5 * (self.xy[i] + self.xy[j])
                    # Also copy to the inverse mapping
                    edge_dict[j][i] = edge_dict[i][j].copy()
        # Loop over this newly created dict
        i_edge = 0
        for i in edge_dict.keys():
            # Loop over neighbors
            for j in edge_dict[i].keys():
                # Store point and i_edge
                self.edge_points[i_edge] = edge_dict[i][j]
                edge_dict[i][j] = i_edge
                i_edge += 1
                # Remove duplicate face
                del edge_dict[j][i]

        # Loop over faces
        self.face_points = np.empty((self.n_faces, 3), dtype=int)
        for i_face in range(self.n_faces):
            # Get dual mesh neighbors
            i, j = self.edge[i_face]
            # Search for these nodes on the primal mesh
            # TODO: This loop might make this all pretty slow
            indices = []
            for idx in range(self.n_primal_cells):
                # If both nodes i and j are part of this primal cell
                if i in self.primal_cell_to_nodes[idx] and j in self.primal_cell_to_nodes[idx]:
                    # Store
                    indices.append(idx)
                    if len(indices) == 2: break

            # If this is a boundary face
            if len(indices) == 1:
                # Start with an empty volume point
                self.face_points[i_face, 0] = -1
            # If this is an interior face
            else:
                # Start with whichever side came up first in the search
                self.face_points[i_face, 0] = indices[0]

            # Add edge point
            try:
                self.face_points[i_face, 1] = edge_dict[i][j]
            except:
                self.face_points[i_face, 1] = edge_dict[j][i]

            # Add final volume point
            self.face_points[i_face, 2] = indices[-1]
        breakpoint()

    def get_face_point_coords(self, i_face):
        '''
        Get coordinates of points on a given face.
        '''
        # If it's a boundary face
        if self.face_points[i_face, 0] == -1:
            coords = np.empty((2, 2))
            coords[0] = self.edge_points[self.face_points[i_face, 1]]
            coords[1] = self.vol_points[self.face_points[i_face, 2]]
        # If it's an interior face
        else:
            coords = np.empty((3, 2))
            coords[0] = self.vol_points[self.face_points[i_face, 0]]
            coords[1] = self.edge_points[self.face_points[i_face, 1]]
            coords[2] = self.vol_points[self.face_points[i_face, 2]]
        return coords
