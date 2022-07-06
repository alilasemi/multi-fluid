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
        # Number of primal mesh cells
        self.n_primal_cells = 2 * (nx - 1) * (ny - 1)
        # Compute nodes
        x = np.linspace(self.xL, self.xR, nx)
        y = np.linspace(self.yL, self.yR, ny)
        grid = np.meshgrid(x, y)
        self.xy = np.empty((self.n, 2))
        self.xy[:, 0] = grid[0].flatten()
        self.xy[:, 1] = grid[1].flatten()

        self.compute_areas()
        self.create_dual_faces()
        self.create_primal_cells()
        self.create_face_points()

    def compute_areas(self):
        '''
        Compute the area of each cell.
        '''
        # Unpack
        nx = self.nx
        ny = self.ny
        # Grid spacing
        dx = (self.xR - self.xL) / (nx - 1)
        dy = (self.yR - self.yL) / (ny - 1)
        self.dx = dx
        self.dy = dy
        def get_area(x, y):
            '''
            Get area of a triangle defined by counterclockwise points.
            '''
            return 0.5*( (x[0]*(y[1]-y[2])) + (x[1]*(y[2]-y[0])) + (x[2]*(y[0]-y[1])) )
        # These are the various triangles that make up the hexahedral dual mesh
        tri0 = get_area([0, dx/3, 0], [0, 2*dy/3, dy/2])
        tri1 = get_area([0, 2*dx/3, dx/3], [0, dy/3, 2*dy/3])
        tri2 = tri0
        tri3 = get_area([0, dx/3, dx/2], [0, -dy/3, 0])
        tri4 = tri3
        self.area = 2 * (tri0 + tri1 + tri2 + tri3 + tri4) * np.ones(nx * ny)
        # Find boundaries and set their area
        boundary_area = tri0 + tri1 + tri2 + tri3 + tri4
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
        self.area[self.corner_NE] = tri0 + tri1 + tri2
        self.area[self.corner_SE] = tri3 + tri4
        self.area[self.corner_NW] = tri3 + tri4
        self.area[self.corner_SW] = tri0 + tri1 + tri2

    def create_dual_faces(self):
        '''
        Create the interior and boundary face information for the dual mesh.
        '''
        # Unpack
        nx = self.nx
        ny = self.ny
        dx = self.dx
        dy = self.dy
        # Allocate space
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
                    self.bc_type[BC_ID] = [cell_ID, 3]
                    self.bc_area_normal[BC_ID] = [dy, 0]
                    # Inflow is negative
                    if i == 0:
                        self.bc_type[BC_ID] = [cell_ID, 2]
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

        # Loop over faces
        for face_ID in range(self.n_faces):
            # compute the edge point as being halfway between the two points
            # defining the edge
            self.edge_points[face_ID] = np.mean(self.xy[self.edge[face_ID]],
                    axis=0)

    def create_primal_cells(self):
        '''
        Create the nodes and neighbors of each primal cell.
        '''
        # Unpack
        nx = self.nx
        ny = self.ny
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

        # Loop over primal cells
        self.nodes_to_primal_cells = np.empty(self.n, dtype=object)
        for cell_ID in range(self.n_primal_cells):
            # Loop over nodes of this primal cell
            for i in self.primal_cell_to_nodes[cell_ID]:
                # If this node hasn't been encountered yet, create the list for
                # it
                if self.nodes_to_primal_cells[i] is None:
                    self.nodes_to_primal_cells[i] = [cell_ID]
                # Otherwise, just append
                else:
                    self.nodes_to_primal_cells[i].append(cell_ID)
        # Loop back through and convert all lists to Numpy arrays
        for i in range(self.n):
            self.nodes_to_primal_cells[i] = np.array(
                    self.nodes_to_primal_cells[i])

    def create_face_points(self):
        '''
        Create the points on each dual mesh face.
        '''
        # Loop over faces
        self.face_points = np.empty((self.n_faces, 3), dtype=int)
        for i_face in range(self.n_faces):
            # Get dual mesh neighbors
            i, j = self.edge[i_face]
            # Get primal cells of each node
            i_primals = self.nodes_to_primal_cells[i]
            j_primals = self.nodes_to_primal_cells[j]
            # The intersection gives the primal cells of this face
            indices = np.intersect1d(i_primals, j_primals)

            # If this is a boundary face
            if indices.size == 1:
                # Start with an empty volume point
                self.face_points[i_face, 0] = -1
            # If this is an interior face
            else:
                # Start with whichever side came up first in the search
                self.face_points[i_face, 0] = indices[0]

            # Add edge point
            self.face_points[i_face, 1] = i_face

            # Add final volume point
            self.face_points[i_face, 2] = indices[-1]

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

    def get_plot_points_primal_cell(self, cell_ID):
        '''
        Get coordinates of points to plot for a primal cell.
        '''
        coords = np.empty((4, 2))
        coords[:3] = self.xy[self.primal_cell_to_nodes[cell_ID]]
        coords[3] = self.xy[self.primal_cell_to_nodes[cell_ID, 0]]
        return coords
