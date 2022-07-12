import numpy as np

from lagrange import LagrangeTriangleP1, LagrangeTriangleP2


class Mesh:
    '''
    Class for generating and storing the mesh.

    The mesh is a rectangular domain with triangular primal cells, where the
    diagonals of the triangles go from bottom-left to top-right. The mesh used
    for the actual computation is the dual mesh.
    '''
    def __init__(self, nx, ny, xL, xR, yL, yR):
        self.nx = nx
        self.ny = ny
        self.xL = xL
        self.xR = xR
        self.yL = yL
        self.yR = yR
        self.n = nx * ny
        # Grid spacing
        self.dx = (xR - xL) / (nx - 1)
        self.dy = (yR - yL) / (ny - 1)
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

        self.create_dual_faces()
        self.create_primal_cells()
        self.create_face_points()
        self.compute_cell_areas()

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

        Dual mesh faces are defined between nodes i and j. In the interior, a
        face extends from the "left" primal cell centroid to the i-j midpoint,
        then finally to the "right" primal cell centroid. This looks like:

            "left" primal ->  L   j  <- node j
                               \ /
                                O  <- edge point
                               / \
                   node i ->  i   R  <- "right" primal

        The i-j midpoint is called an edge point, and there are as many edge
        points as dual faces. The L and R primal centroids are called the left
        and right volume point, and each interior face has two volume points
        while boundary faces have one. The ordering is important - which primal
        is L vs. R is chosen to preserve the counterclockwise node ordering of
        the primal cells.
        '''
        # Loop over faces
        self.face_points = np.empty((self.n_faces, 3), dtype=int)
        for face_ID in range(self.n_faces):
            # Get dual mesh neighbors
            i, j = self.edge[face_ID]
            # Get primal cells of each node
            i_primals = self.nodes_to_primal_cells[i]
            j_primals = self.nodes_to_primal_cells[j]
            # The intersection gives the primal cells of this face
            indices = np.intersect1d(i_primals, j_primals)
            # Check the ordering. This is done by checking the node order of the
            # intersected cells. The primal cells have nodes ordered in
            # counterclockwise order, so indices[0] must contain nodes i, j in
            # this order. This only matters for interior faces (boundary faces
            # always have one primal cell, and this is the "left" cell).
            if len(indices) == 2:
                nodes = self.primal_cell_to_nodes[indices[0]]
                if not (np.all(nodes[[0, 1]] == [i, j])
                        or np.all(nodes[[1, 2]] == [i, j])
                        or np.all(nodes[[2, 0]] == [i, j])):
                    indices = indices[::-1]

            # Start with the first volume point on the "left"
            self.face_points[face_ID, 0] = indices[0]

            # Add edge point
            self.face_points[face_ID, 1] = face_ID

            # If this is a boundary face
            if indices.size == 1:
                # Add an empty volume point
                self.face_points[face_ID, 2] = -1
            # If this is an interior face
            else:
                # Add final volume point
                self.face_points[face_ID, 2] = indices[-1]

    def get_face_point_coords(self, i_face):
        '''
        Get coordinates of points on a given dual mesh face.
        '''
        # If it's a boundary face
        if self.face_points[i_face, 2] == -1:
            coords = np.empty((2, 2))
            coords[0] = self.vol_points[self.face_points[i_face, 0]]
            coords[1] = self.edge_points[self.face_points[i_face, 1]]
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

    def update(self, data):
        '''
        Update the dual mesh to fit the interface better.
        '''
        t = data.t
        max_iter = 10
        for iter in range(max_iter):
            print(f'mesh iteration = {1 + iter}/{max_iter}', end='\r')
            # TODO: This is with a hardcoded phi. This is because I want to neglect
            # error in phi for now.
            u = 50
            radius = .25
            def get_phi(x, y):
                phi = (x - u * t)**2 + y**2 - radius**2
                phi /= self.xL**2 + self.xR**2 - radius**2
                return phi
            def get_grad_phi(x, y):
                gphi = np.array([
                    2 * (x - u * t),
                    2 * y])
                gphi /= self.xL**2 + self.xR**2 - radius**2
                return gphi

            for face_ID in range(self.n_faces):
                # Get dual mesh neighbors
                i, j = self.edge[face_ID]
                # Check for interface
                xi, yi = self.xy[i]
                xj, yj = self.xy[j]
                if get_phi(xi, yi) * get_phi(xj, yj) < 0:
                    # If it's an interface, move the edge point towards phi = 0
                    x, y = self.edge_points[face_ID]
                    phi = get_phi(x, y)
                    gphi = get_grad_phi(x, y)
                    if np.abs(gphi[0]) > 1e-5:
                        self.edge_points[face_ID, 0] -= .3 * phi / gphi[0]
                    if np.abs(gphi[1]) > 1e-5:
                        self.edge_points[face_ID, 1] -= .3 * phi / gphi[1]

            phi_nodes = get_phi(self.xy[:, 0], self.xy[:, 1])
            for cell_ID in range(self.n_primal_cells):
                nodes = self.primal_cell_to_nodes[cell_ID]
                phi = phi_nodes[nodes]
                products = np.array([phi[0] * phi[1], phi[1] * phi[2], phi[2] * phi[0]])
                if np.any(products < 0):
                    x, y = self.vol_points[cell_ID]
                    phi = get_phi(x, y)
                    gphi = get_grad_phi(x, y)
                    self.vol_points[cell_ID, 0] -= .3 * phi / gphi[0]
                    self.vol_points[cell_ID, 1] -= .3 * phi / gphi[1]
        print()

    def compute_cell_areas(self):
        '''
        Compute the area of each dual cell.
        '''
        self.area = np.zeros(self.n)
        # Loop over faces
        for face_ID in range(self.n_faces):
            # Get dual mesh neighbors
            i, j = self.edge[face_ID]
            # Coordinates of the three points on the face
            face_point_coords = self.get_face_point_coords(face_ID)

            # If it's an interior face, use second order elements
            if face_point_coords.shape[0] == 3:
                # The six points defining a second order triangle element
                points = np.empty((6, 2))
                # -- Triangle on side of node i -- #
                # Point 0 is the left node
                points[0] = self.xy[i]
                # Point 2 is the right primal mesh centroid
                points[2] = face_point_coords[2]
                # Point 1 is halfway between 0 and 2
                points[1] = .5 * (points[0] + points[2])
                # Point 5 is the left primal mesh centroid
                points[5] = face_point_coords[0]
                # Point 3 is halfway between 0 and 5
                points[3] = .5 * (points[0] + points[5])
                # Point 4 is the edge point
                points[4] = face_point_coords[1]
                # Create a Lagrange triangle
                tri = LagrangeTriangleP2(points)
                # Add contribution to self.area
                self.area[i] += tri.area

                # -- Triangle on side of node j -- #
                # Point 0 is the right node
                points[0] = self.xy[j]
                # Point 2 is the left primal mesh centroid
                points[2] = face_point_coords[0]
                # Point 1 is halfway between 0 and 2
                points[1] = .5 * (points[0] + points[2])
                # Point 5 is the right primal mesh centroid
                points[5] = face_point_coords[2]
                # Point 3 is halfway between 0 and 5
                points[3] = .5 * (points[0] + points[5])
                # Point 4 is the edge point
                points[4] = face_point_coords[1]
                # Create a Lagrange triangle
                tri = LagrangeTriangleP2(points)
                # Add contribution to self.area
                self.area[j] += tri.area

            # If it's a boundary face, use first order elements
            if face_point_coords.shape[0] == 2:
                # The three points defining a first order triangle element
                points = np.empty((3, 2))
                # -- Triangle on side of node i -- #
                # Point 0 is the left node
                points[0] = self.xy[i]
                # Point 1 is the edge point
                points[1] = face_point_coords[1]
                # Point 2 is the primal mesh centroid
                points[2] = face_point_coords[0]
                # Create a Lagrange triangle
                tri = LagrangeTriangleP1(points)
                # Add contribution to self.area
                self.area[i] += np.abs(tri.area)

                # -- Triangle on side of node j -- #
                # Point 0 is the left node
                points[0] = self.xy[j]
                # Point 1 is the primal mesh centroid
                points[1] = face_point_coords[0]
                # Point 2 is the edge point
                points[2] = face_point_coords[1]
                # Create a Lagrange triangle
                tri = LagrangeTriangleP1(points)
                # Add contribution to self.area
                self.area[j] += np.abs(tri.area)

                # TODO: The np.abs is added since I do not guarantee the
                # direction of boundary faces being outwards pointing normals.
                # Is this going to be a problem?
