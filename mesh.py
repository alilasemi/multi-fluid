import numpy as np
import scipy.optimize

from lagrange import (LagrangeSegmentP1, LagrangeSegmentP2, LagrangeTriangleP1,
        LagrangeTriangleP2)


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
        self.compute_face_area_normals()

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
        self.neighbors = np.empty(self.n, dtype=object)
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
                    # If this is a left boundary, flip the order to preserve
                    # outward normals
                    if i == 0:
                        self.edge[face_ID] = self.edge[face_ID][::-1]
                        self.edge_area_normal[face_ID] *= -1
                    face_ID += 1

                # Make face to the right
                if (i < nx - 1):
                    right_ID = j*nx + i + 1
                    stencil.append(right_ID)
                    self.edge[face_ID] = [cell_ID, right_ID]
                    self.edge_area_normal[face_ID] = rotation90 @ np.array([-dx/3, -2*dy/3])
                    # If this is a top/bottom boundary, cut it in half
                    if j == 0 or j == ny - 1: self.edge_area_normal[face_ID] /= 2
                    # If this is a top boundary, flip the order to preserve
                    # outward normals
                    if j == ny - 1:
                        self.edge[face_ID] = self.edge[face_ID][::-1]
                        self.edge_area_normal[face_ID] *= -1
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
                self.stencil[cell_ID] = stencil.copy()
                # Also store as the neighbors. Later on, the stencil will be
                # modified by the phase field, but the neighbors will stay the
                # same.
                self.neighbors[cell_ID] = stencil.copy()

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

    def update_stencil(self, phi):
        '''
        Update the stencil to not include points crossing an interface.

        Inputs:
        -------
        phi - array containing level set evaluated at nodes (n,)

        Outputs:
        --------
        self.stencil - array of lists of neighbors, not including those crossing
                an interface (n,)
        '''
        # Loop over nodes
        for i in range(self.n):
            stencil = []
            # Loop over neighbors
            for j in self.neighbors[i]:
                # Check if this is a surrogate interface
                is_surrogate = phi[i] * phi[j] < 0
                # If it isn't an interface, include the point in the stencil
                if not is_surrogate:
                    stencil.append(j)
            # Store
            self.stencil[i] = stencil.copy()
            # Also use for the limiter stencil
            self.limiter_stencil[i, :len(stencil)] = stencil
            self.limiter_stencil[i, len(stencil):] = i

        # There are some unfortunate edge cases to handle. In order to solve the
        # linear system to define a gradient, at least three points are needed
        # in the stencil (since a plane is defined by three unknowns). However,
        # depending on phi, the interface may be in between most of the node
        # neighbors of this cell. In the case that only one neighbor remains
        # (therefore two points in the stencil) one solution is to just append
        # the stencil of that one neighbor to the stencil of the current cell.
        # Loop over nodes
        for i in range(self.n):
            if len(self.stencil[i]) == 2:
                # Get the neighbor
                self.stencil[i].remove(i)
                neighbor_ID = self.stencil[i][0]
                # Append neighbor stencil to this stencil. Since the neighbor ID
                # and cell i are both included in the neighbor stencil, we can
                # just copy it directly.
                self.stencil[i] = self.stencil[neighbor_ID].copy()

        # TODO: Here are some other unfortunate edge cases:
        # - if all neighbors are cut off, and a node is left "alone". Probably
        #   just use zero gradient in that situation.
        # - if there is one neighbor, but even that one neighbor only has one
        #   neighbor. Again, probably just use zero gradient.

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
        # Array of empty lists
        unordered_cell_faces = np.empty(self.n, dtype=object)
        for i in range(self.n): unordered_cell_faces[i] = []
        # Loop over faces
        self.face_points = np.empty((self.n_faces, 3), dtype=int)
        for face_ID in range(self.n_faces):
            # Get dual mesh neighbors
            i, j = self.edge[face_ID]
            # Add this face to an unordered list of the faces of these cells.
            # It's unordered because there is no gaurantee that these faces will
            # be ordered counterclockwise or be connected in the order listed.
            unordered_cell_faces[i].append(face_ID)
            unordered_cell_faces[j].append(face_ID)
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

        # TODO: This is slightly jank (but not too bad). Basically, the idea
        # here is that for this mesh, there are at most 12 points per cell.
        # However, I haven't bothered to sift out the duplicates, so really it
        # stores the 18 points (3 from each face, at most). Also, boundary cells
        # don't have this many. Therefore, there are quite a few either
        # duplicates or empty values, all of which get padded with the nodal
        # coordinate. This works, because the only thing this array is used for
        # is the limiter, which is agnostic to both repeated coordinates and
        # coordinates that are not at the face of a cell (aka, the node
        # coordinate). Just don't use it for anything else.
        # Loop over dual cells
        self.max_num_face_points = 18
        self.cell_point_coords = np.empty((self.n, self.max_num_face_points, 2))
        for i in range(self.n):
            coord_counter = 0
            # Loop over faces
            for face_ID in unordered_cell_faces[i]:
                # Get face point coords
                coords = self.get_face_point_coords(face_ID)
                # Loop over coords
                for coord_ID in range(coords.shape[0]):
                    # Store
                    self.cell_point_coords[i, coord_counter] = coords[coord_ID]
                    coord_counter += 1
            # Pad the rest of the array with the node location
            self.cell_point_coords[i, coord_counter:] = self.xy[i]

    def is_boundary(self, face_ID):
        return self.face_points[face_ID, 2] == -1

    def get_face_point_coords(self, i_face, vol_points = None, edge_points = None):
        '''
        Get coordinates of points on a given dual mesh face.
        '''
        if vol_points is None:
            vol_points = self.vol_points
        if edge_points is None:
            edge_points = self.edge_points
        # If it's a boundary face
        if self.is_boundary(i_face):
            coords = np.empty((2, 2))
            coords[0] = vol_points[self.face_points[i_face, 0]]
            coords[1] = edge_points[self.face_points[i_face, 1]]
        # If it's an interior face
        else:
            coords = np.empty((3, 2))
            coords[0] = vol_points[self.face_points[i_face, 0]]
            coords[1] = edge_points[self.face_points[i_face, 1]]
            coords[2] = vol_points[self.face_points[i_face, 2]]
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
        # TODO: This is with a hardcoded phi. This is because I want to neglect
        # error in phi for now.
        # TODO: For some reason, u needs to be 10 times larger here. A bug
        # somewhere?
        # TODO: The mesh plot thick black line does not line up with the
        # adapted points! (see left)
        u_bubble = 50 * 75
        radius = .25
        def get_phi(coords):
            x, y = coords
            phi = (x - u_bubble * t)**2 + y**2 - radius**2
            phi /= self.xL**2 + self.xR**2 - radius**2
            return phi
        def get_grad_phi(coords):
            x, y = coords
            gphi = np.array([
                2 * (x - u_bubble * t),
                2 * y])
            gphi /= self.xL**2 + self.xR**2 - radius**2
            return gphi
        def get_phi_squared(coords):
            return get_phi(coords)**2
        def get_grad_phi_squared(coords):
            return 2 * get_phi(coords) * get_grad_phi(coords)

        for face_ID in range(self.n_faces):
            # TODO: Only works for interior faces
            if self.is_boundary(face_ID): continue

            # Get dual mesh neighbors
            i, j = self.edge[face_ID]
            # Check for interface
            if get_phi(self.xy[i]) * get_phi(self.xy[j]) < 0:
                # If it's an interface, move the face points towards phi = 0
                for i_point in range(3):
                    # -- Edge points -- #
                    if i_point == 1:
                        coords = self.edge_points[self.face_points[face_ID, i_point]]
                        def constraint_1(coords, xy1, xy2):
                            '''
                            Coords must be on the line between xy1 and xy2.

                            This constraint comes from the law of cosines, with
                            the angle set to 0.
                            '''
                            a = np.linalg.norm(coords - xy1)
                            b = np.linalg.norm(xy2 - xy1)
                            c = np.linalg.norm(coords - xy2)
                            return a**2 + b**2 - c**2 - 2*a*b
                        def jac_1(coords, xy1, xy2):
                            '''
                            Jacobian of constraint 1.
                            '''
                            a = np.linalg.norm(coords - xy1)
                            b = np.linalg.norm(xy2 - xy1)
                            grad_a2 = 2*(coords - xy1)
                            grad_c2 = 2*(coords - xy2)
                            grad_a = (coords - xy1) / a
                            return grad_a2 - grad_c2 - 2*b*grad_a
                        def constraint_2(coords, xy1, xy2):
                            '''
                            Coords must be after xy1.

                            This constraint comes from keeping the dot product
                            positive.
                            '''
                            return np.dot( coords - xy1, xy2 - xy1 )
                        def jac_2(coords, xy1, xy2):
                            '''
                            Jacobian of constraint 2.
                            '''
                            return xy2 - xy1
                        def constraint_3(coords, xy1, xy2):
                            '''
                            Coords must be before xy2.

                            This constraint comes from keeping the dot product
                            positive.
                            '''
                            return np.dot( coords - xy2, xy1 - xy2 )
                        def jac_3(coords, xy1, xy2):
                            '''
                            Jacobian of constraint 3.
                            '''
                            return xy1 - xy2
                        # Solve optimization problem for the new node locations,
                        # by moving them as close as possible to the interface
                        # (phi = 0) while still keeping the point between nodes
                        # i and j
                        # The guess value is important - it cannot start the
                        # guess on the edge itself. Instead, I add half the
                        # vector from i to j, rotated 90 degrees.
                        vector = (self.xy[j] - self.xy[i])/3
                        guess = coords + [vector[1], -vector[0]]
                        optimization = scipy.optimize.minimize(get_phi_squared,
                                guess, jac=get_grad_phi_squared,
                                constraints=[{
                                    'type': 'eq',
                                    'fun': constraint_1,
                                    'jac': jac_1,
                                    'args': (self.xy[i], self.xy[j]),
                                    }, {
                                    'type': 'ineq',
                                    'fun': constraint_2,
                                    'jac': jac_2,
                                    'args': (self.xy[i], self.xy[j]),
                                    }, {
                                    'type': 'ineq',
                                    'fun': constraint_3,
                                    'jac': jac_3,
                                    'args': (self.xy[i], self.xy[j]),
                                    }])
                        if not optimization.success:
                            print(f'Oh no! Edge point of face {face_ID} failed to optimize!')
                        coords[:] = optimization.x
                    # -- Volume points -- #
                    else:
                        def constraint_func(coords, node_coords):
                            '''
                            Constraint to keep the coords within the triangle.

                            Thank you to andreasdr on Stack Overflow:
                            https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
                            '''
                            # Compute Barycentric coordinate parameters
                            px, py = coords
                            p0x, p0y = node_coords[0]
                            p1x, p1y = node_coords[1]
                            p2x, p2y = node_coords[2]
                            s = p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py
                            t = p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py
                            return np.array([s, t, 1 - s - t])
                        def constraint_jac(coords, node_coords):
                            '''
                            Compute the Jacobian of the constraint.
                            '''
                            p0x, p0y = node_coords[0]
                            p1x, p1y = node_coords[1]
                            p2x, p2y = node_coords[2]
                            dsdx = p2y - p0y
                            dsdy = p0x - p2x
                            dtdx = p0y - p1y
                            dtdy = p1x - p0x
                            return np.array([
                                    [dsdx, dsdy],
                                    [dtdx, dtdy],
                                    [-dsdx - dtdx, -dsdy - dtdy]])
                        cell_ID = self.face_points[face_ID, i_point]
                        coords = self.vol_points[cell_ID]
                        # Get primal cell nodes
                        nodes = self.primal_cell_to_nodes[cell_ID]
                        node_coords = self.xy[nodes]
                        # Solve optimization problem for the new node locations,
                        # by moving them as close as possible to the interface
                        # (phi = 0) while still keeping the point within the
                        # triangle
                        optimization = scipy.optimize.minimize(get_phi_squared,
                                coords, jac=get_grad_phi_squared,
                                constraints=[{
                                    'type': 'ineq',
                                    'fun': constraint_func,
                                    'jac': constraint_jac,
                                    'args': (node_coords,)}])
                        coords[:] = optimization.x

        # Now that the face points have moved, the area of each dual mesh cell
        # has changed, and thus needs to be recalculated
        self.compute_cell_areas()

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
                # Add contribution to area
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
                # Add contribution to area
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

    def compute_face_area_normals(self):
        '''
        Compute the area-weighted normals of each dual face.
        '''
        self.area_normals_p2 = np.empty((self.n_faces, 2, 2))
        self.quad_pts_phys = np.empty((self.n_faces, 2, 2))
        # Loop over faces
        for face_ID in range(self.n_faces):
            # Get dual mesh neighbors
            i, j = self.edge[face_ID]
            # Coordinates of the three points on the face
            face_point_coords = self.get_face_point_coords(face_ID)
            x = face_point_coords[:, 0]
            y = face_point_coords[:, 1]

            # If it's an interior face, use second order elements
            if face_point_coords.shape[0] == 3:
                # Create a Lagrange segment
                x_seg = LagrangeSegmentP2(x)
                y_seg = LagrangeSegmentP2(y)

            # If it's a boundary face, use first order elements
            if face_point_coords.shape[0] == 2:
                # Create a Lagrange segment
                x_seg = LagrangeSegmentP1(x[:2])
                y_seg = LagrangeSegmentP1(y[:2])

            # Evaluate location of quadrature points in physical space
            self.quad_pts_phys[face_ID, :, 0] = np.matmul(
                    x_seg.get_basis_values(x_seg.quad_pts), x_seg.coords)
            self.quad_pts_phys[face_ID, :, 1] = np.matmul(
                    y_seg.get_basis_values(y_seg.quad_pts), y_seg.coords)

            # Now have d x / d xi and d y / d eta, and this represents the
            # length-weighted vectors along the face. Rotate by 90 degrees
            # to get the normal vector. This is done by the transformation
            # [x, y] -> [-y, x].
            self.area_normals_p2[face_ID, :, 0] = -y_seg.jac
            self.area_normals_p2[face_ID, :, 1] =  x_seg.jac
