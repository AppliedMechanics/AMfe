#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Strcutural assembly.

Basic assembly module for the finite element code. Assumes to have all elements in the inertial frame. Provides an
assembly class which knows the mesh. It can assemble the vector of nonlinear forces, the mass matrix and the tangential
stiffness matrix. Some parts of the code --- mostly the indexing of the sparse matrices --- are substituted by fortran
routines, as they allow for a huge speedup.
"""

import numpy as np
import logging
import time
from scipy.sparse import csr_matrix

from .assembly import Assembly

__all__ = [
    'StructuralAssembly'
]


class StructuralAssembly(Assembly):
    """
    Class handling assembly of elements for structures.

    Attributes
    ----------
    element_mapping : list
        Ragged list containing the global indices for the local variables
        of an element. The entry mapping(i,j) gives the index in the global vector
        of element i with dof j
    boundary_element_mapping : ndarray
        Ragged array equivalently to element_mapping for the neumann
        boundary skin elements.
    _no_of_dofs_per_node : int
        number of degrees of freedem per node
    _node_mapping : ndarray
        mapping from nodeidx and direction to global dof
        rows = nodeidx, columns = directions (0 = x, 1 = y, [2 = z])
    elements_on_node : list
        contains the number of elements that belong to a node. This property is needed for computing
        stresses during postprocessing

    """

    COORD2COL = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, dimension, nodes, connectivity, boundary_connectivity=None):
        """
        Parameters
        ----------
        dimension : int
            Dimension of the mesh being assembled (2 or 3)
        nodes : ndarray
            node coordinates
        connectivity : list
            list with ndarrays that decribe the connectivity
        """

        super().__init__()
        self.logger = logging.getLogger('amfe.assembly.StructuralAssembly')
        self._no_of_dofs_per_node = dimension
        # mapping
        # set standard mapping
        self._node_mapping = np.arange(0, nodes.shape[0]*dimension).reshape(-1, dimension)
        self.element_mapping = [] * len(connectivity)
        if boundary_connectivity is not None:
            self.boundary_element_mapping = [] * len(boundary_connectivity)
        else:
            self.boundary_element_mapping = []
        self.elements_on_node = []
        self.compute_element_mapping(connectivity, boundary_connectivity)
        self.C_csr = None
        self.preallocate(self.no_of_dofs, self.element_mapping)
        return

    @property
    def node_mapping(self):
        return self._node_mapping

    @node_mapping.setter
    def node_mapping(self, value):
        self._node_mapping = value
        # compute_element_indices should be called by an observer
        self.notify()
        return

    @property
    def no_of_dofs_per_node(self):
        """
        Returns
        -------
        no_of_dofs_per_node : int
            returns the number of dofs associated with one node
        """

        return self._no_of_dofs_per_node

    @property
    def no_of_dofs(self):
        return len(np.unique(self._node_mapping))

    def get_dofs_by_nodeidxs(self, nodeidxs, coords):
        """
        Returns the global dofs associated with a given node-row-index and a direction x, y or z

        Parameters
        ----------
        nodeidxs : iterable
            Row indices of nodes where one wants to know their global dofs
        coords : str
            str with a combination of 'x', 'y' and 'z', e.g. 'xy', 'y', 'xyz' where one wants to know the global dofs.

        Returns
        -------
        dofs : ndarray
            array with global dofs
        """
        cols = np.array([self.COORD2COL[coord] for coord in coords], dtype=int)
        rows = np.array(nodeidxs, dtype=int)

        return self.node_mapping[np.ix_(rows, cols)].reshape(-1)

    def compute_element_mapping(self, connectivity, boundary_connectivity=None):
        """
        Compute the mapping between elements, their local dofs and the global dofs.

        The element_mapping is a list, where every element of the list denotes
        the global dofs of the element in the correct order.

        Parameters
        ----------
        connectivity : list
            list with ndarrays describing the connectivity (topology) of volume elements in the mesh
            (row-indices of a node array)
        boundary_connectivity : list
            list with ndarrays describing the connectivity (topology) of boundary elements in the mesh
            (row-indices of a node array)

        Returns
        -------
        None
        """

        # Explanation of following expression: for each element in connectivity and for each node-id of each element
        # take [0,1] (2D-problem) or [0,1,2] (3D-problem) and add 2*node_id (2D-problem) or 3*node_id (3D-problem) and
        # reshape the result... Result (self.element_indices:) the rows are the elements, the columns are the local
        # element dofs the values are the global dofs
        cols = np.arange(self._no_of_dofs_per_node)

        self.element_mapping = [self.node_mapping[np.ix_(element, cols)].reshape(-1) for element in connectivity]
        self.boundary_element_mapping = [self.node_mapping[np.ix_(element, cols)].reshape(-1) for element
                                         in boundary_connectivity]

        # compute nodes_frequency for stress recovery
        nodes_vec = np.concatenate(connectivity)
        self.elements_on_node = np.bincount(nodes_vec)
        return

    def preallocate(self, no_of_dofs, eleidx2globaldofs):
        """
        Compute the sparsity pattern of the assembled matrices and store an empty matrix in self.C_csr.

        The matrix self.C_csr serves as a 'blueprint' matrix which is filled in the assembly process.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This pre-allocation routine can take some while for large matrices. Furthermore it is not implemented
        memory-efficient, so for large systems and low RAM this might become an issue...
        """

        self.logger.info('Pre-allocating the stiffness matrix')
        t1 = time.clock()

        # NOTE
        # the following algorithm only works under the following constraints:
        #   - the mapping starts at zero
        #   - if there are gaps in the mapping, they will not be pre-allocated

        max_dofs_per_element = max((len(i) for i in eleidx2globaldofs))

        # Auxiliary Help-Matrix H which is the blueprint of the local element stiffness matrix
        H = np.zeros((max_dofs_per_element, max_dofs_per_element))

        # pre-allocate the CSR-matrix

        # pre-allocate row_global with maximal possible size for pre-alloc. C_csr
        row_global = np.zeros(len(eleidx2globaldofs) * max_dofs_per_element ** 2, dtype=int)
        # pre-allocate col_global with maximal possible size for pre-alloc. C_csr
        col_global = row_global.copy()
        # set 'dummy' values
        vals_global = np.zeros_like(col_global, dtype=bool)

        # calculate row_global and col_global
        for i, global_dofs_of_current_element in enumerate(eleidx2globaldofs):
            l = len(global_dofs_of_current_element)
            # insert global-dof-ids in l rows (l rows have equal entries)
            H[:l, :l] = global_dofs_of_current_element
            # calculate row_global and col_global such that every possible combination of indices_of_one_element can be
            # returned by (row_global[k], col_global[k]) for all k
            row_global[i * max_dofs_per_element ** 2:(i + 1) * max_dofs_per_element ** 2] = H.reshape(-1)
            col_global[i * max_dofs_per_element ** 2:(i + 1) * max_dofs_per_element ** 2] = H.T.reshape(-1)

        # fill C_csr matrix with dummy entries in those places where matrix will be filled in assembly
        self.C_csr = csr_matrix((vals_global, (row_global, col_global)), shape=(no_of_dofs, no_of_dofs), dtype=float)

        t2 = time.clock()
        self.logger.info('Done pre-allocating stiffness matrix with {0:d} elements and {1:d} dofs.'
                         .format(len(eleidx2globaldofs), no_of_dofs))
        self.logger.info('Time taken for pre-allocation: {0:2.2f} seconds.'.format(t2 - t1))
