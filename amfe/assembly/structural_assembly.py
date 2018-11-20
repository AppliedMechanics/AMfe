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
from .tools import fill_csr_matrix

__all__ = [
    'StructuralAssembly'
]


class StructuralAssembly(Assembly):
    """
    Class handling assembly of elements for structures.

    Attributes
    ----------
    elements_on_node : list
        contains the number of elements that belong to a node. This property is needed for computing
        stresses during postprocessing
    """

    def __init__(self, fields, nodes, connectivity):
        """
        Parameters
        ----------
        fields : tuple
            tuple with names of physical fields
        nodes : ndarray
            node coordinates
        connectivity : list
            list with ndarrays that decribe the connectivity
        """

        super().__init__()
        self.logger = logging.getLogger('amfe.assembly.StructuralAssembly')
        # compute nodes_frequency for stress recovery
        # TODO: move this to another class
        if connectivity is not None:
            nodes_vec = np.concatenate(connectivity)
            self.elements_on_node = np.bincount(nodes_vec)
        else:
            self.elements_on_node = None
        self.C_csr = None
        return

    def preallocate(self, no_of_dofs, elements2global):
        """
        Compute the sparsity pattern of the assembled matrices and store an empty matrix in self.C_csr.

        The matrix self.C_csr serves as a 'blueprint' matrix which is filled in the assembly process.

        Parameters
        ----------
        no_of_dofs : int
            number of degrees of freedom to preallocate a matrix
        elements2global : list
            list with arrays that map the elements to global dof indices

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

        max_dofs_per_element = max((len(i) for i in elements2global))

        # Auxiliary Help-Matrix H which is the blueprint of the local element stiffness matrix
        H = np.zeros((max_dofs_per_element, max_dofs_per_element))

        # pre-allocate the CSR-matrix

        # pre-allocate row_global with maximal possible size for pre-alloc. C_csr
        row_global = np.zeros(len(elements2global) * max_dofs_per_element ** 2, dtype=int)
        # pre-allocate col_global with maximal possible size for pre-alloc. C_csr
        col_global = row_global.copy()
        # set 'dummy' values
        vals_global = np.zeros_like(col_global, dtype=bool)

        # calculate row_global and col_global
        for i, global_dofs_of_current_element in enumerate(elements2global):
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
                         .format(len(elements2global), no_of_dofs))
        self.logger.info('Time taken for pre-allocation: {0:2.2f} seconds.'.format(t2 - t1))
        return

    def assemble_k_and_f(self, nodes, ele_objects, connectivity, element2dofs, dofvalues=None, t=0.):
        """
        Assemble the tangential stiffness matrix and nonliner internal or external force vector.

        This method can be used for assembling K_int and f_int or for assembling K_ext and f_ext depending on which
        ele_objects and connectivities are passed

        Parameters
        ----------
        nodes : ndarray
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivity : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes ndarray
        element2dofs : list of ndarrays
            Mapping the elements to their global dofs
        dofvalues : ndarray
            current values of all dofs (at time t)
        t : float
            time. Default: 0.

        Returns
        --------
        K : sparse.csr_matrix
            unconstrained assembled stiffness matrix in sparse matrix csr-format.
        f : np.array
            unconstrained assembled nonliner internal force vector

        Examples
        ---------
        TODO
        """

        if dofvalues is None:
            maxdof = np.max(element2dofs)
            dofvalues = np.zeros(maxdof + 1)

        # allocate K and f
        K_csr = self.C_csr.copy()
        f_glob = np.zeros(K_csr.shape[1])

        # loop over all elements
        # (i - element index, indices - DOF indices of the element)
        for i, nodeindices in enumerate(connectivity):
            # X - undeformed positions of the i-th element
            X_local = nodes[nodeindices, :].reshape(-1)
            # global dofs of the i-th element
            globaldofindices = element2dofs[i]
            # displacements of the i-th element
            u_local = dofvalues[globaldofindices]
            # computation of the element tangential stiffness matrix and nonlinear force
            K_local, f_local = ele_objects[i].k_and_f_int(X_local, u_local, t)
            # adding the local force to the global one
            f_glob[globaldofindices] += f_local
            # this is equal to K_csr[globaldofindices, globaldofindices] += K_local
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K_local, globaldofindices)
        return K_csr, f_glob

    def assemble_m(self, nodes, ele_objects, connectivity, element2dofs, dofvalues=None, t=0):
        """
        Assembles the mass matrix of the given mesh and element.

        Parameters
        ----------
        nodes : ndarray
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivity : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes ndarray
        element2dofs : list of ndarrays
            Mapping the elements to their global dofs
        dofvalues : ndarray
            current values of all dofs (at time t)
        t : float
            time. Default: 0.

        Returns
        --------
        M : sparse.csr_matrix
            unconstrained assembled mass matrix in sparse matrix csr-format.

        Examples
        ---------
        TODO
        """

        if dofvalues is None:
            maxdof = np.max(element2dofs)
            dofvalues = np.zeros(maxdof + 1)

        M_csr = self.C_csr.copy()

        for i, nodeindices in enumerate(connectivity):
            X_local = nodes[nodeindices, :].reshape(-1)
            globaldofindices = element2dofs[i]
            u_local = dofvalues[globaldofindices]
            M_local = ele_objects[i].m_int(X_local, u_local, t)
            fill_csr_matrix(M_csr.indptr, M_csr.indices, M_csr.data, M_local, globaldofindices)
        return M_csr

    def assemble_k_f_S_E(self, nodes, ele_objects, connectivity, element2dofs, elements_on_node, dofvalues=None, t=0):
        """
        Assemble the stiffness matrix with stress recovery of the given mesh and element.

        Parameters
        ----------
        nodes : ndarray
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivity : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes ndarray
        element2dofs : list of ndarrays
            Mapping the elements to their global dofs
        elements_on_node : ndarray(int)
            ndarray containing number of elements that are assembled belonging to a node
        dofvalues : ndarray
            current values of all dofs (at time t)
        t : float
            time. Default: 0.

        Returns
        --------
        K : sparse.csr_matrix
            unconstrained assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            unconstrained assembled force vector
        S : ndarray
            unconstrained assembled stress tensor
        E : ndarray
            unconstrained assembled strain tensor
        """

        if dofvalues is None:
            maxdof = np.max(element2dofs)
            dofvalues = np.zeros(maxdof + 1)

        # Allocate K and f
        K_csr = self.C_csr.copy()
        f_glob = np.zeros(K_csr.shape[1])
        no_of_nodes = nodes.shape[0]
        E_global = np.zeros((no_of_nodes, 6))
        S_global = np.zeros((no_of_nodes, 6))

        # Loop over all elements
        # (i - element index, indices - DOF indices of the element)
        for i, nodeindices in enumerate(connectivity):
            # X - undeformed positions of the i-th element
            X_local = nodes[nodeindices, :].reshape(-1)
            # global dofs of the i-th element
            globaldofindices = element2dofs[i]
            # displacements of the i-th element
            u_local = dofvalues[globaldofindices]
            # computation of the element tangential stiffness matrix and nonlinear force
            K_local, f_local, S_local, E_local = ele_objects[i].k_f_S_E_int(X_local, u_local, t)
            # adding the local force to the global one
            f_glob[globaldofindices] += f_local
            # this is equal to K_csr[globaldofindices, globaldofindices] += K_local
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K_local, globaldofindices)
            E_global[nodeindices, :] += E_local
            S_global[nodeindices, :] += S_local

        # Correct strains such, that average is taken at the elements
        E_global = (E_global.T/elements_on_node).T
        S_global = (S_global.T/elements_on_node).T
        return K_csr, f_glob, S_global, E_global
