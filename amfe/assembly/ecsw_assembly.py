#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
ECSW assembly.

Provides Assembly routines for the ECSW Assembly
"""

import logging

import numpy as np
import pandas as pd

from .structural_assembly import StructuralAssembly
from .tools import fill_csr_matrix

__all__ = [
    'EcswAssembly'
]


class EcswAssembly(StructuralAssembly):
    """
    Class handling assembly of elements for ECSW

    Attributes
    ----------
    logger : logging.Logger
        Logger object for pushing logging messages
    indices : numpy.array
        dtype = int, array containing the row indices of the arrays that are passed for assembly, that have nonzero
        weights.
    weights : numpy.array
        dtype = float, array containing the nonzero weights for ECSW Assembly
    """

    def __init__(self, weights, indices):
        """

        Parameters
        ----------
        weights : numpy.array
            array containing the nonzero weights for ECSW Assembly
        indices : numpy.array
            dtype = int, localization indices of the elements that have zero weights. Example: Assume, the elements
            [ele1, ele2, ele3, ele4, ele5] are passed, but only ele1, ele4 and ele5 have nonzero weights.
            Then the indices array is np.array([0, 3, 4], dtype=int)
        """
        super().__init__()
        self.logger = logging.getLogger('amfe.assembly.EcswAssembly')
        self.weights = np.array(weights)
        self.indices = np.array(indices, dtype=int)

    def assemble_k_and_f(self, nodes_df, ele_objects, connectivities, elements2dofs, dofvalues=None, t=0., K_csr=None,
                         f_glob=None):
        """
        Assemble the tangential stiffness matrix and nonliner internal or external force vector.

        This method can be used for assembling K_int and f_int or for assembling K_ext and f_ext depending on which
        ele_objects and connectivities are passed

        Parameters
        ----------
        nodes_df : pandas.DataFrame
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivities : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes ndarray
        elements2dofs : list of ndarrays
            Mapping the elements to their global dofs
        dofvalues : ndarray
            current values of all dofs (at time t)
        t : float
            time. Default: 0.
        K_csr : csr_matrix (optional)
            A preallocated csr_matrix can be passed for faster assembly
        f_glob : numpy.array (optional)
            A preallocated numpy.array can be passede for faster assembly

        Returns
        --------
        K : csr_matrix
            global stiffness matrix
        f : ndarray
            global internal force vector

        """

        if dofvalues is None:
            maxdof = np.max(elements2dofs)
            dofvalues = np.zeros(maxdof + 1)

        if K_csr is None:
            no_of_dofs = np.max(elements2dofs) + 1
            K_csr = self.preallocate(no_of_dofs, elements2dofs)

        if f_glob is None:
            f_glob = np.zeros(K_csr.shape[1], dtype=float)

        K_csr.data[:] = 0.0
        f_glob[:] = 0.0

        # loop over all elements
        # (i - element index, indices - DOF indices of the element)
        for weight, index in zip(self.weights, self.indices):
            # X - undeformed positions of the i-th element
            X_local = nodes_df.loc[connectivities[index], :].values.reshape(-1)
            # displacements of the i-th element
            u_local = dofvalues[elements2dofs[index]]
            # computation of the element tangential stiffness matrix and nonlinear force
            K_local, f_local = ele_objects[index].k_and_f_int(X_local, u_local, t)
            # adding the local force to the global one
            f_glob[elements2dofs[index]] += weight*f_local
            # this is equal to K_csr[globaldofindices, globaldofindices] += K_local
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, weight*K_local, elements2dofs[index])
        return K_csr, f_glob

    def assemble_k_f_S_E(self, nodes_df, ele_objects, connectivities, elements2dofs, elements_on_node, dofvalues=None,
                         t=0, K_csr=None, f_glob=None):
        """
        Assemble the stiffness matrix with stress recovery of the given mesh and element.

        Parameters
        ----------
        nodes_df : pandas.DataFrame
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivities : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes pandas.DataFrame
        elements2dofs : list of ndarrays
            Mapping the elements to their global dofs
        elements_on_node : pandas.DataFrame
            DataFrame containing number of elements that are assembled belonging to a node
        dofvalues : ndarray
            current values of all dofs (at time t) ordered by the dofnumbers given by elements2dof list
        t : float
            time. Default: 0.
        K_csr : csr_matrix (optional)
            A preallocated csr_matrix can be passed for faster assembly
        f_glob : numpy.array
            A preallocated numpy.array can be passed for faster assembly

        Returns
        --------
        K : csr_matrix
            global stiffness matrix
        f : ndarray
            global internal force vector
        S : pandas.DataFrame
            unconstrained assembled stress tensor
        E : pandas.DataFrame
            unconstrained assembled strain tensor
        """

        if dofvalues is None:
            maxdof = np.max(elements2dofs)
            dofvalues = np.zeros(maxdof + 1)

        # Allocate K and f
        if K_csr is None:
            no_of_dofs = np.max(elements2dofs) + 1
            K_csr = self.preallocate(no_of_dofs, elements2dofs)

        if f_glob is None:
            f_glob = np.zeros(K_csr.shape[1], dtype=float)

        f_glob[:] = 0.0
        K_csr.data[:] = 0.0

        no_of_nodes =len(nodes_df.index)
        S = pd.DataFrame(data=np.zeros((no_of_nodes, 6), dtype=float),
                         columns=['Sxx', 'Syy', 'Szz', 'Syz', 'Sxz', 'Sxy'], index=nodes_df.index)
        E = pd.DataFrame(data=np.zeros((no_of_nodes, 6), dtype=float),
                         columns=['Exx', 'Eyy', 'Ezz', 'Eyz', 'Exz', 'Exy'], index=nodes_df.index)

        # Loop over all elements
        # (i - element index, indices - DOF indices of the element)
        for weight, index in zip(self.weights, self.indices):
            # X - undeformed positions of the i-th element
            X_local = nodes_df.loc[connectivities[index], :].values.reshape(-1)
            # displacements of the i-th element
            u_local = dofvalues[elements2dofs[index]]
            # computation of the element tangential stiffness matrix and nonlinear force
            K_local, f_local, S_local, E_local = ele_objects[index].k_f_S_E_int(X_local, u_local, t)
            # adding the local force to the global one
            f_glob[elements2dofs[index]] += weight*f_local
            # this is equal to K_csr[globaldofindices, globaldofindices] += K_local
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, weight*K_local, elements2dofs[index])
            E.loc[connectivities[index], :] += weight*E_local
            # QUESTION: SHALL THIS BE ALSO WEIGHTED?
            S.loc[connectivities[index], :] += weight*S_local

        # Correct strains such, that average is taken at the elements
        E = E.divide(E.join(elements_on_node)['elements_on_node'], axis=0)
        S = S.divide(S.join(elements_on_node)['elements_on_node'], axis=0)
        return K_csr, f_glob, S, E
