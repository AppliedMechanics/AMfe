#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Abstract class of assembly algorithms.
"""

__all__ = [
    'Assembly'
]

import time
import abc

import numpy as np
from scipy.sparse import csr_matrix


class Assembly:
    """
    Super class for all assemblies providing observer utilities.
    """

    def __init__(self):
        self._observers = list()
        return

    def add_observer(self, observer):
        self._observers.append(observer)
        return

    def remove_observer(self, observer):
        self._observers.remove(observer)
        return

    def notify(self):
        for observer in self._observers:
            observer.update(self)
        return

    def update(self, obj):
        pass

    @abc.abstractmethod
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
        C_csr : csr_matrix
            Empty csr_matrix for preallocation

        Notes
        -----
        This pre-allocation routine can take some while for large matrices. Furthermore it is not implemented
        memory-efficient, so for large systems and low RAM this might become an issue...
        """
        pass

    @abc.abstractmethod
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

        Returns
        --------
        K : csr_matrix
            global stiffness matrix
        f : ndarray
            global internal force vector

        Examples
        ---------
        TODO
        """
        pass

    @abc.abstractmethod
    def assemble_m(self, nodes_df, ele_objects, connectivities, elements2dofs, dofvalues=None, t=0, M_csr=None):
        """
        Assembles the mass matrix of the given mesh and element.

        Parameters
        ----------
        nodes_df : pandas.Dataframe
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
        M_csr : csr_matrix
            if a preallocated csr_matrix for M exist, it can be passed here

        Returns
        --------
        M : sparse.csr_matrix
            unconstrained assembled mass matrix in sparse matrix csr-format.

        Examples
        ---------
        TODO
        """
        pass

    @abc.abstractmethod
    def assemble_k_f_S_E(self, nodes_df, ele_objects, connectivities, elements2dofs, elements_on_node, dofvalues=None, t=0, K_csr=None, f_glob=None):
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
        pass

    @abc.abstractmethod
    def assemble_f_ext(self, nodes_df, ele_objects, connectivities, elements2dofs, dofvalues=None, t=0., f_glob=None):
        """
        Assemble the external force vector.

        This method can be used for assembling f_ext

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
        f_glob : ndarray
            preallocated ndarray

        Returns
        --------
        f_ext : ndarray
            external force

        Examples
        ---------
        TODO
        """
        pass
