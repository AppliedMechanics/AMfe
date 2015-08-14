#!/bin/env python
# -*- coding: utf-8 -*-
"""
Basic assembly module for the finite element code. Assumes to have all elements in the inertial frame.
Created on Tue Apr 21 11:13:52 2015

@author: Johannes Rutzmoser
"""


import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg

import multiprocessing as mp
from multiprocessing import Pool


#import mesh
#mesh.Mesh.no_of_dofs

class Assembly():
    '''
    Class for the more fancy assembly of meshes with heterogeneous elements.
    '''
    def __init__(self, mesh, element_class_dict):
        '''
        Parameters
        ----
        mesh : instance of the Mesh-class

        element_class_dict : dict
            dict where the official keyword and the the Element-objects are linked

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        self.mesh = mesh
        self.element_class_dict = element_class_dict
        self.save_stresses = False
        pass



    def _assemble_matrix(self, u, decorated_matrix_func):
        '''
        Assembly routine for any matrices.

        Parameters
        ----------
        u : ndarray
            global displacement vector; if set to None, u will be assumed to be zero displacement
        decorated_matrix_func : function
            function with input variables (X, u_local, k, global_element_indices)
            the input variables are
            -----------------------
            X : ndarray
                local coordinates in the reference configuration
            u_local : ndarray
                local displacements
            k : int
                global index of the element (is needed in order to find the element type out of a global list)
            global_element_indices : ndarray
                global indices of the element (is needed for the force assembly)


        Returns
        -------
        Matrix : coo sparse array
            sparse assembled array in coo format
        element_props : list
            list of all saved element props that are pumped out of the decorated matrix func

        Note
        ----


        '''
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        self.element_props_global = []
        node_dof = self.mesh.node_dof
        if u is None:
            u = np.zeros(self.mesh.no_of_dofs)
        # loop over all elements
        for k, element in enumerate(self.mesh.elements):
            # coordinates of element in 1-D array
            X = np.array([self.mesh.nodes[i] for i in element]).reshape(-1)
            # corresponding global coordinates of element in 1-D array
            global_element_indices = np.array([(np.arange(node_dof) + node_dof*i)  for i in element]).reshape(-1)
            u_local = u[global_element_indices]
            # evaluation of element matrix
            element_matrix, element_props = decorated_matrix_func(X, u_local, k, global_element_indices)
            self.row = np.zeros(element_matrix.shape)
            # build a matrix with constant columns and the rows representing the global_element_indices
            self.row[:,:] = global_element_indices
            self.row_global.append(self.row.reshape(-1))
            self.col_global.append((self.row.T).reshape(-1))
            # Attention! Here, the matrix is not copied! Make sure that the
            # element matrix is an object of its own
            self.vals_global.append(element_matrix.reshape(-1))
            self.element_props_global.append(element_props)
        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo, self.element_props_global


    def assemble_k(self, u=None):
        '''
        Assembles the stiffness matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        K : ndarray
            unconstrained assembled stiffness matrix in sparse matrix coo-format.
        '''

        def decorated_k_func(X, u_local, k, global_element_indices=None):
            element = self.element_class_dict[self.mesh.elements_type[k]]
            k_local = element.k_int(X, u_local)
            if self.save_stresses:
                stresses = element.S_voigt
            else:
                stresses = ()
            return k_local, stresses

        K, self.stress_list = self._assemble_matrix(u, decorated_k_func)
        return K


    def assemble_m(self, u=None):
        '''
        Assembles the mass matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        M : ndarray
            unconstrained assembled mass matrix in sparse matrix coo-format.

        Examples
        ---------
        TODO
        '''
        def decorated_m_func(X, u_local, k, global_element_indices=None):
            return self.element_class_dict[self.mesh.elements_type[k]].m_int(X, u_local), ()

        M, _ = self._assemble_matrix(u, decorated_m_func)
        return M


    def assemble_f(self, u):
        '''
        Assembles the force vector of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        f : ndarray
            unconstrained force vector

        '''
        self.global_force = np.zeros(self.mesh.no_of_dofs)
        node_dof = self.mesh.node_dof
        for k, element in enumerate(self.mesh.elements):
            X = np.array([self.mesh.nodes[i] for i in element]).reshape(-1)
            # global coordinates of element
            global_element_indices = np.array([(np.arange(node_dof) + node_dof*i)  for i in element]).reshape(-1)
            self.global_force[global_element_indices] += \
                self.element_class_dict[self.mesh.elements_type[k]].f_int(X, u[global_element_indices])
        return self.global_force

    def assemble_k_and_f(self, u=None):
        '''
        Assembles the tangential stiffness matrix and the force matrix in one
        run as it is very often needed by an implicit integration scheme.

        Takes the advantage, that some element properties only have to be
        computed once.

        Parameters
        -----------
        u : ndarray, optional
            displacement of the unconstrained system in voigt notation

        Returns
        --------
        K : ndarray
            tangential stiffness matrix
        f : ndarray
            nonlinear force vector

        Examples
        ---------
        TODO
        '''
        def decorated_f_and_k_func(X, u_local, k, global_element_indices):
            element = self.element_class_dict[self.mesh.elements_type[k]]
            k_local, f_local = element.k_and_f_int(X, u_local)
            self.global_force[global_element_indices] += f_local
            if self.save_stresses:
                stresses = element.S_voigt
            else:
                stresses = ()
            return k_local, stresses

        self.stress_list = []
        self.global_force = np.zeros(self.mesh.no_of_dofs)
        K, _ = self._assemble_matrix(u, decorated_f_and_k_func)
        return K, self.global_force





class PrimitiveAssembly():
    '''
    Assembly class working directly on the tables of node coordinates and element nodes

    Came historically before more advanced assembly routines and have the status of being for test cases
    '''

    # Hier muessen wir uns mal genau ueberlegen, was alles dem assembly uebergeben werden soll
    # ob das ganze Mesh, oder nur ein paar Attribute
    def __init__(self, nodes=None, elements=None, matrix_function=None, node_dof=2, vector_function=None):
        '''
        Verlangt ein dreispaltiges Koordinatenarray, indem die Koordinaten in x, y, und z-Koordinaten angegeben sind
        Anzahl der Freiheitsgrade für einen Knotenfreiheitsgrad: node_dof gibt an, welche Koordinaten verwendet werden sollen;
        Wenn mehr Koordinaten pro Knoten nötig sind (z.B. finite Rotationen), werden Nullen hinzugefügt
        '''
        self.nodes = nodes
        self.elements = elements
        self.matrix_function = matrix_function
        self.vector_function = vector_function
        self.node_dof = node_dof

        self.row_global = []
        self.col_global = []
        self.vals_global = []

        self.no_of_nodes = len(self.nodes)
        self.no_of_elements = len(self.elements)
        self.no_of_dofs = self.no_of_nodes*self.node_dof
        self.no_of_element_nodes = len(self.elements[0])

        self.ndof_global = self.no_of_dofs
        pass


    def assemble_matrix(self, u=None):
        '''
        assembliert die matrix_function für die Ursprungskonfiguration X und die Verschiebung u.
        '''
        # deletion of former variables
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        # number of dofs per element (6 for triangle since no_of_element_nodes = 3 and node_dof = 2)
        ndof_local = self.no_of_element_nodes*self.node_dof
        # preset for u_local; necessary, when u=None
        u_local = np.zeros(ndof_local)

        for element in self.elements:
            # Koordinaten des elements
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            # element_indices have to be corrected in order respect the dimensions
            element_indices = np.array([[self.node_dof*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            if u is not None:
                u_local = u[element_indices]
            element_matrix = self.matrix_function(X, u_local)
            row = np.zeros((ndof_local, ndof_local))
            row[:,:] = element_indices
            self.row_global.append(row.reshape(-1))
            self.col_global.append((row.T).reshape(-1))
            self.vals_global.append(element_matrix.reshape(-1))

        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo

    def assemble_vector(self, u):
        '''
        Assembliert die Force-Function für die Usprungskonfiguration X und die Verschiebung u
        '''
        global_force = np.zeros(self.ndof_global)
        for element in self.elements:
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            element_indices = np.array([[2*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            global_force[element_indices] += self.vector_function(X, u[element_indices])
        return global_force







class MultiprocessAssembly():
    '''
    Klasse um schnell im Multiprozess zu assemblieren; Verteilt die Assemblierung auf alle Assemblierungsklassen und summiert die anschließend alles auf
    - funktioniert nicht so schnell, wie ich es erwartet hätte; genauere Analysen bisher noch nicht vorhanden, da profile-Tool nich zuverlässig für multiprocessing-Probleme zu funktionieren scheint.
    - ACHTUNG: Diese Klasse ist derzeit nicht in aktiver Nutzung. Möglicherweise macht es Sinn, diese Klasse zu überarbeiten, da sich die gesamte Programmstruktur gerade noch ändert.
    '''
    def __init__(self, assembly_class, list_of_matrix_functions, nodes_array, element_array):
        '''
        ???
        '''
        self.no_of_processes = len(list_of_matrix_functions)
        self.nodes_array = nodes_array
        self.element_array = element_array
        self.list_of_matrix_functions = list_of_matrix_functions
        domain_size = self.nodes_array.shape[0]//self.no_of_processes
        element_domain_list = []
        for i in range(self.no_of_processes - 1):
            element_domain_list.append(self.element_array[i*domain_size:(i+1)*domain_size,:])
        element_domain_list.append(self.element_array[(i+1)*domain_size:,:]) # assemble last domain to the end in order to consider flooring above
        self.assembly_class_list = [assembly_class(self.nodes_array, element_domain_list[i], matrix_function) for i, matrix_function in enumerate(list_of_matrix_functions)]
        pass

    def assemble(self):
        '''
        assembles the mesh with a multiprocessing routine
        '''
        pool = mp.Pool(processes=self.no_of_processes)
        results = [pool.apply_async(assembly_class.assemble) for assembly_class in self.assembly_class_list]
        matrix_coo_list = [j.get() for j in results]
        row_global = np.array([], dtype=int)
        col_global = np.array([], dtype=int)
        data_global = np.array([], dtype=float)
        for matrix_coo in matrix_coo_list:
            row_global = np.append(row_global, matrix_coo.row)
            col_global = np.append(col_global, matrix_coo.col)
            data_global = np.append(data_global, matrix_coo.data)
        matrix_coo = sp.sparse.coo_matrix((data_global, (row_global, col_global)), dtype=float)
        return matrix_coo



