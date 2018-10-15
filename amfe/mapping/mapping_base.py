# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np

__all__ = ['Mapping']


class Mapping:
    def __init__(self, fields, no_of_nodes, connectivity, dofs_by_node, dofs_by_element, **kwargs):
        """

        Parameters
        ----------
        fields : tuple
            tuple with strings that describe the field that shall be mapped, e.g. ('ux', 'uy', 'uz', 'T')
            for a 3D displacement and Temperature field
        no_of_nodes : int
            number of nodes that shall be mapped by the mapping algorithm
        connectivity : ndarray
            iterable containing the nodeidxes (rowindices) of the connectivity of the elements
        dofs_by_node : iterable
            iterable containing the dofs as integers per node according to the field tuple fields
            e.g. if a node has dofs 'uy' and 'T' with given fields tuple ('ux', 'uy', 'uz', 'T') then
            the dofs_by_node entry contains (1,3)
        dofs_by_element : iterable
            same as dofs_by_node but for each element
        kwargs : dict
            keyword value list for future implementations
        """
        self._fields = fields
        self._field2idx = {ele: idx for idx, ele in enumerate(fields)}
        self._nodes2global, self._elements2global = self._get_standard_mapping(fields, no_of_nodes, connectivity,
                                                                               dofs_by_node, dofs_by_element, **kwargs)

    @property
    def nodes2global(self):
        return self._nodes2global

    @nodes2global.setter
    def nodes2global(self, nodes2global):
        self._nodes2global = nodes2global

    @property
    def elements2global(self):
        return self._elements2global

    @elements2global.setter
    def elements2global(self, elements2global):
        self._elements2global = elements2global

    @property
    def field2idx(self):
        return self._field2idx

    def get_dof_by_nodeidx(self, idx, field):
        """
        Returns a the global dof number of a node with certain nodeidx and given field

        Parameters
        ----------
        idx : int
            number of node index in nodes array
        field : str
            string describing the field (e.g. 'ux', 'uy', 'T', ....)

        Returns
        -------
        dof : int
            global dof number
        """
        return self._nodes2global[idx, self._field2idx[field]]

    def get_dofs_by_nodeidxs(self, nodeidxs, fields):
        """
        Returns the global dofs associated with a given node-row-index and a direction x, y or z

        Parameters
        ----------
        nodeidxs : iterable
            Row indices of nodes where one wants to know their global dofs
        fields : tuple
            tuple with strings that describe the fields, the global dofs are asked for

        Returns
        -------
        dofs : ndarray
            array with global dofs in order (node1field1, node1field2, node1field3,... node2field1, node2field2, ...)
        """
        cols = np.array([self.field2idx[field] for field in fields], dtype=int)
        rows = np.array(nodeidxs, dtype=int)

        return self.nodes2global[np.ix_(rows, cols)].reshape(-1)

    def _get_standard_nodes_mapping(self, fields, no_of_nodes, dofs_by_node):
        ndof = len(fields)
        nodes2global = np.ones((no_of_nodes, ndof), dtype=int)*-1
        next_free_lowest_dof_id = 0
        for idx, dofs in enumerate(dofs_by_node):
            nodes2global[idx, dofs] = next_free_lowest_dof_id + np.array(dofs, dtype=int)
            next_free_lowest_dof_id += len(dofs)
        return nodes2global

    def _get_standard_element_mapping(self, nodes2global, connectivity, dofs_by_element):
        """
        Compute the standard mapping between elements, their local dofs and the global dofs.

        The element_mapping is a list, where every element of the list denotes
        the global dofs of the element in the correct order.

        Parameters
        ----------
        nodes2global : ndarray
            ndarray containing the global dof indices for local nodeidxs (rows) and fields (columns)
        connectivity : list
            list with ndarrays describing the connectivity (topology) of volume elements in the mesh
            (row-indices of a node array)
        dofs_by_element : ndarray
            ndarray containing the dofs as integers per element according to the field tuple fields
            e.g. if a 3-node element has dofs 'uy' and 'T' for each node with given fields
            tuple ('ux', 'uy', 'uz', 'T') then the dofs_by_element entry contains (1,3,1,3,1,3)

        Returns
        -------
        element_mapping : ndarray
            list with global dofs for elements
        """

        element_mapping = [np.stack([nodes2global[nodeidx, dof].reshape(-1) for
                                     nodeidx, dof in zip(nodeidxs, dofs)]) for nodeidxs, dofs in
                                     zip(connectivity, dofs_by_element)]
        return element_mapping

    def _get_standard_mapping(self, fields, no_of_nodes, connectivity, dofs_by_node, dofs_by_element, **kwargs):
        """
        Computes the mapping according to a certain algorithm.

        This method can be overwritten by subclasses to get other algorithms to get a mapping for elements
        and nodes.


        Parameters
        ----------
        fields : iterable
            contains strings that describe the fieldnames
        no_of_nodes : int
            number of nodes that shall be mapped
        connectivity : iterable
            contains nodeidxs for each element
        dofs_by_node : ndarray
            contains dofs [0,1,2..] in fields list for each node, ordered by nodeidx
        **kwargs : dict
            keyword-value arguments that can be passed to mapping algorithm
            (important for subclassing if special algorithms need special parameters)

        Returns
        -------
        nodes2global : ndarray
            ndarray containing the global dof indices for local nodeidxs (rows) and fields (columns)
        elements2global : ndarray
            ndarray containing the global dof indices for each element
        """
        nodes2global = self._get_standard_nodes_mapping(fields, no_of_nodes, dofs_by_node)
        elements2global = self._get_standard_element_mapping(nodes2global, connectivity, dofs_by_element)

        return nodes2global, elements2global
