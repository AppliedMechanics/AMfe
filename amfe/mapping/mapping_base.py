# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from collections.abc import Iterable
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = ['MappingBase']


class MappingBase(ABC):
    def __init__(self, fields, nodeids, connectivity, dofs_by_element, **kwargs):
        """

        Parameters
        ----------
        fields : tuple
            tuple with strings that describe the field that shall be mapped, e.g. ('ux', 'uy', 'uz', 'T')
            for a 3D displacement and Temperature field
        nodeids : array
            array containing the nodeids that shall be mapped
        connectivity : ndarray
            iterable containing nodeids of the connectivity in each element
        dofs_by_element : iterable
            iterable containing the dofs as strings per element
            e.g. [(('N', 0, 'ux'), ('N', 0, 'uy'), ('E', 0, 'T'), ('N', 1, 'ux')), ( ..same for 2nd element ), ... )
        kwargs : dict
            keyword value list for future implementations
        """
        self._fields = fields

        # create empty DataFrame
        data = -1*np.ones(len(nodeids), dtype=int)
        self._nodal2global = pd.DataFrame({key: data for key in fields}, index=nodeids)
        # TODO: Insert an elemental2global property to store elemental dofs

        self._elements2global = []*len(connectivity)
        # update nodes2global and elements2global
        self._set_standard_mapping(fields, nodeids, connectivity, dofs_by_element, **kwargs)

    @property
    def nodal2global(self):
        return self._nodal2global

    @nodal2global.setter
    def nodal2global(self, nodal2global):
        self._nodal2global = nodal2global

    @property
    def elements2global(self):
        return self._elements2global

    @elements2global.setter
    def elements2global(self, elements2global):
        self._elements2global = elements2global

    def get_dofs_by_nodeids(self, nodeids, fields):
        """
        Returns the global dofs associated with a given node-row-index and a direction x, y or z

        Parameters
        ----------
        nodeids : iterable
            Nodeids where one wants to know their global dofs
        fields : tuple
            tuple with strings that describe the fields, the global dofs are asked for (e.g. 'ux', 'uy', 'T', ....)

        Returns
        -------
        dofs : ndarray
            array with global dofs. Rows = nodeids, columns = fields
        """
        if not isinstance(nodeids, Iterable):
            nodeids = [nodeids]
        return self._nodal2global.loc[nodeids, fields].values

    def update_mapping(self, fields, nodeids, connectivity, dofs_by_element, **kwargs):
        """
        Update the mapping (nodal2global and elements2global)

        Parameters
        ----------
        fields : tuple
            tuple with strings that describe the field that shall be mapped, e.g. ('ux', 'uy', 'uz', 'T')
            for a 3D displacement and Temperature field
        nodeids : array
            array containing the nodeids that shall be mapped
        connectivity : ndarray
            iterable containing nodeids of the connectivity in each element
        dofs_by_element : iterable
            iterable containing the dofs as strings per element
            e.g. [(('N', 0, 'ux'), ('N', 0, 'uy'), ('E', 0, 'T'), ('N', 1, 'ux')), ( ..same for 2nd element ), ... )
        kwargs : dict
            keyword value list for future implementations

        Returns
        -------
        None
        """
        self._set_standard_mapping(fields, nodeids, connectivity, dofs_by_element, **kwargs)

    @abstractmethod
    def _set_standard_mapping(self, fields, nodeids, connectivity, dofs_by_element, **kwargs):
        """
        Computes the mapping according to a certain algorithm.

        This private method must be overwritten by subclasses of Mapping (Template Pattern).

        This method can be overwritten by subclasses to get other algorithms to get a mapping for elements
        and nodes.


        Parameters
        ----------
        fields : iterable
            contains strings that describe the fieldnames
        nodeids : array
            array containing the nodeids that shall be mapped
        connectivity : ndarray
            iterable containing nodeids of the connectivity in each element
        dofs_by_element : iterable
            iterable containing the dofs as strings per element
            e.g. [(('N', 0, 'ux'), ('N', 0, 'uy'), ('E', 0, 'T'), ('N', 1, 'ux')), ( ..same for 2nd element ), ... )
        kwargs : dict
            keyword value list for future implementations
            (important for subclassing if special algorithms need special parameters)

        Returns
        -------
        None
        """
        raise NotImplementedError('The _set_standard_mapping method must be implemented in subclasses')

