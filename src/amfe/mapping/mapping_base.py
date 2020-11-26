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
    def __init__(self):
        self._nodal2global = None

        self._elements2global = pd.DataFrame({'global_dofs': []})

    def __str__(self):
        """
        Returns information about the StandardMapping when using print(instance)

        Returns
        -------
        info_of_StandardMapping: string
        #     no_of_dofs, no_of_mapped_elements, address_RAM
        """
        return "--- Info about StandardMapping ---\nNo of dof: {0:,>}\nNo of mapped elements: {1}\n" \
               "Address in RAM: {2}\n"\
               .format(self.no_of_dofs, self.no_of_elements, id(self))

    @property
    def no_of_dofs(self):
        return len(np.unique(np.concatenate(self._elements2global['global_dofs'].values)))

    @property
    def no_of_elements(self):
        return len(self._elements2global)

    @property
    def nodal2global(self):
        return self._nodal2global

    @nodal2global.setter
    def nodal2global(self, nodal2global):
        self._nodal2global = nodal2global

    @property
    def elements2global(self):
        return self._elements2global['global_dofs'].values

    @elements2global.setter
    def elements2global(self, elements2global):
        self._elements2global = elements2global

    def get_dofs_by_ids(self, ids):
        return self._elements2global.loc[ids, 'global_dofs'].values

    def get_dofs_by_nodeids(self, nodeids, fields=('all')):
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
        if fields is ('all'):
            fields = self._nodal2global.keys()
        return self._nodal2global.loc[nodeids, fields].values

    def update_mapping(self, fields, nodeids, connectivity, dofs_by_element, callbacks, callbackargs, **kwargs):
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
            iterable containing nodeids of the connectivity in each element (same length as the elementids)
        dofs_by_element : iterable
            iterable containing the dofs as strings per element
            e.g. [(('N', 0, 'ux'), ('N', 0, 'uy'), ('E', 0, 'T'), ('N', 1, 'ux')), ( ..same for 2nd element ), ... )
        callbacks : list
            callback function with signature void: callback(int: id, *args) for writing the information about the mapping id
            that has been inserted in the dataframe of the mapping class
        callbackargs : list of lists
            lists containing arguments that are passd to the callbackfunction as *args
        kwargs : dict
            keyword value list for future implementations

        Returns
        -------
        None
        """
        self._set_standard_mapping(fields, nodeids, connectivity, dofs_by_element, callbacks, callbackargs, **kwargs)

    @abstractmethod
    def _set_standard_mapping(self, fields, nodeids, connectivity, dofs_by_element, callbacks, callbackargs, **kwargs):
        """
        Computes the mapping according to a certain algorithm.

        This private method must be overwritten by subclasses of Mapping (Template Pattern).

        This method can be overwritten by subclasses to get other algorithms to get a mapping for elements
        and nodes.


        Parameters
        ----------
        fields : tuple
            tuple with strings that describe the field that shall be mapped, e.g. ('ux', 'uy', 'uz', 'T')
            for a 3D displacement and Temperature field
        nodeids : array
            array containing the nodeids that shall be mapped
        connectivity : ndarray
            iterable containing nodeids of the connectivity in each element (same length as the elementids)
        dofs_by_element : iterable
            iterable containing the dofs as strings per element
            e.g. [(('N', 0, 'ux'), ('N', 0, 'uy'), ('E', 0, 'T'), ('N', 1, 'ux')), ( ..same for 2nd element ), ... )
        callbacks : list
            callback function with signature void: callback(int: id, *args) for writing the information about the mapping id
            that has been inserted in the dataframe of the mapping class
        callbackargs : list of lists
            lists containing arguments that are passd to the callbackfunction as *args
        kwargs : dict
            keyword value list for future implementations (important for subclassing)

        Returns
        -------
        None
        """
        raise NotImplementedError('The _set_standard_mapping method must be implemented in subclasses')

