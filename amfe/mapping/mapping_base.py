# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
import pandas as pd

__all__ = ['Mapping']


class Mapping:
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

    def get_dof_by_nodeid(self, nodeid, field):
        """
        Returns a the global dof number of a node with certain nodeid and given field

        Parameters
        ----------
        nodeid : int
            number of node id
        field : str
            string describing the field (e.g. 'ux', 'uy', 'T', ....)

        Returns
        -------
        dof : int
            global dof number
        """
        return self._nodal2global.loc[nodeid, field]

    def get_dofs_by_nodeidxs(self, nodeids, fields):
        """
        Returns the global dofs associated with a given node-row-index and a direction x, y or z

        Parameters
        ----------
        nodeids : iterable
            Nodeids where one wants to know their global dofs
        fields : tuple
            tuple with strings that describe the fields, the global dofs are asked for

        Returns
        -------
        dofs : ndarray
            array with global dofs. Rows = nodeids, columns = fields
        """
        return self._nodal2global.loc[nodeids, fields].values

    def _set_standard_mapping(self, fields, nodeids, connectivity, dofs_by_element, **kwargs):
        """
        Computes the mapping according to a certain algorithm.

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
        # make empty pandas Dataframe for nodes2global
        data = -1*np.ones(len(nodeids), dtype=int)
        self._nodal2global = pd.DataFrame({key: data for key in fields}, index=nodeids)
        # allocate list for elements2global
        self._elements2global = [None]*len(connectivity)

        # collect node dofs
        current_global = 0
        # iterate over all elements
        for index, (element_connectivity, element_dofinfos) in enumerate(zip(connectivity, dofs_by_element)):
            # iterate over dofs of element
            global_dofs_for_element = []
            for localdofnumber, dofinfo in enumerate(element_dofinfos):
                # investigate current dof
                localnodenumber = dofinfo[1]  # or local elementdof number
                field = dofinfo[2]  # physical field (e.g. 'ux')
                # check if dof is a nodal dof
                if dofinfo[0] == 'N':
                    nodeid = element_connectivity[localnodenumber]
                    # check if node already has a global dof for this field
                    stored_global = self._nodal2global.loc[nodeid, field]
                    if stored_global == -1:
                        # set a global dof for this node and field combination
                        self._nodal2global.loc[nodeid, field] = current_global
                        global_dofs_for_element.append(current_global)
                        # increment current_global dof number
                        current_global += 1
                    else:
                        global_dofs_for_element.append(stored_global)
                # elif check if dof is elemental dof
                elif dofinfo[0] == 'E':
                    raise NotImplementedError('The mapping for elemental degrees of freedom is not implemented in'
                                              'this mapping class')
                else:
                    raise ValueError('Doftype must be E or N')
            # Get global dof numbers of element
            self._elements2global[index] = np.array(global_dofs_for_element, dtype=int)
