# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
import pandas as pd
import logging

from .mapping_base import MappingBase

__all__ = ['StandardMapping']


class StandardMapping(MappingBase):
    """
    Mapping Class providing a Standard Mapping

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for logging
    _nodal2global : pandas.DataFrame
        DataFrame containing Mapping Information of the nodal dofs
        index: nodeid
        columns: field strings
        values: global dof ids
    _elements2global : pandas.DataFrame
        DataFrame containing Mapping Information for any kind of entity.
        In this case it is the mapping of the local dofs of elements to global dofs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger('amfe.mapping.StandardMapping')

    def _set_standard_mapping(self, fields, nodeids, connectivity, dofs_by_element, callbacks, callbackargs, **kwargs):
        """
        Compute the mapping attributes for a 'Standard' Mapping

        Parameters
        ----------
        fields : list of strings
            Describing the physical fields thath shall be mapped (e.g. ['ux', 'uy', 'uz', 'T']
        nodeids : numpy.array
            Array containing the nodeids of the nodes that shall be mapped
        connectivity : numpy.array
            Array containing the connectivity information of the elements that shall be mapped
        dofs_by_element : numpy.array
            Array containing the tuples with the dofs information of the elements
        callbacks : numpy.array
            Array containing pointers for callback functions that are called after a global dof entry has been
            associated with the element
        callbackargs : list
            argument list, that are passed to the callback function that is called after a global dof entry has been
            associated with the element
        kwargs : None
            not used here

        Returns
        -------
        None
        """
        # make empty pandas Dataframe for nodes2global
        data = -1*np.ones(len(nodeids), dtype=int)
        self._nodal2global = pd.DataFrame({key: data for key in fields}, index=nodeids)
        # allocate list for elements2global
        self._elements2global = pd.DataFrame([None]*len(connectivity), columns=['global_dofs'])

        # collect node dofs
        current_global = 0
        no_of_elements = len(connectivity)
        # iterate over all elements
        for index, (element_connectivity, element_dofinfos, callback, callbackarg) in enumerate(zip(connectivity,
                                                                                                    dofs_by_element,
                                                                                                    callbacks,
                                                                                                    callbackargs)):

            print('Added element {:10d} of {:10d}'.format(index, no_of_elements))
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
                    stored_global = self._nodal2global.at[nodeid, field]
                    if stored_global == -1:
                        # set a global dof for this node and field combination
                        self._nodal2global.at[nodeid, field] = current_global
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
            self._elements2global.at[index, 'global_dofs'] = np.array(global_dofs_for_element, dtype=int)
            callback(index, callbackarg)
