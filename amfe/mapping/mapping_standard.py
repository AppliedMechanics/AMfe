# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
import pandas as pd

from .mapping_base import MappingBase

__all__ = ['StandardMapping']


class StandardMapping(MappingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_standard_mapping(self, fields, nodeids, elementids, connectivity, dofs_by_element, **kwargs):
        # make empty pandas Dataframe for nodes2global
        data = -1*np.ones(len(nodeids), dtype=int)
        self._nodal2global = pd.DataFrame({key: data for key in fields}, index=nodeids)
        # allocate list for elements2global
        self._elements2global = pd.DataFrame([None]*len(elementids), index=elementids, columns=['global_dofs'])

        # collect node dofs
        current_global = 0
        # iterate over all elements
        for index, (elementid, element_connectivity, element_dofinfos) in enumerate(zip(elementids, connectivity,
                                                                                        dofs_by_element)):
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
            self._elements2global.loc[elementid]['global_dofs'] = np.array(global_dofs_for_element, dtype=int)
