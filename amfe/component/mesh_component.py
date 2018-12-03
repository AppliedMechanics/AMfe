#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from copy import deepcopy
import numpy as np
import pandas as pd

from amfe.mesh import Mesh
from amfe.mapping import StandardMapping
from .component_base import ComponentBase
from amfe.component.constants import ELEPROTOTYPEHELPERLIST
from amfe.neumann.neumann_manager import *

__all__ = ['MeshComponent']


class MeshComponent(ComponentBase):
    # The following class attributes must be overwritten by subclasses
    ELEMENTPROTOTYPES = dict(((element[0], None) for element in ELEPROTOTYPEHELPERLIST))

    def __init__(self, mesh=Mesh()):
        super().__init__()
        self._mesh = mesh
        self._mapping = StandardMapping()
        self._fields = None
        no_of_volume_elements = mesh.no_of_elements
        indices = mesh.el_df[mesh.el_df['is_boundary'] != True].index
        self._ele_obj_df = pd.DataFrame([None]*no_of_volume_elements, index=indices, columns=['ele_obj'])

        self._neumann = NeumannManager()

    # -- PROPERTIES --------------------------------------------------------------------------------------
    @property
    def ele_obj(self):
        return self._ele_obj_df['ele_obj'].values

    # -- ASSIGN MATERIAL METHODS -------------------------------------------------------------------------
    def assign_material(self, materialobj, propertynames, tag='_groups'):
        if tag == '_groups':
            eleids = self._mesh.get_elementids_by_groups(propertynames)
        elif tag == '_eleids':
            eleids = propertynames
        else:
            eleids = self._mesh.get_elementids_by_tags(propertynames)
        self._assign_material_by_eleids(materialobj, eleids)

    def _assign_material_by_eleids(self, materialobj, eleids):
        prototypes = deepcopy(self.ELEMENTPROTOTYPES)
        for prototype in prototypes.values():
            prototype.material = materialobj
        ele_shapes = self._mesh.get_ele_shapes_by_ids(eleids)
        self._ele_obj_df.loc[eleids, 'ele_obj'] = [prototypes[ele_shape] for ele_shape in ele_shapes]
        self._update_mapping()
        self._C_csr = self._assembly.preallocate(self._mapping.no_of_dofs, self._mapping.elements2global)
        self._M_csr = self._C_csr.copy()
        self._f_glob = np.zeros(self._C_csr.shape[1])
        # TODO: CAREFUL! THE FOLLOWING LINE WILL NOT WORK IT IS JUST FOR THE CURRENT MERGE REQUEST
        self._constraints.update(self._mapping.no_of_dofs)

    # -- ASSIGN NEUMANN CONDITION METHODS -----------------------------------------------------------------
    def assign_neumann_condition(self, condition, property_names, tag='_groups', name='Unknown'):
        if tag == '_groups':
            eleids = self._mesh.get_elementids_by_groups(property_names)
        elif tag == '_eleids':
            eleids = property_names
        else:
            eleids = self._mesh.get_elementids_by_tags(property_names)
            
        # get ele_shapes of the elements belonging to the passed eleidxes
        ele_shapes = self._mesh.get_ele_shapes_by_elementids(eleids)
            
        self._neumann.assign_neumann_by_eleids(condition, eleids, ele_shapes, property_names, tag, name)

    # -- ASSIGN CONSTRAINTS METHODS ------------------------------------------------------------------------

    # -- MAPPING METHODS -----------------------------------------------------------------------------------
    def _update_mapping(self):
        # collect parameters for call of update_mapping
        fields = self._fields
        nodeids = self._mesh.nodes_df.index.values
        elementids_and_connectivity = self._ele_obj_df[self._ele_obj_df.notna().values].join(self._mesh.el_df)['connectivity']
        dofs_by_element = [element.dofs() for element in self._ele_obj_df[self._ele_obj_df.notna().values]['ele_obj'].values]
        # call update_mapping
        self._mapping.update_mapping(fields, nodeids, elementids_and_connectivity.index,
                                     elementids_and_connectivity.values, dofs_by_element)

    # -- GETTER FOR SYSTEM MATRICES ------------------------------------------------------------------------
    #
    # MUST BE IMPLEMENTED IN SUBCLASSES
    #
