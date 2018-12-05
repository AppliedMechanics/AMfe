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
        self._ele_obj_df = pd.DataFrame([], columns=['physics', 'fk_mesh', 'ele_obj', 'fk_mapping'])
        self._ele_obj_df['fk_mapping'] = self._ele_obj_df['fk_mapping'].astype(int)
        self._ele_obj_df['fk_mesh'] = self._ele_obj_df['fk_mesh'].astype(int)
        self._ele_obj_df = self._ele_obj_df.set_index(['physics', 'fk_mesh'])
        self._neumann = NeumannManager()

    # -- PROPERTIES --------------------------------------------------------------------------------------
    @property
    def ele_obj(self):
        return self._ele_obj_df['ele_obj'].values

    # -- ASSIGN MATERIAL METHODS -------------------------------------------------------------------------
    def assign_material(self, materialobj, propertynames, physics, tag='_groups'):
        if tag == '_groups':
            eleids = self._mesh.get_elementids_by_groups(propertynames)
        elif tag == '_eleids':
            eleids = propertynames
        else:
            eleids = self._mesh.get_elementids_by_tags(propertynames)
        self._assign_material_by_eleids(materialobj, eleids, physics)

    def _assign_material_by_eleids(self, materialobj, eleids, physics):
        prototypes = deepcopy(self.ELEMENTPROTOTYPES)
        for prototype in prototypes.values():
            prototype.material = materialobj
        ele_shapes = self._mesh.get_ele_shapes_by_ids(eleids)
        new_df = pd.DataFrame({'physics': [physics]*len(ele_shapes), 'fk_mesh': eleids,
                               'ele_obj': [prototypes[ele_shape] for ele_shape in ele_shapes],
                               'fk_mapping': np.ones(len(ele_shapes), dtype=int)*-1})
        new_df = new_df.set_index(['physics', 'fk_mesh'])
        self._ele_obj_df = new_df.combine_first(self._ele_obj_df)
        self._ele_obj_df = self._ele_obj_df.sort_index()
        self._ele_obj_df['fk_mapping'] = self._ele_obj_df['fk_mapping'].astype(int)
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
        self._update_mapping()

    # -- ASSIGN CONSTRAINTS METHODS ------------------------------------------------------------------------

    # -- MAPPING METHODS -----------------------------------------------------------------------------------
    def _update_mapping(self):
        # collect parameters for call of update_mapping
        fields = self._fields
        nodeids = self._mesh.nodes_df.index.values

        volume_connectivites = self._ele_obj_df.join(self._mesh.el_df, on='fk_mesh')['connectivity'].values
        dofs_by_elements = [element.dofs() for element in self._ele_obj_df['ele_obj'].values]
        volume_callbacks = [self.write_mapping_key]*len(dofs_by_elements)
        volume_callbackargs = self._ele_obj_df.index.get_values()

        boundary_connectivities = self._neumann.el_df.join(self._mesh.el_df, on='fk_mesh')['connectivity'].values
        dofs_by_elements_boundary = [element.dofs() for element in self._neumann.el_df['neumann_obj'].values]
        boundary_callbacks = [self._neumann.write_mapping_key]*len(dofs_by_elements_boundary)
        boundary_callbackargs = self._neumann.el_df.index.get_values()

        connectivities = np.concatenate([volume_connectivites, boundary_connectivities])
        dofs_by_elements.extend(dofs_by_elements_boundary)
        callbacks = np.concatenate([volume_callbacks, boundary_callbacks])
        callbackargs = np.concatenate([volume_callbackargs, boundary_callbackargs])

        # call update_mapping
        self._mapping.update_mapping(fields, nodeids, connectivities, dofs_by_elements, callbacks, callbackargs)

    def write_mapping_key(self, fk, local_id):
        self._ele_obj_df.loc[local_id, 'fk_mapping'] = fk
    # -- GETTER FOR SYSTEM MATRICES ------------------------------------------------------------------------
    #
    # MUST BE IMPLEMENTED IN SUBCLASSES
    #
