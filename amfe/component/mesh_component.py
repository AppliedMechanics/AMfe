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
from .component_base import ComponentBase
from amfe.component.constants import ELEPROTOTYPEHELPERLIST

__all__ = ['MeshComponent']


class MeshComponent(ComponentBase):
    # The following class attributes must be overwritten by subclasses
    ELEMENTPROTOTYPES = dict(((element[0], None) for element in ELEPROTOTYPEHELPERLIST))
    BOUNDARYELEMENTFACTORY = dict(((element[0], None) for element in ELEPROTOTYPEHELPERLIST))

    def __init__(self, mesh=Mesh()):
        super().__init__()
        self._mesh = mesh
        no_of_volume_elements = mesh.no_of_elements
        indices = mesh.el_df[mesh.el_df['is_boundary'] != True].index
        self._ele_obj_df = pd.DataFrame([None]*no_of_volume_elements, index=indices, columns=['ele_obj'])

        # Neumann Properties
        # ------------------
        # Dataframe for reconstructing applied conditions
        self._neumann_df = pd.DataFrame(columns=['name', 'tag', 'property_names', 'function', 'direction',
                                                 'shadow_area'])
        # Dataframe containing element_objects and their position in connectivity array and their
        # foreign key to the _neumann_df they belong to
        self._neumann_obj_df = pd.DataFrame(columns=['ele_obj', 'connectivity_idx', 'fk_neumann_df'])
        self._neumann_obj_df['fk_neumann_df'] = self._neumann_obj_df['fk_neumann_df'].astype(int)
        self._neumann_obj_df['connectivity_idx'] = self._neumann_obj_df['connectivity_idx'].astype(int)

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
        elif tag == '_eleidxs':
            eleids = self._mesh.get_elementids_by_elementiloc(propertynames)
        else:
            eleids = self._mesh.get_elementids_by_tags(propertynames)
        self._assign_material_by_eleids(materialobj, eleids)

    def _assign_material_by_eleids(self, materialobj, eleids):
        prototypes = deepcopy(self.ELEMENTPROTOTYPES)
        for prototype in prototypes.values():
            prototype.material = materialobj
        ele_shapes = self._mesh.get_ele_shapes_by_ids(eleids)
        self._ele_obj_df.loc[eleids, 'ele_obj'] = [prototypes[ele_shape] for ele_shape in ele_shapes]

    # -- ASSIGN NEUMANN CONDITION METHODS -----------------------------------------------------------------
    def assign_neumann_condition(self, val, direction, property_names, tag='_groups', shadow_area=False,
                                 name='Unknown'):
        if tag == '_groups':
            eleidxes = self._mesh.get_elementiloc_by_groups(property_names)
        elif tag == '_eleidxs':
            eleidxes = property_names
        else:
            eleidxes = self._mesh.get_elementidxs_by_tags(property_names)
        dfindex = self._neumann_df.index.max() + 1
        if pd.isnull(dfindex):
            dfindex = 0
        self._add_neumann_condition_by_eleidxs(val, direction, eleidxes, shadow_area, dfindex)
        df_data = {'name': name, 'tag': tag, 'property_names': [property_names], 'function': val,
                   'direction': [direction], 'shadow_area': shadow_area}
        self._neumann_df = self._neumann_df.append(pd.DataFrame(df_data, index=[dfindex]), sort=True)

    def _add_neumann_condition_by_eleidxs(self, val, direction, eleidxes, shadow_area, index):
        #
        # extends _neumann_obj_df by new b.c. with Neumann_elements
        #

        # Create prototypes for each boundary element shape
        neumann_ele_prototypes = deepcopy(self.BOUNDARYELEMENTFACTORY)
        for element_shape in neumann_ele_prototypes:
            # create prototype for current element_shape
            neumann_ele_prototypes[element_shape] = neumann_ele_prototypes[element_shape](val=1.0,
                                                                                          time_func=val,
                                                                                          direct=direction,
                                                                                          shadow_area=shadow_area)
        # get ele_shapes of the elements belonging to the passed eleidxes
        ele_shapes = self._mesh.get_ele_shapes_by_elementiloc(eleidxes)
        # create pointers to eleobjects
        neumann_ele_objects = np.array([neumann_ele_prototypes[ele_shape] for ele_shape in ele_shapes],
                                       dtype=object)
        # add tuple (eleidxes, ele_object) to neumann_ele_obj
        df = pd.DataFrame(
            {'ele_obj': neumann_ele_objects, 'connectivity_idx': eleidxes,
             'fk_neumann_df': np.ones(len(neumann_ele_objects), dtype=int) * index}
        )
        self._neumann_obj_df = self._neumann_obj_df.append(df)

    # -- ASSIGN CONSTRAINTS METHODS ------------------------------------------------------------------------

    # -- MAPPING METHODS -----------------------------------------------------------------------------------

    # -- GETTER FOR SYSTEM MATRICES ------------------------------------------------------------------------
    #
    # MUST BE IMPLEMENTED IN SUBCLASSES
    #
